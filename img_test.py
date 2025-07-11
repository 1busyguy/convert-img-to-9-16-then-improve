import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import fal_client
import asyncio
import nest_asyncio
nest_asyncio.apply()
import io
import tempfile
import requests
import os
import zipfile
import time
import numpy as np
import random
import traceback
from dotenv import load_dotenv

load_dotenv()
FAL_KEY = os.getenv("FAL_KEY")
if not FAL_KEY or FAL_KEY == "YOUR_FAL_API_KEY":
    st.error("API key missing! Please set FAL_KEY in your .env file.")
    st.stop()

def center_image_on_canvas(image, canvas_size=(1080, 1920), fill_color=(255, 255, 255)):
    canvas = Image.new("RGB", canvas_size, fill_color)
    img_w, img_h = image.size
    can_w, can_h = canvas_size
    scale = min(can_w / img_w, can_h / img_h, 1.0)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    offset = ((can_w - new_w) // 2, (can_h - new_h) // 2)
    canvas.paste(image, offset)
    return canvas

def upload_image_get_url(image: Image.Image):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
    url = asyncio.run(fal_client.upload_file_async(tmp_path))
    return url

async def call_fal_kontext(image_url, prompt, num_inference_steps, guidance_scale):
    handler = await fal_client.submit_async(
        "fal-ai/flux-kontext/dev",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": 1,
            "enable_safety_checker": False,
            "output_format": "png",
            "resolution_mode": "match_input",
        }
    )
    return await handler.get()

async def call_fal_modify(image_url, prompt, strength):
    handler = await fal_client.submit_async(
        "fal-ai/luma-photon/modify",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "strength": strength,
            "aspect_ratio": "9:16",
        }
    )
    return await handler.get()

async def call_fal_realism(image_url, guidance_scale, lora_scale):
    handler = await fal_client.submit_async(
        "fal-ai/image-editing/realism",
        arguments={
            "image_url": image_url,
            "guidance_scale": guidance_scale,
            "num_inference_steps": 40,
            "enable_safety_checker": False,
            "lora_scale": lora_scale,
        }
    )
    return await handler.get()

def download_image(img_url):
    img_bytes = requests.get(img_url).content
    img = Image.open(io.BytesIO(img_bytes))
    return img, img_bytes

def combine_images_with_settings(images, settings, font_path=None):
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    font_size = 28

    # Try to load a truetype font if available, else use default
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Calculate height for settings text (multiline)
    text_heights = []
    for text in settings:
        lines = text.count('\n') + 1
        text_heights.append(lines * (font_size + 2))
    max_text_height = max(text_heights) + 10

    total_width = sum(widths)
    total_height = max(heights) + max_text_height + 10

    result = Image.new('RGB', (total_width, total_height), (245,245,245))
    draw = ImageDraw.Draw(result)

    x_offset = 0
    for idx, (img, text) in enumerate(zip(images, settings)):
        result.paste(img, (x_offset, 0))
        # Calculate size of text block using multiline_textbbox
        bbox = draw.multiline_textbbox((0,0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        img_cx = x_offset + (img.size[0] - text_width) // 2
        text_y = img.size[1] + 6
        draw.multiline_text((img_cx, text_y), text, fill=(60,60,60), font=font, align="center")
        x_offset += img.size[0]
    return result

st.title("Best Result Finder (10 Strict, Randomized, Well-Spaced Variants, All Settings Shown)")

uploaded = st.file_uploader("Upload a source image", type=["jpg", "jpeg", "png"])

DEFAULT_PROMPT = (
    "Expand this image by replacing all white space with a seamless continuation of the scene, "
    "matching the colors, style, and lighting of the original, so the new areas look naturally integrated."
)
MODIFY_PROMPT = "Enhance image quality, unify style and make colors vibrant without changing content."

if uploaded and st.button("Run Sweep!"):
    orig_img = Image.open(uploaded).convert("RGB")
    centered_img = center_image_on_canvas(orig_img)
    base_name = os.path.splitext(os.path.basename(uploaded.name))[0]

    # Create unique subfolder for this test in COMPLETED directory
    COMPLETED_DIR = "completed"
    os.makedirs(COMPLETED_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    TEST_FOLDER = os.path.join(COMPLETED_DIR, f"{base_name}_{ts}")
    os.makedirs(TEST_FOLDER, exist_ok=True)

    # Save original
    orig_save_path = os.path.join(TEST_FOLDER, f"original.png")
    centered_img.save(orig_save_path)

    images = [centered_img]
    settings_list = ["Original\n(Flux CFG: —\nModify Strength: —\nRealism CFG: —\nRealism Lora: —)"]

    # Save settings for original
    with open(os.path.join(TEST_FOLDER, f"original.txt"), "w") as f:
        f.write("Original\n(Flux CFG: —, Modify Strength: —, Realism CFG: —, Realism Lora: —)")

    # STRICT allowed parameter lists
    flux_cfg_values = [round(x, 1) for x in np.arange(1.0, 20.1, 0.5)]
    modify_strength_values = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
    realism_cfg_values = [round(x, 1) for x in np.arange(0.0, 20.1, 0.5)]
    realism_lora_values = [round(x, 1) for x in np.arange(0.0, 2.01, 0.1)]

    num_variants = 10

    def spaced_variants(lst, num):
        idxs = np.linspace(0, len(lst)-1, num=num, dtype=int)
        out = [lst[i] for i in idxs]
        random.shuffle(out)
        return out

    flux_cfg_variants = spaced_variants(flux_cfg_values, num_variants)
    modify_strength_variants = spaced_variants(modify_strength_values, num_variants)
    realism_cfg_variants = spaced_variants(realism_cfg_values, num_variants)
    realism_lora_variants = spaced_variants(realism_lora_values, num_variants)

    for i in range(num_variants):
        flux_cfg = float(flux_cfg_variants[i])
        modify_strength = float(modify_strength_variants[i])
        realism_cfg = float(realism_cfg_variants[i])
        realism_lora = float(realism_lora_variants[i])

        with st.spinner(f"Variant {i+1}: Flux CFG={flux_cfg:.1f}, Modify Strength={modify_strength:.1f}, Realism CFG={realism_cfg:.1f}, Realism Lora={realism_lora:.1f}"):
            # Kontext/Flux step
            try:
                kont_url = upload_image_get_url(centered_img)
                kont_res = asyncio.run(call_fal_kontext(
                    kont_url, DEFAULT_PROMPT, 50, flux_cfg))
                kont_img_url = kont_res["images"][0]["url"]
                kont_img, _ = download_image(kont_img_url)
            except Exception as e:
                st.error(f"Kontext API failed on Variant {i+1} (Flux CFG={flux_cfg:.1f}): {e}\n{traceback.format_exc()}")
                continue

            # Modify step
            st.write(f"Calling Modify with: URL={kont_url}, strength={modify_strength}, prompt='{MODIFY_PROMPT}'")
            try:
                mod_url = upload_image_get_url(kont_img)
                mod_res = asyncio.run(call_fal_modify(
                    mod_url, MODIFY_PROMPT, modify_strength))
                mod_img_url = mod_res["images"][0]["url"]
                mod_img, _ = download_image(mod_img_url)
            except Exception as e:
                st.error(f"Full Modify error: {e}\nInput URL: {mod_url}\nStrength: {modify_strength}\nPrompt: {MODIFY_PROMPT}\n{traceback.format_exc()}")
                continue

            # Realism step
            try:
                real_url = upload_image_get_url(mod_img)
                real_res = asyncio.run(call_fal_realism(
                    real_url, realism_cfg, realism_lora))
                real_img_url = real_res["images"][0]["url"]
                real_img, _ = download_image(real_img_url)
            except Exception as e:
                st.error(f"Realism API failed on Variant {i+1} (CFG={realism_cfg:.1f}, Lora={realism_lora:.1f}): {e}\n{traceback.format_exc()}")
                continue

        images.append(real_img)
        config_text = (
            f"Variant {i+1}\n"
            f"Flux CFG: {flux_cfg:.1f}\n"
            f"Modify Strength: {modify_strength:.1f}\n"
            f"Realism CFG: {realism_cfg:.1f}\n"
            f"Realism Lora: {realism_lora:.1f}"
        )
        settings_list.append(config_text)

        # Save image and config to test subfolder
        variant_img_path = os.path.join(TEST_FOLDER, f"variant_{i+1}.png")
        real_img.save(variant_img_path)
        with open(os.path.join(TEST_FOLDER, f"variant_{i+1}.txt"), "w") as f:
            f.write(config_text)

    # Combine all images horizontally with settings under each
    combined_img = combine_images_with_settings(images, settings_list)
    combined_img_path = os.path.join(TEST_FOLDER, "combined_results.png")
    combined_img.save(combined_img_path)

    st.subheader("Comparison Sheet")
    st.image(combined_img, caption="All Results Side by Side (with settings)")

    # ZIP download button for all test results in this run
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        with zipfile.ZipFile(tmp_zip, 'w') as zf:
            for fname in os.listdir(TEST_FOLDER):
                zf.write(os.path.join(TEST_FOLDER, fname), arcname=fname)
        tmp_zip.seek(0)
        zip_bytes = tmp_zip.read()
    st.download_button("Download This Test's Results (ZIP)", data=zip_bytes, file_name=f"{base_name}_{ts}_results.zip")
