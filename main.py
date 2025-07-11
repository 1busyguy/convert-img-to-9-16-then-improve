import streamlit as st
from PIL import Image
import fal_client
import asyncio
import nest_asyncio
nest_asyncio.apply()
import io
import tempfile
import requests
import os
import zipfile
from dotenv import load_dotenv

# --- Load environment variables from .env ---
load_dotenv()
FAL_KEY = os.getenv("FAL_KEY")
if not FAL_KEY or FAL_KEY == "YOUR_FAL_API_KEY":
    st.error("API key missing! Please set FAL_KEY in your .env file.")
    st.stop()

COMPLETED_DIR = "completed"
os.makedirs(COMPLETED_DIR, exist_ok=True)

st.set_page_config(page_title="Kontext + Modify + Realism Batch Edit", layout="centered")
st.title("Kontext → Modify → Realism (FAL AI Pipeline)")

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

def safe_dir_name(name):
    return "".join(c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in name)

def upload_image_get_url(image: Image.Image):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
    url = asyncio.run(fal_client.upload_file_async(tmp_path))
    return url

# Kontext defaults
DEFAULT_KONTEXT_PROMPT = (
    "Outpaint the image, filling all white areas with a seamless continuation that matches the original scene’s colors, style, and lighting. Subtly enhance the whole image for detail and vibrancy, while keeping it natural and consistent."
)
DEFAULT_KONTEXT_STEPS = 50
DEFAULT_KONTEXT_GUIDANCE = 8.0
DEFAULT_KONTEXT_NUM_IMAGES = 1
DEFAULT_KONTEXT_SAFETY = False
DEFAULT_KONTEXT_OUT_FORMAT = "png"
DEFAULT_KONTEXT_RES_MODE = "match_input"

# Modify defaults
DEFAULT_MODIFY_PROMPT = "Sharpen and upscale the image to higher resolution. Improve clarity, enhance fine details, and make colors vibrant and cohesive, while keeping the original content, facial features, and overall style unchanged. Ensure the result looks natural and flawless."
DEFAULT_MODIFY_STRENGTH = 0.2
DEFAULT_MODIFY_ASPECT = "9:16"

# Realism defaults
DEFAULT_REALISM_GUIDANCE = 4.0
DEFAULT_REALISM_STEPS = 40
DEFAULT_REALISM_SAFETY = False
DEFAULT_REALISM_LORA = .40

async def call_fal_kontext(
    image_url,
    prompt=DEFAULT_KONTEXT_PROMPT,
    num_inference_steps=DEFAULT_KONTEXT_STEPS,
    guidance_scale=DEFAULT_KONTEXT_GUIDANCE,
    num_images=DEFAULT_KONTEXT_NUM_IMAGES,
    enable_safety_checker=DEFAULT_KONTEXT_SAFETY,
    output_format=DEFAULT_KONTEXT_OUT_FORMAT,
    resolution_mode=DEFAULT_KONTEXT_RES_MODE,
):
    handler = await fal_client.submit_async(
        "fal-ai/flux-kontext/dev",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
            "resolution_mode": resolution_mode,
        }
    )
    result = await handler.get()
    return result

async def call_fal_modify(
    image_url,
    prompt=DEFAULT_MODIFY_PROMPT,
    strength=DEFAULT_MODIFY_STRENGTH,
    aspect_ratio=DEFAULT_MODIFY_ASPECT,
):
    handler = await fal_client.submit_async(
        "fal-ai/luma-photon/modify",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "strength": strength,
            "aspect_ratio": aspect_ratio,
        }
    )
    result = await handler.get()
    return result

async def call_fal_realism(
    image_url,
    guidance_scale=DEFAULT_REALISM_GUIDANCE,
    num_inference_steps=DEFAULT_REALISM_STEPS,
    enable_safety_checker=DEFAULT_REALISM_SAFETY,
    lora_scale=DEFAULT_REALISM_LORA,
):
    handler = await fal_client.submit_async(
        "fal-ai/image-editing/realism",
        arguments={
            "image_url": image_url,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "enable_safety_checker": enable_safety_checker,
            "lora_scale": lora_scale,
        }
    )
    result = await handler.get()
    return result

def download_and_display_image(img_url):
    img_bytes = requests.get(img_url).content
    result_img = Image.open(io.BytesIO(img_bytes))
    return result_img, img_bytes

uploaded_files = st.file_uploader(
    "Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

st.subheader("Kontext (Expansion) Prompt")
kont_prompt = st.text_area("Prompt for Kontext", value=DEFAULT_KONTEXT_PROMPT)
kont_steps = st.number_input("Kontext: Num inference steps", min_value=10, max_value=100, value=DEFAULT_KONTEXT_STEPS)
kont_guidance = st.slider("Kontext: Guidance scale", 1.0, 20.0, value=DEFAULT_KONTEXT_GUIDANCE, step=0.5)
kont_num_images = st.number_input("Kontext: Num images per input", min_value=1, max_value=2, value=DEFAULT_KONTEXT_NUM_IMAGES)
kont_safety = st.checkbox("Kontext: Enable Safety Checker", value=DEFAULT_KONTEXT_SAFETY)
kont_out_format = st.selectbox("Kontext: Output format", options=["png", "jpeg"], index=0)
kont_res_mode = st.selectbox("Kontext: Resolution mode", options=["match_input", "auto", "9:16", "16:9", "1:1", "4:5"], index=0)

st.subheader("Modify (Polish) Prompt")
mod_prompt = st.text_area("Prompt for Modify", value=DEFAULT_MODIFY_PROMPT)
mod_strength = st.slider("Modify: Strength", 0.0, 1.0, value=DEFAULT_MODIFY_STRENGTH, step=0.1)
mod_aspect = st.selectbox("Modify: Aspect Ratio", options=["9:16", "16:9", "1:1", "4:3", "3:4", "21:9", "9:21"], index=0)

st.subheader("Realism (Facial/Skin Enhance)")
realism_guidance = st.slider("Realism: Guidance scale", 0.0, 20.0, value=DEFAULT_REALISM_GUIDANCE, step=0.5)
realism_steps = st.number_input("Realism: Num inference steps", min_value=10, max_value=80, value=DEFAULT_REALISM_STEPS)
realism_safety = st.checkbox("Realism: Enable Safety Checker", value=DEFAULT_REALISM_SAFETY)
realism_lora = st.slider("Realism: LoRA scale", 0.0, 2.0, value=DEFAULT_REALISM_LORA, step=0.1)

if uploaded_files and st.button("Batch Expand + Polish + Realism Enhance!"):
    st.info("Processing images. This will take time per batch...")

    for idx, uploaded in enumerate(uploaded_files):
        fname = os.path.splitext(os.path.basename(uploaded.name))[0]
        safe_name = safe_dir_name(fname)
        img_dir = os.path.join(COMPLETED_DIR, safe_name)
        flux_dir = os.path.join(img_dir, "Flux")
        modify_dir = os.path.join(img_dir, "Modify")
        realism_dir = os.path.join(img_dir, "Realism")
        os.makedirs(flux_dir, exist_ok=True)
        os.makedirs(modify_dir, exist_ok=True)
        os.makedirs(realism_dir, exist_ok=True)

        # Step 0: Original
        orig_img = Image.open(uploaded).convert("RGB")
        centered_img = center_image_on_canvas(orig_img)
        orig_path = os.path.join(img_dir, f"original_{fname}.png")
        centered_img.save(orig_path)

        kont_img_url = upload_image_get_url(centered_img)

        # Step 1: Kontext (Flux)
        with st.spinner(f"Kontext generating for {uploaded.name}..."):
            kont_result = asyncio.run(
                call_fal_kontext(
                    image_url=kont_img_url,
                    prompt=kont_prompt,
                    num_inference_steps=kont_steps,
                    guidance_scale=kont_guidance,
                    num_images=kont_num_images,
                    enable_safety_checker=kont_safety,
                    output_format=kont_out_format,
                    resolution_mode=kont_res_mode,
                )
            )
        try:
            kont_img_url_result = kont_result["images"][0]["url"]
            kont_img, kont_bytes = download_and_display_image(kont_img_url_result)
            # Save Flux output in both main dir and Flux subdir
            flux_outfile_main = os.path.join(img_dir, f"flux_{fname}.{kont_out_format}")
            flux_outfile = os.path.join(flux_dir, f"flux_{fname}.{kont_out_format}")
            with open(flux_outfile_main, "wb") as f:
                f.write(kont_bytes)
            with open(flux_outfile, "wb") as f:
                f.write(kont_bytes)
        except Exception as e:
            st.error(f"Kontext failed: {e}")
            continue

        # Step 2: Modify
        kont_url_for_modify = upload_image_get_url(kont_img)
        with st.spinner(f"Modify polishing for {uploaded.name}..."):
            mod_result = asyncio.run(
                call_fal_modify(
                    image_url=kont_url_for_modify,
                    prompt=mod_prompt,
                    strength=mod_strength,
                    aspect_ratio=mod_aspect,
                )
            )
        try:
            mod_img_url_result = mod_result["images"][0]["url"]
            mod_img, mod_bytes = download_and_display_image(mod_img_url_result)
            # Save Modify output in both main dir and Modify subdir
            modify_outfile_main = os.path.join(img_dir, f"modify_{fname}.{kont_out_format}")
            modify_outfile = os.path.join(modify_dir, f"modify_{fname}.{kont_out_format}")
            with open(modify_outfile_main, "wb") as f:
                f.write(mod_bytes)
            with open(modify_outfile, "wb") as f:
                f.write(mod_bytes)
        except Exception as e:
            st.error(f"Modify failed: {e}")
            continue

        # Step 3: Realism
        mod_url_for_realism = upload_image_get_url(mod_img)
        with st.spinner(f"Realism facial/skin enhance for {uploaded.name}..."):
            realism_result = asyncio.run(
                call_fal_realism(
                    image_url=mod_url_for_realism,
                    guidance_scale=realism_guidance,
                    num_inference_steps=realism_steps,
                    enable_safety_checker=realism_safety,
                    lora_scale=realism_lora,
                )
            )
        try:
            realism_img_url_result = realism_result["images"][0]["url"]
            realism_img, realism_bytes = download_and_display_image(realism_img_url_result)
            # Save Realism output in both main dir and Realism subdir
            realism_outfile_main = os.path.join(img_dir, f"realism_{fname}.{kont_out_format}")
            realism_outfile = os.path.join(realism_dir, f"realism_{fname}.{kont_out_format}")
            with open(realism_outfile_main, "wb") as f:
                f.write(realism_bytes)
            with open(realism_outfile, "wb") as f:
                f.write(realism_bytes)
        except Exception as e:
            st.error(f"Realism failed: {e}")
            continue

        st.success(f"Saved all stages to {img_dir}")

        # Display all results side-by-side (suppress errors)
        try:
            cols = st.columns(4)
            with cols[0]:
                st.image(orig_img, caption=f"Original: {uploaded.name}", use_container_width=True)
            with cols[1]:
                st.image(kont_img, caption=f"Kontext Output", use_container_width=True)
            with cols[2]:
                st.image(mod_img, caption=f"Modify Final", use_container_width=True)
            with cols[3]:
                st.image(realism_img, caption=f"Realism Enhanced", use_container_width=True)
        except Exception:
            pass
    # Gather all realism-enhanced images for zip download
    if 'realism_zip_bytes' not in st.session_state:
        st.session_state.realism_zip_bytes = []

    # Save zip just for realism-enhanced outputs
    realism_zip_path = os.path.join(img_dir, f"{safe_name}_realism_only.zip")
    with zipfile.ZipFile(realism_zip_path, 'w') as zipf:
        zipf.write(orig_path, arcname=os.path.basename(orig_path))
        zipf.write(flux_outfile_main, arcname=os.path.basename(flux_outfile_main))
        zipf.write(modify_outfile_main, arcname=os.path.basename(modify_outfile_main))
        zipf.write(realism_outfile_main, arcname=os.path.basename(realism_outfile_main))
    with open(realism_zip_path, "rb") as f:
        realism_zip_bytes = f.read()
        st.download_button(
            f"Download all 4 images for {fname} as zip",
            data=realism_zip_bytes,
            file_name=f"{safe_name}_all_outputs.zip"
        )
        st.session_state.realism_zip_bytes.append((fname, realism_zip_bytes))

# (Optional) Download all results for all images in a single ZIP
if uploaded_files:
    # Gather all per-image folders into a global zip
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        with zipfile.ZipFile(tmp_zip, 'w') as zf:
            for uploaded in uploaded_files:
                fname = os.path.splitext(os.path.basename(uploaded.name))[0]
                safe_name = safe_dir_name(fname)
                img_dir = os.path.join(COMPLETED_DIR, safe_name)
                for root, dirs, files in os.walk(img_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, COMPLETED_DIR)
                        zf.write(file_path, arcname=arcname)
        tmp_zip.seek(0)
        zip_bytes = tmp_zip.read()
    st.download_button(
        "Download ALL outputs (all images/folders, zip)",
        data=zip_bytes,
        file_name="all_results.zip"
    )
