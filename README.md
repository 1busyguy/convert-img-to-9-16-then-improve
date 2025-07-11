# Kontext → Modify → Realism Batch Image Editor

A Streamlit app for batch image expansion, enhancement, and realism using FAL AI APIs.

## Features

- Batch upload and process images through three AI-powered steps:
  - **Kontext:** Outpaint (expand) the image.
  - **Modify:** Polish and unify quality and color.
  - **Realism:** Enhance facial and skin realism.
- Adjustable controls for each stage.
- Download per-image or all results as zip files.

## Setup

### 1. Clone the repository

```bash
git clone <your_repo_url>
cd <your_repo_folder>
```

### 2. Environment variables

Copy `.env.example` to `.env` and add your FAL AI API Key:

```ini
FAL_KEY=your_actual_fal_api_key
```

### 3. Install dependencies

Install Python dependencies (Python 3.8+):

```bash
pip install streamlit pillow python-dotenv requests nest_asyncio fal_client
```

### 4. Run the app

```bash
streamlit run main.py
```

## Usage

1. **Upload your images** (PNG or JPG).
2. **Adjust settings** for each stage as desired.
3. **Click the process button** to batch run all stages.
4. **Preview and download outputs** for each image, or all outputs as a single zip.

## Output

Processed images and zip files are saved in the `completed/` directory.

## .env

Your `.env` file should look like:

```ini
FAL_KEY=your_actual_fal_api_key
```

## Notes

- Requires an internet connection.
- This app uses FAL’s cloud image models—API key required.

---
