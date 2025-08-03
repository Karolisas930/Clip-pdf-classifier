import subprocess
subprocess.run(["pip", "install", "git+https://github.com/openai/CLIP.git"], check=True)

import streamlit as st
import torch
import clip
from PIL import Image
import os
import fitz # PyMuPDF
import pandas as pd

from ocr_utils import extract_text_from_image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

st.title("📄 Smart PDF Classifier with CLIP")

st.set_page_config(page_title="Smart PDF/Image Classifier", layout="wide")
st.title("📄 Smart PDF/Image Classifier with CLIP")

# ——— Single, multi-type uploader ———
uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg"],
    key="uploader_mixed",
)

# If nothing uploaded, show info and stop here
if not uploaded:
    st.info("Please upload a PDF or image file to classify.")
    st.stop()

# ——— Branch by MIME type ———
if uploaded.type == "application/pdf":
    # save PDF to disk
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.getbuffer())

    # render first page from PDF
    doc = fitz.open("temp.pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    image_path = "preview.png"
    pix.save(image_path)
else:
    # handle image upload directly
    image = Image.open(uploaded)
    image_path = "preview.png"
    image.save(image_path)

# ——— Now both branches set `image_path` ———
# Display preview
st.image(image_path, caption="Preview", use_column_width=True)

# OCR
st.subheader("📝 Extracted OCR Text")
ocr_text = extract_text_from_image(image_path)
st.write(ocr_text)

# …followed by your CLIP encoding/classification…

    # Load labels from hierarchy.csv


# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Streamlit UI
st.set_page_config(page_title="PDF Classifier", layout="wide")
st.title("📄 Smart PDF Classifier")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save the uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Convert first page to image
    doc = fitz.open("temp.pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    image_path = "preview.png"
    pix.save(image_path)

    # Extract text from image (OCR)
    st.image(image_path, caption="Preview of First Page", use_column_width=True)
    ocr_text = extract_text_from_image(image_path)
    st.subheader("🔍 Extracted OCR Text")
    st.write(ocr_text)

    # Load Hierarchical Labels
    df_sections = pd.read_csv("Industry-Grid view.csv")
    df_categories = pd.read_csv("Service Category-Grid view.csv")
    df_subcategories = pd.read_csv("Specialization-Grid view.csv")

    # Merge and Create Hierarchical Labels
    merged = df_subcategories.merge(df_categories, left_on='category_id', right_on='id', suffixes=('_subcat', '_cat'))
    merged = merged.merge(df_sections, left_on='section_id', right_on='id', suffixes=('', '_sec'))

    merged['full_label'] = merged['name_sec'] + " > " + merged['name_cat'] + " > " + merged['name_subcat']

    labels = merged['full_label'].tolist()
    text_tokens = clip.tokenize(labels).to(device)

    # Image Feature Extraction
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        logits_per_image, _ = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    st.subheader("📌 Classification Results")
    top_results = sorted(zip(labels, probs[0]), key=lambda x: x[1], reverse=True)[:5]

    for label, prob in top_results:
        st.write(f"**{label}** — {prob*100:.2f}%")
    try:
        df = pd.read_csv("hierarchy.csv")

        # Combine hierarchy to a readable label
        df.fillna("", inplace=True)
        df['full_label'] = df['Industry'].astype(str) + " > " + df['Service Category'].astype(str) + " > " + df['Specialization'].astype(str)
        labels = df['full_label'].dropna().unique().tolist()
    except Exception as e:
        st.error(f"Failed to load hierarchy.csv: {e}")
        labels = []

    if labels:
        # Process image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(labels).to(device)

        # CLIP prediction
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)
            logits_per_image, _ = model(image, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Show results
        st.subheader("🔍 Prediction Results:")
        for label, prob in zip(labels, probs[0]):
            if prob > 0.5:
                st.write(f"**{label}** — {prob:.2%}")
    else:
        st.warning("No valid labels loaded for prediction.")
