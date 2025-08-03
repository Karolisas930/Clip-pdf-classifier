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

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Save PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Convert first page to image
    doc = fitz.open("temp.pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    image_path = "preview.png"
    pix.save(image_path)

    # Display preview
    st.image(image_path, caption="Preview of PDF")

    # OCR text
    ocr_text = extract_text_from_image(image_path)
    st.subheader("📝 Extracted OCR Text:")
    st.write(ocr_text)

    # Load labels from hierarchy.csv
    try:
        df = pd.read_csv("hierarchy.csv")

        # Combine hierarchy to a readable label
        df.fillna("", inplace=True)
        df['full_label'] = df['section'].astype(str) + " > " + df['category'].astype(str) + " > " + df['subcategory'].astype(str)
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
