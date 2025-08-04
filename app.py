import os
import streamlit as st
import torch
import clip
from PIL import Image
import fitz # PyMuPDF
import pandas as pd
import numpy as np
from ocr_utils import extract_text_from_image

# ── 1) Must be the very first Streamlit command ──
st.set_page_config(
    page_title="Smart PDF/Image Classifier",
    layout="wide",
)

st.title("📄🔍 Smart PDF/Image Classifier with CLIP")

# ── 2) Quick sanity check for NumPy ──
st.markdown(f"✅ **NumPy** imported, version: `{np.__version__}`")

# ── 3) Load & cache your hierarchy labels ──
uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg"],
    key="uploader_mixed",
)
if not uploaded:
    st.info("Please upload a PDF or image file to classify.")
    st.stop()

# ── 4) Load CLIP + tokenize/encode your labels once ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

# truncate long labels so they fit in CLIP’s 77‐token window
TEXT_TOKENS = clip.tokenize(LABELS, truncate=True).to(DEVICE)
with torch.no_grad():
    TEXT_FEATURES = MODEL.encode_text(TEXT_TOKENS)

# ── 5) File uploader ──
uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg"],
    key="uploader_mixed",
)
if not uploaded:
    st.info("Please upload a PDF or image file to classify.")
    st.stop()

# ── 6) PDF → first‐page preview or image → save as preview.png ──
if uploaded.type == "application/pdf":
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.getbuffer())
    doc = fitz.open("temp.pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    image_path = "preview.png"
    pix.save(image_path)
else:
    image = Image.open(uploaded)
    image_path = "preview.png"
    image.save(image_path)

# ── 7) Show it + OCR text ──
st.image(image_path, caption="Preview", use_container_width=True)
st.subheader("📝 Extracted OCR Text")
st.write(extract_text_from_image(image_path))

# ── 8) Run CLIP & show top 5 ──
st.subheader("🔮 Classification Results")
img_tensor = PREPROCESS(Image.open(image_path)).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    image_features = MODEL.encode_image(img_tensor)
    logits = (image_features @ TEXT_FEATURES.T).softmax(dim=-1)
    probs = logits.cpu().numpy()[0]

for label, prob in sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)[:5]:
    st.write(f"**{label}** — {prob:.2%}")
