import os
import streamlit as st
import torch
import clip
from PIL import Image
import fitz # PyMuPDF
import pandas as pd
import numpy as np
from ocr_utils import extract_text_from_image

# ─── 1) Page config & title ────────────────────────────────────────────────────
st.set_page_config(page_title="Smart PDF/Image Classifier", layout="wide")
st.title("📄🔍 Smart PDF/Image Classifier with CLIP")

# ─── 2) Sanity check NumPy import ─────────────────────────────────────────────
st.markdown(f"✅ **NumPy** imported, version: {np.__version__}")

# ─── 3) Uploader ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg"],
    key="uploader_mixed",
)
if not uploaded:
    st.info("Please upload a PDF or image file to classify.")
    st.stop()

# ─── 4) Turn uploaded into preview.png ────────────────────────────────────────
image_path = "preview.png"
if uploaded.type == "application/pdf":
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.getbuffer())
    doc = fitz.open("temp.pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    pix.save(image_path)
else:
    img = Image.open(uploaded)
    img.save(image_path)

# ─── 5) Show preview + OCR ────────────────────────────────────────────────────
st.image(image_path, caption="Preview", use_container_width=True)
st.subheader("📝 Extracted OCR Text")
st.write(extract_text_from_image(image_path))

# ─── 6) Load & cache your labels from hierarchy.csv ───────────────────────────
@st.cache_data
def load_labels():
    df = pd.read_csv("hierarchy.csv").fillna("")
    df["full_label"] = (
        df["Industry"].astype(str)
        + " > " + df["Service Category"].astype(str)
        + " > " + df["Specialization"].astype(str)
    )
    return [lbl for lbl in df["full_label"].unique() if lbl.strip()]

LABELS = load_labels()

# ─── 7) Defer heavy CLIP work until button click ───────────────────────────────
if st.button("▶️ Run CLIP classification"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # truncate so no “too long for context” errors
    text_tokens = clip.tokenize(LABELS, truncate=True).to(device)
    img_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        text_features = model.encode_text(text_tokens)
        logits = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

    st.subheader("🔮 Classification Results")
    top5 = sorted(zip(LABELS, logits), key=lambda x: x[1], reverse=True)[:5]
    for label, p in top5:
        st.write(f"**{label}** — {p:.2%}")
