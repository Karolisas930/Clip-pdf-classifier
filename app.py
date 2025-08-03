import os
import streamlit as st
import torch
import clip
from PIL import Image
import fitz # PyMuPDF
import pandas as pd
from ocr_utils import extract_text_from_image

# — Load & cache your single hierarchy.csv —
@st.cache_data
def load_labels():
    df = pd.read_csv("hierarchy.csv").fillna("")
    df["full_label"] = (
        df["Industry"].astype(str)
        + " > "
        + df["Service Category"].astype(str)
        + " > "
        + df["Specialization"].astype(str)
    )
    return [lbl for lbl in df["full_label"].unique() if lbl.strip() != ""]

LABELS = load_labels()

# — Load CLIP model once —
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
TEXT_TOKENS = clip.tokenize(LABELS).to(DEVICE)

# — Streamlit page config & title —
st.set_page_config(page_title="Smart PDF/Image Classifier", layout="wide")
st.title("📄🔍 Smart PDF/Image Classifier with CLIP")

# — Single uploader for PDF or image —
uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg"],
    key="uploader_mixed",
)
if not uploaded:
    st.info("Please upload a PDF or image file to classify.")
    st.stop()

# — Branch on file type to produce a single preview image —
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

# — Display preview and extract OCR text —
st.image(image_path, caption="Preview", use_column_width=True)
st.subheader("📝 Extracted OCR Text")
st.write(extract_text_from_image(image_path))

# — CLIP classification against your hierarchy labels —
st.subheader("🔮 Classification Results")
img_tensor = PREPROCESS(Image.open(image_path)).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    image_features = MODEL.encode_image(img_tensor)
    text_features = MODEL.encode_text(TEXT_TOKENS)
    logits = (image_features @ text_features.T).softmax(dim=-1)
    probs = logits.cpu().numpy()[0]

for label, prob in sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)[:5]:
    st.write(f"**{label}** — {prob:.2%}")
