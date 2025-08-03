import os import streamlit as st import torch import clip from PIL import Image import fitz # PyMuPDF import pandas as pd from ocr_utils import extract_text_from_image

Load CLIP model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

Load hierarchical labels

@st.cache_data def load_labels(): # combine three CSVs into a flat list of labels df_sec = pd.read_csv("Industry-Grid view.csv").fillna("") df_cat = pd.read_csv("Service Category-Grid view.csv").fillna("") df_sub = pd.read_csv("Specialization-Grid view.csv").fillna("") merged = ( df_sub.merge(df_cat, left_on="category_id", right_on="id", suffixes=("_sub", "_cat")) .merge(df_sec, left_on="section_id", right_on="id", suffixes=("", "_sec")) ) merged["full_label"] = ( merged["name_sec"] + " > " + merged["name_cat"] + " > " + merged["name_subcat"] ) return merged["full_label"].dropna().unique().tolist()

LABELS = load_labels() TEXT_TOKENS = clip.tokenize(LABELS).to(DEVICE)

Streamlit UI setup

st.set_page_config(page_title="Smart PDF/Image Classifier", layout="wide") st.title("📄🔍 Smart PDF/Image Classifier with CLIP")

File upload (PDF or image)

uploaded = st.file_uploader( "Upload a PDF or image file", type=["pdf", "png", "jpg", "jpeg"] )

if uploaded: # Determine mime type file_type = uploaded.type if file_type == "application/pdf": # save to temp.pdf with open("temp.pdf", "wb") as f: f.write(uploaded.read()) # open first page doc = fitz.open("temp.pdf") page = doc.load_page(0) pix = page.get_pixmap() image_path = "preview.png" pix.save(image_path) else: # image formats image = Image.open(uploaded) image_path = "preview.png" image.save(image_path)

# Display preview
st.image(image_path, caption="Preview", use_column_width=True)

# OCR text
ocr_text = extract_text_from_image(image_path)
st.subheader("📝 Extracted OCR Text")
st.write(ocr_text)

# CLIP classification
st.subheader("🔮 Classification Results")
img = PREPROCESS(Image.open(image_path)).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    image_features = MODEL.encode_image(img)
    text_features = MODEL.encode_text(TEXT_TOKENS)
    logits = (image_features @ text_features.T).softmax(dim=-1)
    probs = logits.cpu().numpy()[0]

# show top 5 labels
top5 = sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)[:5]
for label, prob in top5:
    st.write(f"**{label}**: {prob:.2%}")
