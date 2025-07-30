import subprocess
subprocess.run(["pip", "install", "git+https://github.com/openai/CLIP.git"], check=True)

import clip

import streamlit as st
import torch
import clip
from PIL import Image
import os
import fitz # PyMuPDF

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

st.title("📄 PDF Classifier using CLIP")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Convert first page of PDF to image
    doc = fitz.open("temp.pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    image_path = "preview.png"
    pix.save(image_path)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Define your labels
    labels = ["invoice", "contract", "report", "letter", "email", "slide"]
    text_tokens = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        logits_per_image, _ = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    st.image(image_path, caption="Preview of PDF")
    st.subheader("Prediction Results:")
    for label, prob in zip(labels, probs[0]):
        st.write(f"{label}: {prob:.2%}")
