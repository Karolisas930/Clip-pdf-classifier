import os
import streamlit as st
import numpy as np
import torch
import clip
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import requests

# Optional OCR import (we'll guard calls below)
try:
    from ocr_utils import extract_text_from_image
except Exception:  # missing dep or import error
    extract_text_from_image = None

# ──────────────────────────────────────────────────────────────────────────────
# Page config
st.set_page_config(page_title="Smart PDF/Image Classifier", layout="wide")
st.title("📄🔍 Smart PDF/Image Classifier with CLIP")

# ──────────────────────────────────────────────────────────────────────────────
# Load taxonomy from GitHub Pages (single source of truth)
TAXO_URL = "https://karolisas930.github.io/Clip-pdf-classifier/data/taxonomy.json"

@st.cache_data(show_spinner=False, ttl=600)
def fetch_taxonomy(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def _label_for(node, taxo, lang="en"):
    # Prefer the global labels map; fall back to node.labels; then EN.
    for lc in (lang, "en"):
        if lc in taxo.get("labels", {}) and node["id"] in taxo["labels"][lc]:
            return (taxo["labels"][lc][node["id"]] or "").strip()
        if lc in node.get("labels", {}) and node["labels"][lc]:
            return node["labels"][lc].strip()
    return ""

def build_labels_from_taxonomy(taxo: dict, lang="en"):
    labels = []
    def walk(node, crumbs):
        name = _label_for(node, taxo, lang)
        here = crumbs + ([name] if name else [])
        if node["type"] == "subcategory":
            labels.append(" > ".join(here))
        for ch in node.get("children", []):
            walk(ch, here)
    for sec in taxo.get("tree", []):
        walk(sec, [])
    return sorted({l for l in labels if l.strip()})

with st.sidebar:
    st.subheader("Taxonomy")
    taxo = fetch_taxonomy(TAXO_URL)
    langs = taxo.get("meta", {}).get("languages", ["en"])
    default_idx = langs.index("en") if "en" in langs else 0
    lang = st.selectbox("Label language", options=langs, index=default_idx)
    last_gen = taxo.get("meta", {}).get("generated_at", "")
    if last_gen:
        st.caption(f"Generated: {last_gen}")

LABELS = build_labels_from_taxonomy(taxo, lang=lang)
st.write(f"✅ Loaded **{len(LABELS)}** labels from taxonomy ({lang.upper()}).")

# ──────────────────────────────────────────────────────────────────────────────
# Uploader
uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg"],
    key="uploader_mixed",
)
if not uploaded:
    st.info("Please upload a PDF or image file to classify.")
    st.stop()

# Turn upload into an in-memory preview image (PIL.Image)
preview_img: Image.Image | None = None
if uploaded.type == "application/pdf":
    # Render first page directly from bytes at 2× scale for readability
    doc = fitz.open(stream=uploaded.getvalue(), filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    preview_img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
else:
    preview_img = Image.open(uploaded).convert("RGB")
    
# Show preview (robust: PNG bytes with NumPy fallback)
col1, col2 = st.columns([1, 1])
with col1:
    if preview_img is None:
        st.error("Couldn’t build a preview image.")
        st.stop()
    try:
        buf = BytesIO()
        preview_img.save(buf, format="PNG")
        buf.seek(0)
        st.image(buf.getvalue(), caption="Preview", use_container_width=True)
    except Exception as e:
        st.warning(f"Preview via PNG bytes failed ({e}). Falling back.")
        st.image(np.asarray(preview_img), caption="Preview", use_container_width=True)
        
# OCR (best-effort; won’t crash the app)
def run_ocr_safe(img: Image.Image) -> str:
    if extract_text_from_image is None:
        return "OCR not available in this build."
    # Try passing the PIL image; if their util expects a path, fall back.
    try:
        return extract_text_from_image(img) or ""
    except TypeError:
        tmp = "_ocr_preview.png"
        img.save(tmp)
        try:
            return extract_text_from_image(tmp) or ""
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
    except Exception as e:
        return f"OCR failed: {e}"

with col2:
    st.subheader("📝 Extracted OCR Text")
    st.write(run_ocr_safe(preview_img))

# ──────────────────────────────────────────────────────────────────────────────
# CLIP classification (batched so it fits CPU/GPU memory)
def clip_scores(image_pil: Image.Image, labels, model, preprocess, device, batch_size=256):
    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores = []
        for i in range(0, len(labels), batch_size):
            batch = labels[i:i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).cpu().numpy()[0]
            scores.extend(sims.tolist())
    scores = np.array(scores, dtype=np.float32)
    # softmax across all labels
    exps = np.exp(scores - scores.max())
    probs = exps / exps.sum()
    return probs

if st.button("▶️ Run CLIP classification"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner(f"Loading CLIP on {device}…"):
        model, preprocess = clip.load("ViT-B/32", device=device)

    with st.spinner("Scoring image against taxonomy labels…"):
        probs = clip_scores(preview_img, LABELS, model, preprocess, device, batch_size=256)

    st.subheader("🔮 Top matches")
    topk = 5
    idx = np.argsort(-probs)[:topk]
    for rank, i in enumerate(idx, start=1):
        st.write(f"**{rank}. {LABELS[i]}** — {probs[i]:.2%}")

    with st.expander("Show top 30 as a table"):
        topn = np.argsort(-probs)[:30]
        st.dataframe(
            {"label": [LABELS[i] for i in topn], "prob": [float(probs[i]) for i in topn]},
            use_container_width=True,
        )
