# app.py
# Smart PDF/Image classifier with CLIP + optional OCR + taxonomy from GitHub Pages

import os
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF

# CLIP + Torch
import torch
import clip

# Optional OCR (won't crash the app if missing)
try:
    from ocr_utils import extract_text_from_image
except Exception:
    extract_text_from_image = None


# ───────────────────────────── Page setup ─────────────────────────────
st.set_page_config(page_title="Smart PDF/Image Classifier", layout="wide")
st.title("📄🔍 Smart PDF/Image Classifier with CLIP")


# ─────────────────────────── Taxonomy loading ─────────────────────────
TAXO_URL = "https://karolisas930.github.io/Clip-pdf-classifier/data/taxonomy.json"

@st.cache_data(show_spinner=False, ttl=600)
def fetch_taxonomy(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def _label_for(node: dict, taxo: dict, lang: str = "en") -> str:
    # Prefer global labels map, then node.labels, with EN fallback.
    for lc in (lang, "en"):
        if lc in taxo.get("labels", {}) and node["id"] in taxo["labels"][lc]:
            return (taxo["labels"][lc][node["id"]] or "").strip()
        if lc in node.get("labels", {}) and node["labels"].get(lc):
            return (node["labels"][lc] or "").strip()
    return ""

def build_labels_from_taxonomy(taxo: dict, lang: str = "en") -> list[str]:
    labels: list[str] = []

    def walk(node: dict, crumbs: list[str]) -> None:
        name = _label_for(node, taxo, lang)
        here = crumbs + ([name] if name else [])
        if node.get("type") == "subcategory":
            labels.append(" > ".join(here))
        for ch in node.get("children", []):
            walk(ch, here)

    for sec in taxo.get("tree", []):
        walk(sec, [])
    # dedupe + keep non-empty, sorted for stability
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


# ───────────────────────────── File upload ────────────────────────────
uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg"],
    key="uploader_mixed",
)
if not uploaded:
    st.info("Please upload a PDF or image file to classify.")
    st.stop()


# ───────────────────── Build a preview PIL.Image ──────────────────────
def pdf_first_page_to_image(pdf_bytes: bytes, scale: float = 2.0) -> Image.Image:
    """Render first page of a PDF (from bytes) into a PIL RGB image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

if uploaded.type == "application/pdf":
    preview_img: Image.Image = pdf_first_page_to_image(uploaded.getvalue(), scale=2.0)
else:
    preview_img = Image.open(uploaded).convert("RGB")


# ───────────────────── Streamlit image compat helper ──────────────────
def st_image_compat(data, caption: str = ""):
    """Call st.image using the arg name supported by the runtime."""
    try:
        st.image(data, caption=caption, use_container_width=True)
    except TypeError:
        # Older Streamlit expects use_column_width
        st.image(data, caption=caption, use_column_width=True)


# ───────────────────────── Show preview + OCR ─────────────────────────
col1, col2 = st.columns([1, 1])
with col1:
    try:
        buf = BytesIO()
        preview_img.save(buf, format="PNG")
        buf.seek(0)
        st_image_compat(buf.getvalue(), "Preview")
    except Exception:
        st_image_compat(np.asarray(preview_img), "Preview")

def run_ocr_safe(img: Image.Image) -> str:
    """Try OCR if available; handle both PIL-image and path-based utilities."""
    if extract_text_from_image is None:
        return "OCR not available in this build."
    # Try passing PIL image directly; if the util wants a path, fall back.
    try:
        txt = extract_text_from_image(img)
        if isinstance(txt, str):
            return txt
    except TypeError:
        pass
    except Exception as e:
        return f"OCR failed: {e}"

    tmp = "_ocr_preview.png"
    try:
        img.save(tmp)
        return extract_text_from_image(tmp) or ""
    except Exception as e:
        return f"OCR failed: {e}"
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

with col2:
    st.subheader("📝 Extracted OCR Text")
    st.write(run_ocr_safe(preview_img))


# ─────────────────────── CLIP model + scoring ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_clip(device: str = "cpu"):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def clip_scores(image_pil: Image.Image, labels: list[str], model, preprocess, device: str, batch_size: int = 256) -> np.ndarray:
    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores: list[float] = []
        for i in range(0, len(labels), batch_size):
            batch = labels[i : i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).cpu().numpy()[0].tolist()
            scores.extend(sims)
    scores_arr = np.asarray(scores, dtype=np.float32)
    exps = np.exp(scores_arr - scores_arr.max())  # stable softmax
    return exps / exps.sum()


# ─────────────────────────── Run classification ───────────────────────
if st.button("▶️ Run CLIP classification"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner(f"Loading CLIP on {device}…"):
        model, preprocess = load_clip(device)

    with st.spinner("Scoring image against taxonomy labels…"):
        probs = clip_scores(preview_img, LABELS, model, preprocess, device, batch_size=256)

    st.subheader("🔮 Top matches")
    topk = 5
    top_idx = np.argsort(-probs)[:topk]
    for rank, i in enumerate(top_idx, start=1):
        st.write(f"**{rank}. {LABELS[i]}** — {probs[i]:.2%}")

    with st.expander("Show top 30 as a table"):
        topn = np.argsort(-probs)[:30]
        st.dataframe(
            {"label": [LABELS[i] for i in topn], "prob": [float(probs[i]) for i in topn]},
            use_container_width=True,
    )
