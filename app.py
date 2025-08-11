# app.py
# Smart PDF/Image classifier with CLIP + optional OCR + taxonomy from GitHub Pages

import os
import re
import hashlib
from io import BytesIO
from typing import List

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

def build_labels_from_taxonomy(taxo: dict, lang: str = "en") -> List[str]:
    labels: List[str] = []

    def walk(node: dict, crumbs: List[str]) -> None:
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

    # 🔧 Prompt + OCR settings
    prompt_mode = st.selectbox(
        "Text prompt for CLIP",
        ["subcategory", "sub+parent", "full path"],
        index=0,
    )
    ocr_boost = st.slider("OCR boost", 0.0, 1.0, 0.25, 0.05)

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
    ocr_text = run_ocr_safe(preview_img)
    st.write(ocr_text)


# ─────────────────────── CLIP model + caching ─────────────────────────
MODEL_NAME = "ViT-B/32"

@st.cache_resource(show_spinner=False)
def load_clip(model_name: str, device: str = "cpu"):
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess

# ---------- Prompt helpers ----------
def label_to_prompt(label: str, mode: str) -> str:
    parts = [p.strip() for p in label.split(">")]
    sub = parts[-1] if parts else label
    if mode == "subcategory":
        base = sub
    elif mode == "sub+parent" and len(parts) >= 2:
        base = f"{parts[-2]} {sub}"
    else:  # "full path"
        base = ", ".join(parts)
    return f"a photo of {base.lower()}"

def build_prompts(labels: List[str], mode: str) -> List[str]:
    return [label_to_prompt(l, mode) for l in labels]

def _prompts_signature(prompts: List[str], language: str, model_name: str, mode: str) -> str:
    """Compact signature so text features get cached per language + model + prompts."""
    m = hashlib.md5()
    m.update(model_name.encode()); m.update(b"|"); m.update(language.encode()); m.update(b"|"); m.update(mode.encode()); m.update(b"|")
    for s in prompts:
        m.update(s.encode()); m.update(b"\0")
    return m.hexdigest()

@st.cache_resource(show_spinner=False)
def cached_text_features(
    signature: str,
    prompts: List[str],
    model_name: str,
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Compute & cache normalized text features (CPU tensor) for given prompts."""
    model, _ = load_clip(model_name, device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            toks = clip.tokenize(prompts[i:i + batch_size], truncate=True).to(device)
            txt = model.encode_text(toks)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            feats.append(txt.cpu())
    return torch.cat(feats, dim=0)  # (N, D) on CPU

# OCR-aware re-ranking (very lightweight)
_token_re = re.compile(r"[A-Za-z0-9]+", re.UNICODE)
def _tokens(s: str) -> set:
    return set(_token_re.findall(s.lower()))

def ocr_weights(labels: List[str], ocr_text: str, base_mode: str = "subcategory") -> np.ndarray:
    if not isinstance(ocr_text, str) or not ocr_text.strip():
        return np.ones(len(labels), dtype=np.float32)
    text_tokens = _tokens(ocr_text)
    w = np.ones(len(labels), dtype=np.float32)
    for i, lbl in enumerate(labels):
        toks = _tokens(label_to_prompt(lbl, base_mode))
        if text_tokens.intersection(toks):
            w[i] += 0.5  # simple bump if any token matches
    return w

def classify_image(preview_img: Image.Image, labels: List[str], language: str, mode: str, ocr_text: str, ocr_boost: float) -> np.ndarray:
    """Return probabilities over labels using cached text embeddings + optional OCR boost."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip(MODEL_NAME, device)

    prompts = build_prompts(labels, mode)
    sig = _prompts_signature(prompts, language, MODEL_NAME, mode)
    text_feats_cpu = cached_text_features(sig, prompts, MODEL_NAME, device)

    with torch.no_grad():
        img = preprocess(preview_img).unsqueeze(0).to(device)
        img_feat = model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        tf = text_feats_cpu.to(device)
        probs = (img_feat @ tf.T).softmax(dim=-1).cpu().numpy()[0]

    # OCR re-ranking (if enabled)
    if ocr_boost > 0:
        w = ocr_weights(labels, ocr_text, base_mode="subcategory")
        probs = probs * (1.0 + ocr_boost * (w - 1.0))
        probs = probs / probs.sum()

    return probs


# ─────────────────────────── Run classification ───────────────────────
if st.button("▶️ Run CLIP classification"):
    with st.spinner("Scoring image against taxonomy labels…"):
        probs = classify_image(preview_img, LABELS, lang, prompt_mode, ocr_text, ocr_boost)

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
