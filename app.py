# app.py
# Smart PDF/Image classifier with CLIP + optional OCR + taxonomy from GitHub Pages

import os
import re
import time
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

# Keep CPU usage sane on small hosts
torch.set_num_threads(min(4, os.cpu_count() or 4))

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
def pdf_page_to_image(pdf_bytes: bytes, page_index: int, scale: float = 2.0) -> Image.Image:
    """Render selected page of a PDF (from bytes) into a PIL RGB image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

# If it's a PDF, offer page + scale controls in the sidebar
if uploaded.type == "application/pdf":
    pdf_bytes = uploaded.getvalue()
    tmp_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    with st.sidebar:
        page_no = st.number_input("PDF page", 1, tmp_doc.page_count, 1)
        render_scale = st.slider("Render scale", 1.0, 3.0, 2.0, 0.5)
    preview_img: Image.Image = pdf_page_to_image(pdf_bytes, int(page_no - 1), render_scale)
else:
    preview_img = Image.open(uploaded).convert("RGB")

# ───────────────────── Streamlit image compat helper ──────────────────
def st_image_compat(data, caption: str = ""):
    """Call st.image using the arg name supported by the runtime."""
    try:
        st.image(data, caption=caption, use_container_width=True)
    except TypeError:
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

# Speed helpers: cache image embedding + OCR prefilter
def img_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

@st.cache_resource(show_spinner=False)
def cached_image_feature(img_md5: str, img_bytes: bytes, model_name: str, device: str):
    """Encode + normalize image once; cache by MD5."""
    model, preprocess = load_clip(model_name, device)
    pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    with torch.no_grad():
        t = preprocess(pil).unsqueeze(0).to(device)
        feat = model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu()  # keep on CPU; move to device when scoring

def ocr_prefilter(labels: List[str], ocr_text: str, cap: int = 400) -> List[str]:
    """Keep at most `cap` labels that share tokens with OCR; fallback to all."""
    if not isinstance(ocr_text, str) or not ocr_text.strip():
        return labels
    tt = _tokens(ocr_text)
    scored = []
    for l in labels:
        toks = _tokens(label_to_prompt(l, "subcategory"))
        overlap = len(tt.intersection(toks))
        scored.append((overlap, l))
    scored.sort(key=lambda x: (-x[0], x[1]))
    if scored and scored[0][0] == 0:
        return labels
    return [l for _, l in scored[:cap]]

def classify_image(preview_img: Image.Image,
                   labels: List[str],
                   language: str,
                   mode: str,
                   ocr_text: str,
                   ocr_boost: float,
                   candidate_cap: int = 400) -> tuple[np.ndarray, List[str], torch.Tensor]:
    """
    Returns: (probs, labels_used, img_feat_cpu)
    - labels may be reduced via OCR prefilter for speed
    - img_feat is cached so refinements are instant
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Cache image embedding
    bytes_png = img_png_bytes(preview_img)
    img_md5 = hashlib.md5(bytes_png).hexdigest()
    img_feat_cpu = cached_image_feature(img_md5, bytes_png, MODEL_NAME, device)  # (1, D) on CPU

    # 2) OCR prefilter to shrink label set
    labels_used = ocr_prefilter(labels, ocr_text, cap=candidate_cap)

    # 3) Build prompts & cached text features
    prompts = build_prompts(labels_used, mode)
    sig = _prompts_signature(prompts, language, MODEL_NAME, mode)
    text_feats_cpu = cached_text_features(sig, prompts, MODEL_NAME, device)      # (N, D) CPU

    # 4) Score on the chosen device
    with torch.no_grad():
        tf = text_feats_cpu.to(device)
        img = img_feat_cpu.to(device)
        logits = img @ tf.T
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    # 5) Optional OCR reweight (light)
    if ocr_boost > 0:
        w = ocr_weights(labels_used, ocr_text, base_mode="subcategory")
        probs = probs * (1.0 + ocr_boost * (w - 1.0))
        probs = probs / probs.sum()

    return probs, labels_used, img_feat_cpu

# ─────────── helpers for refine/use & notifications ───────────
def parent_prefix(label: str) -> str:
    """Everything before the last ' > ' segment."""
    parts = [p.strip() for p in label.split(">")]
    return " > ".join(parts[:-1]) if len(parts) > 1 else ""

def notify(msg: str, icon: str = "✅"):
    """Streamlit toast fallback for older versions."""
    try:
        st.toast(msg, icon=icon)
    except Exception:
        st.success(msg)

# ─────────────────────────── Controls ───────────────────────────
with st.sidebar:
    st.markdown("---")
    top_k = st.slider("Top-K results", 3, 20, 5)
    min_prob = st.slider("Minimum probability", 0.0, 0.2, 0.02, 0.01)
    candidate_cap = st.slider("Max labels to score (after OCR filter)", 100, 1000, 400, 50)

    # Show current refine/selection state
    if st.session_state.get("refine_prefix"):
        st.info(f"Refining within **{st.session_state['refine_prefix']}**")
        if st.button("Clear refine"):
            st.session_state.pop("refine_prefix", None)
            st.experimental_rerun()

    if st.session_state.get("final_label"):
        st.success(f"Chosen: {st.session_state['final_label']}")
        st.caption(f"Confidence: {st.session_state.get('final_prob', 0.0):.2%}")

# ─────────────────────────── Run classification ───────────────────────
if st.button("▶️ Run CLIP classification"):
    # Use refined pool if the user clicked “Refine” earlier
    base_pool = LABELS
    if st.session_state.get("refine_prefix"):
        pfx = st.session_state["refine_prefix"]
        base_pool = [l for l in LABELS if l.startswith(pfx + " > ")]

    t0 = time.time()
    with st.spinner("Scoring image against taxonomy labels…"):
        probs, labels_pool, _img_feat = classify_image(
            preview_img, base_pool, lang, prompt_mode, ocr_text, ocr_boost, candidate_cap
        )
    elapsed = time.time() - t0

    st.subheader(f"🔮 Top matches  _(took {elapsed:.2f}s)_")
    ranked = np.argsort(-probs)
    filtered = [i for i in ranked if probs[i] >= min_prob][:top_k]

    if not filtered:
        st.info("No labels above the selected probability threshold.")
        st.stop()

    # If we’re inside a refined branch, remind the user + offer clear
    if st.session_state.get("refine_prefix"):
        active_pfx = st.session_state["refine_prefix"]
        st.info(f"Currently refining within **{active_pfx}** ({len(labels_pool)} labels).")
        if st.button("🔄 Clear refine", key="clear_refine_top"):
            st.session_state.pop("refine_prefix", None)
            st.experimental_rerun()

    # Show results with actions
    for rank, i in enumerate(filtered, start=1):
        lbl = labels_pool[i]
        prob = float(probs[i])

        c1, c2, c3 = st.columns([5, 1.2, 1.2])
        with c1:
            st.write(f"**{rank}. {lbl}** — {prob:.2%}")
        with c2:
            if st.button("Refine", key=f"ref_{i}"):
                pfx = parent_prefix(lbl)
                if pfx:
                    st.session_state["refine_prefix"] = pfx
                    st.experimental_rerun()
        with c3:
            if st.button("Use", key=f"use_{i}"):
                st.session_state["final_label"] = lbl
                st.session_state["final_prob"] = prob
                notify("Label selected")

    # Table + CSV download (uses the same pool the model saw)
    with st.expander("Show top 30 as a table"):
        topn = ranked[:30]
        table = {"label": [labels_pool[i] for i in topn],
                 "prob": [float(probs[i]) for i in topn]}
        import pandas as pd
        df = pd.DataFrame(table)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

    # If the user picked a label, offer export & quick search
    if st.session_state.get("final_label"):
        st.markdown("### ✅ Selected label")
        st.code(st.session_state["final_label"])

        import json, urllib.parse as _up, time as _t
        payload = {
            "label": st.session_state["final_label"],
            "prob": st.session_state.get("final_prob"),
            "language": lang,
            "prompt_mode": prompt_mode,
            "ocr_boost": ocr_boost,
            "refine_prefix": st.session_state.get("refine_prefix"),
            "timestamp": int(_t.time()),
        }
        st.download_button(
            "Download selection (JSON)",
            json.dumps(payload, indent=2).encode("utf-8"),
            file_name="selection.json",
            mime="application/json",
        )

        url = f"https://www.google.com/search?q={_up.quote(st.session_state['final_label'])}"
        st.markdown(f"[🔎 Search this label on the web]({url})")
