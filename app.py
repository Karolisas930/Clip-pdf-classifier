import os
import json
from pathlib import Path

import streamlit as st
import torch
import clip
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from ocr_utils import extract_text_from_image

# ─── 1) Page config & title ────────────────────────────────────────────────────
st.set_page_config(page_title="Smart PDF/Image Classifier", layout="wide")
st.title("📄🔍 Smart PDF/Image Classifier")

# ─── 2) Sanity check NumPy import ─────────────────────────────────────────────
st.markdown(f"✅ **NumPy** imported, version: {np.__version__}")

# ─── 2.5) Text → taxonomy helpers (multilingual embeddings) ───────────────────
@st.cache_resource
def load_text_encoder():
    # Small, multilingual (DE/EN/PL/…)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def _softmax(x: np.ndarray, tau: float = 0.07) -> np.ndarray:
    x = (x - x.max()) / max(tau, 1e-6)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)

@st.cache_resource
def build_label_embeddings(prompts: list[str]) -> np.ndarray:
    enc = load_text_encoder()
    embs = enc.encode(prompts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)  # (N, D) L2-normalized

def text_to_probs(query: str, label_embs: np.ndarray, tau: float = 0.07) -> np.ndarray | None:
    if not query or not query.strip():
        return None
    enc = load_text_encoder()
    q = enc.encode([query], normalize_embeddings=True)  # (1, D)
    sims = (q @ label_embs.T)[0]                       # cosine similarity
    return _softmax(sims, tau=tau)                     # probs over labels

# Optional: cache CLIP so first run is faster
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

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
    img = Image.open(uploaded).convert("RGB")
    img.save(image_path)

# ─── 5) Show preview + OCR ────────────────────────────────────────────────────
st.image(image_path, caption="Preview", use_container_width=True)
st.subheader("📝 Extracted OCR Text")
st.write(extract_text_from_image(image_path))

# ─── 6) Load taxonomy.json (IDs + localized labels) ───────────────────────────
@st.cache_data
def load_taxonomy(lang: str = "en"):
    p = Path("data/taxonomy.json")
    if not p.exists():
        st.error("data/taxonomy.json not found. Run your GitHub Action (make_json.py) or place the file locally.")
        st.stop()

    data = json.loads(p.read_text(encoding="utf-8"))
    labels_map = data["labels"].get(lang, data["labels"].get("en", {}))  # id -> localized label
    tree = data["tree"]  # list of sections with children

    # Build leaf prompts (breadcrumbs) + parallel arrays of sub IDs and paths
    prompts: list[str] = []
    sub_ids: list[str] = []
    paths: dict[str, dict] = data.get("paths", {})  # sub_id -> {section_id, category_id}

    def label_for(_id: str) -> str:
        return labels_map.get(_id, _id)

    def walk_section(sec: dict):
        sec_id = sec["id"]
        for cat in sec.get("children", []):
            cat_id = cat["id"]
            for sub in cat.get("children", []):
                sub_id = sub["id"]
                crumb = " > ".join([label_for(sec_id), label_for(cat_id), label_for(sub_id)])
                prompts.append(crumb)
                sub_ids.append(sub_id)

    for s in tree:
        walk_section(s)

    return {
        "prompts": prompts,       # breadcrumb texts for CLIP & text matcher
        "sub_ids": sub_ids,       # subcategory IDs in the same order as prompts
        "labels_map": labels_map, # id -> localized label
        "paths": paths            # sub_id -> {section_id, category_id}
    }

LANG = st.selectbox("🌐 Language for labels", ["en","de","pl","tr","uk","ru","nl","fr","es","it"], index=0)
TX = load_taxonomy(LANG)
PROMPTS = TX["prompts"]
SUB_IDS  = TX["sub_ids"]
LABELS_MAP = TX["labels_map"]
PATHS = TX["paths"]

# Build text-embedding cache for prompts
LABEL_EMBS = build_label_embeddings(PROMPTS)

with st.expander("Debug: first 10 prompts"):
    for s in PROMPTS[:10]:
        st.write("•", s)

# ─── 7) Combined classification (image + text) ────────────────────────────────
job_text = st.text_area(
    "📝 Describe the job (optional)",
    placeholder="e.g., Move a sofa to 2nd floor in Mannheim, small elevator, weekend preferred…"
)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    w_img = st.slider("Weight: Image", 0.0, 1.0, 0.6, 0.05)
with c2:
    w_txt = st.slider("Weight: Text", 0.0, 1.0, 0.4, 0.05)
with c3:
    tau = st.slider("Temperature (sharper → lower)", 0.02, 0.20, 0.07, 0.01)

if st.button("▶️ Run classification"):
    # 1) CLIP image → probs over PROMPTS
    model, preprocess, device = load_clip()
    img_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(PROMPTS, truncate=True).to(device)
    with torch.inference_mode():
        img_feats = model.encode_image(img_tensor)
        txt_feats = model.encode_text(text_tokens)
        probs_img = (img_feats @ txt_feats.T).softmax(dim=-1)[0].cpu().numpy()

    # 2) Text → probs over PROMPTS
    probs_txt = text_to_probs(job_text or "", LABEL_EMBS, tau=tau)

    # 3) Combine (robust weighted geometric mean)
    if probs_txt is None or w_txt == 0.0:
        combined = probs_img
        used = "Image only"
    elif w_img == 0.0:
        combined = probs_txt
        used = "Text only"
    else:
        p = (np.power(probs_img + 1e-12, w_img) *
             np.power(probs_txt + 1e-12, w_txt))
        combined = p / (p.sum() + 1e-12)
        used = "Combined (geo-mean)"

    # 4) Show top-5 and confirm
    st.subheader(f"🔮 Top suggestions — {used}")
    topk = 5
    idxs = np.argsort(-combined)[:topk]

    for i in idxs:
        st.write(f"**{PROMPTS[i]}** — {combined[i]:.1%}")

    choice = st.radio(
        "✅ Confirm the best match",
        options=list(idxs),
        format_func=lambda i: f"{PROMPTS[i]} — {combined[i]:.1%}"
    )

    if st.button("Save selection"):
        sub_id = SUB_IDS[choice]
        path = PATHS.get(sub_id, {})
        sec_id = path.get("section_id", "")
        cat_id = path.get("category_id", "")
        # TODO: send {section_id, category_id, subcategory_id} to Airtable/DB
        st.success(
            f"Saved IDs → section: {sec_id} / category: {cat_id} / subcategory: {sub_id}\n\n"
            f"Label: {PROMPTS[choice]}"
        )
