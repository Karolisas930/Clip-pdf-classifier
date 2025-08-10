# utils/shared.py
import json, pathlib, torch, clip
from PIL import Image
import streamlit as st

@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

@st.cache_data
def load_taxonomy_for_clip(lang="en"):
    data = json.loads(pathlib.Path("data/taxonomy.json").read_text(encoding="utf-8"))
    labels_map = data["labels"].get(lang, data["labels"]["en"])
    tree = data["tree"]
    prompts, sub_ids = [], []
    def walk(n, crumb):
        name = labels_map.get(n["id"], n["id"])
        t2 = crumb + [name]
        kids = n.get("children") or []
        if kids:
            for ch in kids: walk(ch, t2)
        else:
            prompts.append(" > ".join(t2))
            sub_ids.append(n["id"])
    for s in tree: walk(s, [])
    return prompts, sub_ids, data["paths"], labels_map

@st.cache_data
def tokenize_labels(prompts, device_str):
    dev = torch.device(device_str)
    return clip.tokenize(prompts, truncate=True).to(dev)

# thin wrapper so both pages import same OCR name
def extract_text_from_image(path):
    from ocr_utils import extract_text_from_image as _ocr
    return _ocr(path)
