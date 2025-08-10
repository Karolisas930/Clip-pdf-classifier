# scripts/make_json.py
from pathlib import Path
import os
from utils import build_tree_from_disk, to_json

def main():
    lang = os.getenv("TAXO_LANG", "en")     # e.g. TAXO_LANG=de python scripts/make_json.py
    fallback = os.getenv("TAXO_FALLBACK", "en")
    nodes = build_tree_from_disk("data")
    out = Path("data/taxonomy.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_json(nodes, lang=lang, fallback=fallback), encoding="utf-8")
    print(f"wrote {out.resolve()} (lang={lang}, fallback={fallback})")

if __name__ == "__main__":
    main()
