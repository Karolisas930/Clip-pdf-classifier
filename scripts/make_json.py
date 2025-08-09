#!/usr/bin/env python3
"""
Reads sections.csv / categories.csv / subcategories.csv and writes data/taxonomy.json
for your Webflow embed or any front-end.
"""

from pathlib import Path
import csv, json

DATA = Path("data")

SECTIONS_CSV = DATA / "sections.csv"
CATEGORIES_CSV = DATA / "categories.csv"
SUBCATEGORIES_CSV = DATA / "subcategories.csv"
OUT_JSON = DATA / "taxonomy.json"

def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def pick_label(row, lang, fallback_col):
    if lang in row and row[lang]:
        return row[lang].strip()
    if fallback_col in row and row[fallback_col]:
        return row[fallback_col].strip()
    if "en" in row and row["en"]:
        return row["en"].strip()
    return next((v.strip() for k,v in row.items() if v and k not in ("section_id","category_id","subcategory_id","icon")), "")

def build_tree(lang="en"):
    sections = read_csv(SECTIONS_CSV)
    categories = read_csv(CATEGORIES_CSV)
    subs = read_csv(SUBCATEGORIES_CSV)

    # groupings
    cats_by_section = {}
    for c in categories:
        cats_by_section.setdefault(c["section_id"], []).append(c)
    subs_by_category = {}
    for s in subs:
        subs_by_category.setdefault(s["category_id"], []).append(s)

    def order(rows, sort_col):
        def key(r):
            v = r.get(sort_col, "")
            return (v == "" or v is None, str(v), r.get("en",""))
        return sorted(rows, key=key)

    tree = []
    for s in order(sections, "sort_order_section"):
        s_node = {
            "id": s["section_id"],
            "type": "section",
            "label": pick_label(s, lang, "section_name_en"),
            "icon": s.get("icon",""),
            "children": []
        }
        for c in order(cats_by_section.get(s["section_id"], []), "sort_order_category"):
            c_node = {
                "id": c["category_id"],
                "type": "category",
                "label": pick_label(c, lang, "category_name_en"),
                "icon": c.get("icon",""),
                "children": []
            }
            for g in order(subs_by_category.get(c["category_id"], []), "sort_order_subcategory"):
                c_node["children"].append({
                    "id": g["subcategory_id"],
                    "type": "subcategory",
                    "label": pick_label(g, lang, "subcategory_name_en"),
                    "icon": g.get("icon","")
                })
            s_node["children"].append(c_node)
        tree.append(s_node)
    return tree

def main():
    out = {
        "version": 1,
        "default_lang": "en",
        "tree": build_tree("en"), # default bundle in English
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✓ Wrote", OUT_JSON)

if __name__ == "__main__":
    main()
  
