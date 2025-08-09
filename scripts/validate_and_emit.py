import pandas as pd
from pathlib import Path
name: Build hierarchy CSVs

on:
  push:
    paths:
      - 'data/**'
      - 'scripts/validate_and_emit.py'
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install pandas openpyxl

      - name: Generate CSVs
        run: python scripts/validate_and_emit.py

      - name: Commit generated CSVs
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: regenerate sections/categories/subcategories CSVs"
          file_pattern: |
            data/sections.csv
            data/categories.csv
            data/subcategories.csv

DATA = Path("data")

HIER = DATA / "services_hierarchy_extended.csv"
SEC_XLSX = DATA / "SECTIONS.xlsx"        # master (editable)
CAT_XLSX = DATA / "CATEGORIES.xlsx"      # master (editable)
SUB_XLSX = DATA / "SUBCATEGORIES.xlsx"   # master (editable)

OUT_SECTIONS = DATA / "sections.csv"         # generated
OUT_CATEGORIES = DATA / "categories.csv"     # generated
OUT_SUBCATEGORIES = DATA / "subcategories.csv"  # generated

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def reorder(df, id_cols, sort_col, icon_col):
    # language columns = short lowercase like en,de,pl,lt (<=3 chars)
    fixed = set(id_cols + [sort_col, icon_col])
    langs = [c for c in df.columns if c not in fixed and len(c) <= 3 and c.islower()]
    meta  = [c for c in df.columns if c not in fixed.union(langs)]
    ordered = id_cols + ([sort_col] if sort_col in df.columns else []) \
              + ([icon_col] if icon_col in df.columns else []) \
              + sorted(set(langs)) + meta
    # drop dupes while preserving order
    seen=set(); ordered=[c for c in ordered if (c in df.columns) and not (c in seen or seen.add(c))]
    return df[ordered]

def main():
    # --- load
    h = pd.read_csv(HIER, dtype=str).fillna("")
    sec = pd.read_excel(SEC_XLSX, dtype=str).fillna("")
    cat = pd.read_excel(CAT_XLSX, dtype=str).fillna("")
    sub = pd.read_excel(SUB_XLSX, dtype=str).fillna("")

    # normalize headers
    for df in (h, sec, cat, sub):
        df.columns = [c.strip().lower() for c in df.columns]

    # sanity: hierarchy must have these
    need_h = {
        "section_id","section_name_en",
        "category_id","category_name_en",
        "subcategory_id","subcategory_name_en"
    }
    missing = need_h - set(h.columns)
    if missing:
        raise SystemExit(f"[ERROR] hierarchy missing columns: {missing}")

    # ensure id columns exist in masters
    for name, df, req in [
        ("SECTIONS.xlsx", sec, {"section_id"}),
        ("CATEGORIES.xlsx", cat, {"category_id","section_id"}),
        ("SUBCATEGORIES.xlsx", sub, {"subcategory_id","category_id"}),
    ]:
        miss = req - set(df.columns)
        if miss:
            raise SystemExit(f"[ERROR] {name} missing columns: {miss}")

    # add icon + sort if missing
    sec = ensure_cols(sec, ["sort_order_section","icon"])
    cat = ensure_cols(cat, ["sort_order_category","icon"])
    sub = ensure_cols(sub, ["sort_order_subcategory","icon"])

    # 'en' fallback from hierarchy names
    if "en" not in sec.columns: sec["en"] = ""
    if "en" not in cat.columns: cat["en"] = ""
    if "en" not in sub.columns: sub["en"] = ""

    sec_en_map = h.drop_duplicates("section_id").set_index("section_id")["section_name_en"]
    cat_en_map = h.drop_duplicates("category_id").set_index("category_id")["category_name_en"]
    sub_en_map = h.drop_duplicates("subcategory_id").set_index("subcategory_id")["subcategory_name_en"]

    sec.loc[sec["en"].eq(""), "en"] = sec["section_id"].map(sec_en_map).fillna("")
    cat.loc[cat["en"].eq(""), "en"] = cat["category_id"].map(cat_en_map).fillna("")
    sub.loc[sub["en"].eq(""), "en"] = sub["subcategory_id"].map(sub_en_map).fillna("")

    # FK validation
    sections_ids    = set(h["section_id"])
    categories_ids  = set(h["category_id"])

    bad_cat_fk = sorted(set(cat["section_id"]) - sections_ids)
    bad_sub_fk = sorted(set(sub["category_id"]) - categories_ids)

    if bad_cat_fk:
        print(f"[WARN] {len(bad_cat_fk)} category.section_id not found in hierarchy sections (e.g. {bad_cat_fk[:5]})")
    if bad_sub_fk:
        print(f"[WARN] {len(bad_sub_fk)} subcategory.category_id not found in hierarchy categories (e.g. {bad_sub_fk[:5]})")

    # final column order
    sec_out = reorder(sec, ["section_id"], "sort_order_section", "icon")
    cat_out = reorder(cat, ["category_id","section_id"], "sort_order_category", "icon")
    sub_out = reorder(sub, ["subcategory_id","category_id"], "sort_order_subcategory", "icon")

    # write UTF-8 CSVs
    OUT_SECTIONS.parent.mkdir(parents=True, exist_ok=True)
    sec_out.to_csv(OUT_SECTIONS, index=False, encoding="utf-8")
    cat_out.to_csv(OUT_CATEGORIES, index=False, encoding="utf-8")
    sub_out.to_csv(OUT_SUBCATEGORIES, index=False, encoding="utf-8")

    # quick summary
    print("✓ Wrote:")
    print(" -", OUT_SECTIONS)
    print(" -", OUT_CATEGORIES)
    print(" -", OUT_SUBCATEGORIES)

if __name__ == "__main__":
    main()

