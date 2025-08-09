#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolve effective icons for sections, categories, and subcategories.

Reads:
  data/sections.csv
  data/categories.csv
  data/subcategories.csv

Writes:
  data/icon_resolved_flat.csv

Icon cascade:
  subcategory.icon → category.icon → section.icon → /assets/icons/default.svg
"""

from __future__ import annotations
from pathlib import Path
import csv
import sys

# ---- config ---------------------------------------------------------------

DATA_DIR  = Path("data")
SECTIONS_CSV      = DATA_DIR / "sections.csv"
CATEGORIES_CSV    = DATA_DIR / "categories.csv"
SUBCATEGORIES_CSV = DATA_DIR / "subcategories.csv"
OUT_CSV           = DATA_DIR / "icon_resolved_flat.csv"

DEFAULT_ICON = "/assets/icons/default.svg"

# ---- helpers --------------------------------------------------------------

def read_csv(path: Path):
    """Load CSV as list[dict]; supports UTF-8 with BOM."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames or []

def pick_label(row: dict) -> str:
    """Choose a human label for the row (prefer 'en')."""
    for k in (
        "en",
        "name_en",
        "section_name_en",
        "category_name_en",
        "subcategory_name_en",
        "label",
        "title",
    ):
        v = (row.get(k) or "").strip()
        if v:
            return v
    # fallback: first non-id, non-icon, non-empty field
    for k, v in row.items():
        if k.endswith("_id") or k == "icon" or not v:
            continue
        return str(v).strip()
    return ""

# ---- main ----------------------------------------------------------------

def main() -> int:
    # Check files
    missing = [p for p in (SECTIONS_CSV, CATEGORIES_CSV, SUBCATEGORIES_CSV) if not p.exists()]
    if missing:
        print("[ERROR] Missing CSVs:", ", ".join(str(p) for p in missing), file=sys.stderr)
        return 2

    sections, _ = read_csv(SECTIONS_CSV)
    categories, _ = read_csv(CATEGORIES_CSV)
    subs, _      = read_csv(SUBCATEGORIES_CSV)

    # Index for inheritance
    sec_by_id = { s.get("section_id"): s for s in sections }
    cat_by_id = { c.get("category_id"): c for c in categories }

    out_rows = []

    # Sections
    for s in sections:
        sec_icon = (s.get("icon") or "").strip()
        icon = sec_icon or DEFAULT_ICON
        source = "section" if sec_icon else "default"
        out_rows.append({
            "level": "section",
            "id": s.get("section_id",""),
            "parent_id": "",
            "label": pick_label(s),
            "resolved_icon": icon,
            "source": source,
        })

    # Categories
    for c in categories:
        sec = sec_by_id.get(c.get("section_id"))
        cat_icon = (c.get("icon") or "").strip()
        sec_icon = (sec.get("icon") or "").strip() if sec else ""
        if cat_icon:
            icon, source = cat_icon, "category"
        elif sec_icon:
            icon, source = sec_icon, "section"
        else:
            icon, source = DEFAULT_ICON, "default"
        out_rows.append({
            "level": "category",
            "id": c.get("category_id",""),
            "parent_id": c.get("section_id",""),
            "label": pick_label(c),
            "resolved_icon": icon,
            "source": source,
        })

    # Subcategories
    for g in subs:
        cat = cat_by_id.get(g.get("category_id"))
        sec = sec_by_id.get(cat.get("section_id")) if cat else None

        sub_icon = (g.get("icon") or "").strip()
        cat_icon = (cat.get("icon") or "").strip() if cat else ""
        sec_icon = (sec.get("icon") or "").strip() if sec else ""

        if sub_icon:
            icon, source = sub_icon, "subcategory"
        elif cat_icon:
            icon, source = cat_icon, "category"
        elif sec_icon:
            icon, source = sec_icon, "section"
        else:
            icon, source = DEFAULT_ICON, "default"

        out_rows.append({
            "level": "subcategory",
            "id": g.get("subcategory_id",""),
            "parent_id": g.get("category_id",""),
            "label": pick_label(g),
            "resolved_icon": icon,
            "source": source,
        })

    # Write output
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["level", "id", "parent_id", "label", "resolved_icon", "source"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {OUT_CSV} ({len(out_rows)} rows).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
