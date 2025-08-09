#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys, csv, re

DATA_DIR  = Path("data")
ICONS_DIR = Path("assets/icons")
SECTIONS_CSV      = DATA_DIR / "sections.csv"
CATEGORIES_CSV    = DATA_DIR / "categories.csv"
SUBCATEGORIES_CSV = DATA_DIR / "subcategories.csv"
REPORT_CSV        = DATA_DIR / "icon_report.csv"

DEFAULT_ICON = "/assets/icons/default.svg"
EXTERNAL_RE = re.compile(r"^(https?:)?//|^data:", re.I)

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []

def pick_label(row: dict):
    for key in ("en","name_en","label","title"):
        v = row.get(key, "").strip()
        if v: return v
    for k,v in row.items():
        if k.endswith("_id"): 
            continue
        if v and k not in ("icon","sort_order_section","sort_order_category","sort_order_subcategory"):
            return v.strip()
    return ""

def to_repo_path(icon_val: str) -> str | Path | None:
    if not icon_val:
        return None
    v = icon_val.strip()
    if not v:
        return None
    if EXTERNAL_RE.match(v):
        return "external"
    if v.startswith("/"):
        return Path(v.lstrip("/"))
    if "/" in v:
        return Path(v)
    return ICONS_DIR / v

def file_exists(target: str | Path | None) -> bool:
    if target in (None, "external"):
        return True
    return Path(target).exists()

def main() -> int:
    if not (SECTIONS_CSV.exists() and CATEGORIES_CSV.exists() and SUBCATEGORIES_CSV.exists()):
        print("[ERROR] One or more data CSVs are missing under /data.", file=sys.stderr)
        print("Expected:", SECTIONS_CSV, CATEGORIES_CSV, SUBCATEGORIES_CSV, file=sys.stderr)
        return 2

    sections, _ = read_csv(SECTIONS_CSV)
    categories, _ = read_csv(CATEGORIES_CSV)
    subs, _ = read_csv(SUBCATEGORIES_CSV)

    sec_by_id = { s.get("section_id"): s for s in sections }
    cat_by_id = { c.get("category_id"): c for c in categories }

    def resolved_for_sub(g: dict):
        cat = cat_by_id.get(g.get("category_id") or "")
        sec = sec_by_id.get(cat.get("section_id") if cat else "")
        for level, row in (("subcategory", g), ("category", cat or {}), ("section", sec or {})):
            icon_val = (row or {}).get("icon","")
            if icon_val:
                return (icon_val, level)
        return (DEFAULT_ICON, "default")

    def resolved_for_cat(c: dict):
        sec = sec_by_id.get(c.get("section_id") or "")
        for level, row in (("category", c), ("section", sec or {})):
            icon_val = (row or {}).get("icon","")
            if icon_val:
                return (icon_val, level)
        return (DEFAULT_ICON, "default")

    errors, warnings, report = [], [], []

    for s in sections:
        icon_val = s.get("icon","").strip()
        res = icon_val or DEFAULT_ICON
        src = "section" if icon_val else "default"
        target = to_repo_path(res)
        exists = file_exists(target)
        if icon_val and not exists:
            errors.append(f"[sections] {s.get('section_id')} → missing icon file: {target}")
        if not icon_val:
            warnings.append(f"[sections] {s.get('section_id')} has blank icon (uses default)")
        report.append({"level":"section","id":s.get("section_id",""),"label":pick_label(s),
                       "icon_set":bool(icon_val),"resolved_icon":res,"source":src,"exists":exists})

    for c in categories:
        icon_val, src = resolved_for_cat(c)
        target = to_repo_path(icon_val)
        exists = file_exists(target)
        if c.get("icon","").strip() and not exists:
            errors.append(f"[categories] {c.get('category_id')} → missing icon file: {target}")
        if src == "default":
            warnings.append(f"[categories] {c.get('category_id')} has no icon and inherits default")
        report.append({"level":"category","id":c.get("category_id",""),"label":pick_label(c),
                       "icon_set":bool(c.get('icon','').strip()),"resolved_icon":icon_val,"source":src,"exists":exists})

    for g in subs:
        icon_val, src = resolved_for_sub(g)
        target = to_repo_path(icon_val)
        exists = file_exists(target)
        if g.get("icon","").strip() and not exists:
            errors.append(f"[subcategories] {g.get('subcategory_id')} → missing icon file: {target}")
        if src == "default":
            warnings.append(f"[subcategories] {g.get('subcategory_id')} has no icon and inherits default")
        report.append({"level":"subcategory","id":g.get("subcategory_id",""),"label":pick_label(g),
                       "icon_set":bool(g.get('icon','').strip()),"resolved_icon":icon_val,"source":src,"exists":exists})

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level","id","label","icon_set","resolved_icon","source","exists"])
        writer.writeheader()
        writer.writerows(report)

    print(f"Report written to {REPORT_CSV} ({len(report)} rows).")
    if warnings:
        print("\\n== WARNINGS ==")
        for w in warnings[:50]: print(" -", w)
        if len(warnings) > 50: print(f" ... and {len(warnings)-50} more.")
    if errors:
        print("\\n== ERRORS ==")
        for e in errors: print(" -", e)
        print("\\nFAIL: missing icon files detected.")
        return 1
    print("\\n✓ No missing icon files. Inheritance & defaults applied.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
