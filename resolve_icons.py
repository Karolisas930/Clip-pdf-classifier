#!/usr/bin/env python3
from pathlib import Path
import csv

DATA_DIR  = Path("data")
SECTIONS_CSV      = DATA_DIR / "sections.csv"
CATEGORIES_CSV    = DATA_DIR / "categories.csv"
SUBCATEGORIES_CSV = DATA_DIR / "subcategories.csv"
OUT_CSV           = DATA_DIR / "icon_resolved_flat.csv"

DEFAULT_ICON = "/assets/icons/default.svg"

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames or []

def pick_label(row: dict):
    for k in ("en","name_en","label","title"):
        v = (row.get(k) or "").strip()
        if v: return v
    for k,v in row.items():
        if not k.endswith("_id") and k != "icon" and v:
            return str(v)
    return ""

def main():
    sections, _ = read_csv(SECTIONS_CSV)
    categories, _ = read_csv(CATEGORIES_CSV)
    subs, _      = read_csv(SUBCATEGORIES_CSV)

    sec_by_id = { s.get("section_id"): s for s in sections }
    cat_by_id = { c.get("category_id"): c for c in categories }

    rows = []

    for s in sections:
        rows.append({"level":"section","id":s.get("section_id",""),"label":pick_label(s),
                     "resolved_icon": s.get("icon","") or DEFAULT_ICON,
                     "source":"section" if s.get("icon","") else "default"})

    for c in categories:
        sec = sec_by_id.get(c.get("section_id"))
        if c.get("icon",""):
            icon, src = c["icon"], "category"
        elif sec and sec.get("icon",""):
            icon, src = sec["icon"], "section"
        else:
            icon, src = DEFAULT_ICON, "default"
        rows.append({"level":"category","id":c.get("category_id",""),"label":pick_label(c),
                     "resolved_icon":icon,"source":src})

    for g in subs:
        cat = cat_by_id.get(g.get("category_id"))
        sec = sec_by_id.get(cat.get("section_id")) if cat else None
        if g.get("icon",""):
            icon, src = g["icon"], "subcategory"
        elif cat and cat.get("icon",""):
            icon, src = cat["icon"], "category"
        elif sec and sec.get("icon",""):
            icon, src = sec["icon"], "section"
        else:
            icon, src = DEFAULT_ICON, "default"
        rows.append({"level":"subcategory","id":g.get("subcategory_id",""),"label":pick_label(g),
                     "resolved_icon":icon,"source":src})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level","id","label","resolved_icon","source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV} ({len(rows)} rows).")

if __name__ == "__main__":
    main()
