#!/usr/bin/env python3
"""
validate_schema.py
Quick checks for hierarchy + translations data files.

Usage:
  python scripts/validate_schema.py \
    --hier data/services_hierarchy_extended.csv \
    --trans data/translations.csv \
    --max-empty 200
Exit codes:
  0 = OK
  1 = Warnings only (non-fatal)
  2 = Errors (fail CI)
"""
import argparse, sys, csv, json, re
from collections import defaultdict

def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hier", required=True, help="Path to services_hierarchy*.csv")
    ap.add_argument("--trans", required=True, help="Path to translations.csv (key + language columns)")
    ap.add_argument("--max-empty", type=int, default=500, help="Warn if a language has more empty cells than this")
    args = ap.parse_args()

    errors = []
    warnings = []

    hier = read_csv(args.hier)
    if not hier:
        errors.append(f"HIER: file has no rows: {args.hier}")
    trans = read_csv(args.trans)
    if not trans:
        errors.append(f"TRANS: file has no rows: {args.trans}")

    # Detect ID columns in hierarchy
    id_cols = [c for c in (hier[0].keys() if hier else []) if c.endswith("_id")]
    if not id_cols:
        errors.append("HIER: no *_id columns found (expected section_id/category_id/subcategory_id)")

    # Collect all IDs from hierarchy
    ids = set()
    for r in hier:
        for c in id_cols:
            v = r.get(c, "").strip()
            if v:
                ids.add(v)

    # Basic uniqueness check for each id column
    for c in id_cols:
        seen = set()
        dups = set()
        for r in hier:
            v = r.get(c, "").strip()
            if not v:
                continue
            if v in seen:
                dups.add(v)
            else:
                seen.add(v)
        if dups:
            errors.append(f"HIER: duplicate values in {c}: {sorted(list(dups))[:10]} ... ({len(dups)} dup values)")

    # Trans columns
    lang_cols = [c for c in (trans[0].keys() if trans else []) if c != "key"]
    if "key" not in (trans[0].keys() if trans else []):
        errors.append("TRANS: missing 'key' column as first column")
    if not lang_cols:
        errors.append("TRANS: no language columns detected")

    # Coverage
    trans_keys = set(r.get("key","").strip() for r in trans if r.get("key","").strip())
    missing = ids - trans_keys
    extra = trans_keys - ids
    if missing:
        errors.append(f"COVERAGE: {len(missing)} hierarchy IDs missing from translations (showing up to 15): {sorted(list(missing))[:15]}")
    if extra:
        warnings.append(f"COVERAGE: {len(extra)} translation keys not present in hierarchy (up to 15): {sorted(list(extra))[:15]}")

    # Empty counts per language
    empties = {}
    for lc in lang_cols:
        empties[lc] = sum(1 for r in trans if not r.get(lc,"").strip())
        if empties[lc] > args.max_empty:
            warnings.append(f"LANG '{lc}' has {empties[lc]} empty cells (> max-empty {args.max_empty})")

    # Print summary JSON (useful in CI logs)
    summary = {
        "hier_rows": len(hier),
        "trans_rows": len(trans),
        "id_columns": id_cols,
        "languages": lang_cols,
        "missing_ids_count": len(missing),
        "extra_keys_count": len(extra),
        "empties": empties,
    }
    print(json.dumps(summary, indent=2))

    if errors:
        print("\n❌ ERRORS:")
        for e in errors:
            print(" -", e)
        sys.exit(2)

    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(" -", w)
        sys.exit(1)

    print("\n✅ Validation passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
