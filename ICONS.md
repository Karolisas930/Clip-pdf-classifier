# Icons & Inheritance

Place your icons in `/assets/icons/` (prefer SVG).
In CSVs (`data/sections.csv`, `data/categories.csv`, `data/subcategories.csv`), the `icon` field can be blank to inherit.

Cascade:
- Subcategory: sub.icon → cat.icon → sec.icon → `/assets/icons/default.svg`
- Category: cat.icon → sec.icon → default
- Section: sec.icon → default

Tools:
- `scripts/check_icons.py` — validates and writes `data/icon_report.csv`
- `scripts/resolve_icons.py` — writes `data/icon_resolved_flat.csv`

Run:
```
python scripts/check_icons.py
python scripts/resolve_icons.py
```