from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Where your generated CSVs live
DATA_DIR = Path("data")

SECTIONS_CSV      = DATA_DIR / "sections.csv"
CATEGORIES_CSV    = DATA_DIR / "categories.csv"
SUBCATEGORIES_CSV = DATA_DIR / "subcategories.csv"

# ---------- data models ----------

@dataclass
class Node:
    id: str
    type: str  # "section" | "category" | "subcategory"
    label: str
    icon: str = ""
    children: List["Node"] = field(default_factory=list)

# ---------- loading & helpers ----------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def _pick_label(row: Dict[str, str], lang: str, fallback_col: str) -> str:
    # prefer the language column if present & non-empty; else fallback to English name col; else 'en'
    if lang in row and row[lang]:
        return row[lang].strip()
    if fallback_col in row and row[fallback_col]:
        return row[fallback_col].strip()
    if "en" in row and row["en"]:
        return row["en"].strip()
    # last resort: any non-id value
    for k, v in row.items():
        if k not in ("section_id","category_id","subcategory_id","icon") and v:
            return v.strip()
    return ""

def load_tables():
    sections   = _read_csv(SECTIONS_CSV)
    categories = _read_csv(CATEGORIES_CSV)
    subs       = _read_csv(SUBCATEGORIES_CSV)
    return sections, categories, subs

def build_tree(lang: str = "en") -> List[Node]:
    sections, categories, subs = load_tables()

    # index for grouping / lookups
    cats_by_section: Dict[str, List[Dict[str, str]]] = {}
    for c in categories:
        cats_by_section.setdefault(c["section_id"], []).append(c)

    subs_by_category: Dict[str, List[Dict[str, str]]] = {}
    for s in subs:
        subs_by_category.setdefault(s["category_id"], []).append(s)

    # sort helpers
    def sort_key(row: Dict[str, str], sort_col: str) -> tuple:
        v = row.get(sort_col, "")
        # empty sort_order goes to the end; then alpha by English
        return (v == "" or v is None, str(v), row.get("en", ""))

    # build nodes
    tree: List[Node] = []
    for s in sorted(sections, key=lambda r: sort_key(r, "sort_order_section")):
        s_node = Node(
            id=s["section_id"],
            type="section",
            label=_pick_label(s, lang, "section_name_en"),
            icon=s.get("icon","") or "",
            children=[]
        )

        for c in sorted(cats_by_section.get(s["section_id"], []), key=lambda r: sort_key(r, "sort_order_category")):
            c_node = Node(
                id=c["category_id"],
                type="category",
                label=_pick_label(c, lang, "category_name_en"),
                icon=c.get("icon","") or "",
                children=[]
            )

            for g in sorted(subs_by_category.get(c["category_id"], []), key=lambda r: sort_key(r, "sort_order_subcategory")):
                c_node.children.append(Node(
                    id=g["subcategory_id"],
                    type="subcategory",
                    label=_pick_label(g, lang, "subcategory_name_en"),
                    icon=g.get("icon","") or "",
                    children=[]
                ))

            s_node.children.append(c_node)
        tree.append(s_node)

    return tree

# ---------- extras ----------

def flatten(tree: List[Node]) -> List[Node]:
    out: List[Node] = []
    def walk(n: Node):
        out.append(n)
        for ch in n.children:
            walk(ch)
    for root in tree:
        walk(root)
    return out

def id_to_label(lang: str = "en") -> Dict[str, str]:
    """Quick map of *any* node ID -> localized label."""
    table: Dict[str, str] = {}
    for n in flatten(build_tree(lang)):
        table[n.id] = n.label
    return table

def search(q: str, lang: str = "en") -> List[Node]:
    qn = q.strip().lower()
    if not qn:
        return []
    return [n for n in flatten(build_tree(lang)) if qn in n.label.lower()]
