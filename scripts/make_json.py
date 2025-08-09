from pathlib import Path
from utils import build_tree_from_disk, to_json

nodes = build_tree_from_disk("data")
out = Path("data/taxonomy.json")
out.write_text(to_json(nodes, lang="en", fallback="en"), encoding="utf-8")
print(f"Wrote {out.resolve()}")
