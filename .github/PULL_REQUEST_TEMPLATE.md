## 📦 Data Update Summary

### What Changed
<!-- Briefly describe the updates in this PR -->
- Updated: **Sections.xlsx**
- Updated: **Categories.xlsx**
- Updated: **Subcategories.xlsx**
- Added/Changed icons or sort orders
- Validated foreign keys and relationships

---

### ✅ Checks Before Merge
- [ ] **Data Validation** – No missing or invalid IDs
- [ ] **Foreign Key Check** – All `section_id` / `category_id` match hierarchy
- [ ] **Language Check** – All translations in required languages present
- [ ] **Icon & Sort Order** – Confirmed all rows have correct icon + order
- [ ] **Scripts Run** – `validate_and_emit.py` and `make_json.py` executed successfully

---

### 🔗 Related Files in This PR
- `/data/SECTIONS.xlsx`
- `/data/CATEGORIES.xlsx`
- `/data/SUBCATEGORIES.xlsx`
- `/data/services_hierarchy_extended.csv`
- `/scripts/validate_and_emit.py`
- `/scripts/make_json.py`

---

### 🗒 Notes for Reviewers
<!-- Optional: Add any special instructions, known issues, or review notes -->

---

**Generated Files After Merge:**
- `sections.csv`
- `categories.csv`
- `subcategories.csv`
- `taxonomy.json`
