#!/usr/bin/env python3
"""
Build the page-level inspection tool for the benchmark site.

Adapts the archived inspector (archive_sds/inspector_templates/) for the
benchmark site. Scans a data directory and produces site/templates.html
with the full interactive SPA (sidebar nav, split-view, bbox overlays,
annotation table with pagination).

Usage:
    python site/build_inspector.py --data-dir synthetic-document-dataset/run2
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────
STANDARD_TYPES = {"PERSON", "ORG", "DATE", "ADDRESS", "AMOUNT", "PHONE", "ID"}


def _human_label(template_type: str) -> str:
    """Convert template_type to human-readable label."""
    special = {
        "aia_g702_g703": "AIA G702/G703",
        "cms1500": "CMS-1500",
        "ub04_hospital_claims": "UB-04 Hospital Claims",
        "irs_form_1040": "IRS Form 1040",
        "w2_form": "W-2 Form",
        "w4_form": "W-4 Form",
        "w9_form": "W-9 Form",
        "i9_form": "I-9 Form",
        "nda": "NDA",
        "msds": "MSDS",
        "cms485_home_health": "CMS-485 Home Health",
        "adp_report": "ADP Report",
        "irs_form": "IRS Form",
        "irs_tax_transcript": "IRS Tax Transcript",
        "irs_transcript": "IRS Transcript",
        "aia_construction_pay_app": "AIA Construction Pay App",
        "schedule_k1": "Schedule K-1",
        "schedule_k1_1065": "Schedule K-1 (1065)",
        "tax_1099": "Tax 1099",
        "uspto_document": "USPTO Document",
    }
    return special.get(template_type, template_type.replace("_", " ").title())


def get_template_category(template_type: str) -> str:
    """Determine category for a template based on name patterns."""
    t = template_type.lower()

    if any(x in t for x in ["medical", "cms", "prescription", "patient", "lab_report",
                             "hospital", "health", "eob", "explanation_of_benefits", "after_visit"]):
        return "Healthcare"
    if any(x in t for x in ["legal", "pleading", "affidavit", "court", "deed", "agreement",
                             "contract", "nda", "power_of_attorney", "will", "patent", "trademark"]):
        return "Legal"
    if any(x in t for x in ["bank", "invoice", "statement", "loan", "insurance", "claim",
                             "purchase_order", "credit", "payroll", "pay_stub", "remittance"]):
        return "Financial"
    if any(x in t for x in ["tax", "irs", "w2", "w4", "w9", "i9", "1040", "1099",
                             "schedule_k1", "transcript"]):
        return "Tax & Government"
    if any(x in t for x in ["passport", "driver_license", "birth_certificate", "visa",
                             "social_security", "voter", "diploma", "certificate"]):
        return "Identity Documents"
    if any(x in t for x in ["flight", "itinerary", "ticket", "event", "restaurant", "hotel"]):
        return "Travel & Hospitality"
    if any(x in t for x in ["blueprint", "building", "construction", "permit", "site_daily",
                             "change_order", "aia"]):
        return "Construction"
    if any(x in t for x in ["work_order", "timesheet", "meeting_minutes", "order_confirmation",
                             "shipping", "delivery", "fax", "business_card"]):
        return "Business Operations"
    if any(x in t for x in ["lease", "rental", "closing", "warranty_deed", "rent_roll"]):
        return "Real Estate"
    return "General"


def make_thumbnail(img: Image.Image, max_w: int = 300) -> Image.Image:
    """Resize image to fit within max width, preserving aspect ratio."""
    if img.width <= max_w:
        return img.copy()
    ratio = max_w / img.width
    new_h = int(img.height * ratio)
    return img.resize((max_w, new_h), Image.LANCZOS)


# ── Data pipeline ────────────────────────────────────────────────────────────

def scan_data_dir(data_dir: Path, site_dir: Path):
    """Scan labels and build template data in the format the archived viewer expects.

    Returns (templates_data, categories) where templates_data is a list of dicts
    matching the schema expected by viewer.js buildTemplateHTML().
    """
    labels_dir = data_dir / "labels"
    images_dir = data_dir / "images"

    if not labels_dir.exists():
        print(f"Error: {labels_dir} not found")
        sys.exit(1)

    # Group label files by doc_id (strip _page_N suffix)
    doc_pages = defaultdict(list)  # doc_id -> [(page_num, label_file, data, stem)]
    for lf in sorted(labels_dir.glob("*.json")):
        with open(lf, encoding="utf-8") as f:
            data = json.load(f)

        stem = lf.stem
        img_path = images_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue

        if "_page_" in stem:
            doc_id = stem.rsplit("_page_", 1)[0]
        else:
            doc_id = stem

        page_num = data.get("page_number", 1)
        doc_pages[doc_id].append((page_num, lf, data, stem))

    # Pick one doc per template code (first doc_id per 3-char prefix)
    template_code_map = {}
    for doc_id in sorted(doc_pages.keys()):
        code = doc_id[:3]
        if code not in template_code_map:
            template_code_map[code] = doc_id

    repo_root = site_dir.parent

    templates_data = []
    categories = defaultdict(list)

    for code in sorted(template_code_map.keys()):
        doc_id = template_code_map[code]
        pages = sorted(doc_pages[doc_id], key=lambda x: x[0])
        first_data = pages[0][2]

        template_type = first_data.get("template_type", "unknown")
        display_name = _human_label(template_type)
        category = get_template_category(template_type)
        total_pages = len(pages)
        orig_width = first_data.get("image_width", first_data.get("width", 816))
        orig_height = first_data.get("image_height", first_data.get("height", 1056))

        # Collect all annotations, flat list with page field
        # _idx is the global index used by viewer.js for bbox↔table interaction
        all_annotations = []
        for page_num, lf, data, stem in pages:
            for ann in data.get("annotations", []):
                if len(ann.get("box_2d", [])) != 4:
                    continue
                all_annotations.append({
                    "_idx": len(all_annotations),
                    "label": ann.get("label", ""),
                    "entity_type": ann["entity_type"],
                    "text_content": (ann.get("text_content") or "")[:120],
                    "box_2d": ann["box_2d"],
                    "page": page_num,
                    "section_id": ann.get("section_id"),
                    "field_id": ann.get("field_id"),
                })

        # Collect all sections (DoCO-FD FormSection)
        all_sections = []
        for page_num, lf, data, stem in pages:
            for sec in data.get("sections", []):
                if len(sec.get("box_2d", [])) != 4:
                    continue
                all_sections.append({
                    "_idx": len(all_sections),
                    "section_id": sec.get("section_id", ""),
                    "section_type": sec.get("section_type", "FormSection"),
                    "box_2d": sec["box_2d"],
                    "page": page_num,
                })

        # Collect all fields (DoCO-FD FormField)
        all_fields = []
        for page_num, lf, data, stem in pages:
            for fld in data.get("fields", []):
                if len(fld.get("box_2d", [])) != 4:
                    continue
                all_fields.append({
                    "_idx": len(all_fields),
                    "field_id": fld.get("field_id", ""),
                    "field_type": fld.get("field_type", "FormField"),
                    "section_id": fld.get("section_id"),
                    "box_2d": fld["box_2d"],
                    "page": page_num,
                })

        unique_pages = sorted(set(a["page"] for a in all_annotations)) or list(range(1, total_pages + 1))
        unique_labels = sorted(set(a["label"] for a in all_annotations if a["label"]))
        unique_types = sorted(set(a["entity_type"] for a in all_annotations if a["entity_type"]))

        # Build page data arrays
        page_buttons_data = []
        page_containers_data = []
        for page_num, lf, data, stem in pages:
            # Thumbnail: relative from site/templates.html
            thumb_src = f"assets/inspector/thumbs/{stem}.jpg"

            # Full image: relative from site/ to data_dir/images/
            img_abs = (images_dir / f"{stem}.jpg").resolve()
            try:
                rel = img_abs.relative_to(repo_root.resolve())
                img_src = "../" + str(rel).replace("\\", "/")
            except ValueError:
                img_src = str(img_abs).replace("\\", "/")

            page_buttons_data.append({
                "page_num": page_num,
                "thumb_src": thumb_src,
            })
            page_containers_data.append({
                "page_num": page_num,
                "img_src": img_src,
                "annotations": [a for a in all_annotations if a["page"] == page_num],
                "fields": [f for f in all_fields if f["page"] == page_num],
            })

        template_data = {
            "template_name": code,
            "display_name": display_name,
            "doc_id": doc_id,
            "category": category,
            "template_type": template_type,
            "total_pages": total_pages,
            "annotation_count": len(all_annotations),
            "orig_width": orig_width,
            "orig_height": orig_height,
            "annotations": all_annotations,
            "sections": all_sections,
            "fields": all_fields,
            "unique_pages": unique_pages,
            "unique_labels": unique_labels,
            "unique_types": unique_types,
            "page_buttons_data": page_buttons_data,
            "page_containers_data": page_containers_data,
        }
        templates_data.append(template_data)
        categories[category].append(template_data)

    return templates_data, categories


def build_thumbnails(templates_data: list, images_dir: Path, thumbs_dir: Path):
    """Generate thumbnails for all pages."""
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for tmpl in templates_data:
        for btn in tmpl["page_buttons_data"]:
            stem = btn["thumb_src"].rsplit("/", 1)[-1].replace(".jpg", "")
            thumb_path = thumbs_dir / f"{stem}.jpg"
            if thumb_path.exists():
                count += 1
                continue
            img_path = images_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue
            img = Image.open(img_path)
            thumb = make_thumbnail(img, 300)
            thumb.save(thumb_path, quality=80)
            count += 1
    return count


# ── HTML generation (uses archived inspector templates) ──────────────────────

SITE_NAV = """\
<nav style="position:fixed;top:0;left:0;right:0;z-index:1001;
     background:#1a1d27;border-bottom:1px solid #2e3348;
     padding:0 24px;display:flex;align-items:center;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <a href="index.html" style="font-weight:700;font-size:.95rem;padding:10px 16px 10px 0;
     border-right:1px solid #2e3348;margin-right:4px;color:#e2e4eb;text-decoration:none;">SDD Benchmark</a>
  <a href="index.html" style="color:#9498ab;text-decoration:none;font-size:.85rem;padding:10px 14px;">Home</a>
  <a href="templates.html" style="color:#6366f1;text-decoration:none;font-size:.85rem;
     padding:10px 14px;border-bottom:2px solid #6366f1;">Templates</a>
  <a href="paper.html" style="color:#9498ab;text-decoration:none;font-size:.85rem;padding:10px 14px;">Paper</a>
</nav>"""

SITE_NAV_HEIGHT = 41  # px


def build_html(templates_data: list, categories: dict, base_html_path: Path, site_dir: Path):
    """Build templates.html from existing base + embedded data."""
    base_html = base_html_path.read_text(encoding="utf-8")

    # Build new sidebar navigation
    nav_html = '<h2>Templates</h2>\n'
    for category in sorted(categories.keys()):
        cat_id = category.lower().replace(" ", "-").replace("&", "and")
        cat_items = sorted(categories[category], key=lambda x: x["display_name"])
        nav_html += f'''<div class="nav-category">
  <div class="nav-category-header" onclick="toggleCategory('{cat_id}')">
    <span class="category-icon">▶</span>
    <span class="category-title">{category}</span>
    <span class="category-count">({len(cat_items)})</span>
  </div>
  <ul class="nav-category-items collapsed" id="cat-{cat_id}">
'''
        for tmpl in cat_items:
            name = tmpl["template_name"]
            display = tmpl["display_name"]
            doc_id = tmpl["doc_id"]
            nav_html += f'    <li><a href="#{name}" data-template="{name}">{display}<span class="template-meta">{doc_id}</span></a></li>\n'
        nav_html += '  </ul>\n</div>\n'

    # Replace sidebar nav content (between <div class="sidebar-nav"> and </div>)
    base_html = re.sub(
        r'(<div class="sidebar-nav">).*?(</div>\s*<!-- end sidebar -->)',
        rf'\1\n{nav_html}</div><!-- end sidebar -->',
        base_html,
        flags=re.DOTALL
    )

    # Build template_type → code lookup for deep-linking from gallery
    type_to_code = {}
    for tmpl in templates_data:
        type_to_code[tmpl["template_type"]] = tmpl["template_name"]

    # Replace templates data JSON using string find/replace instead of regex
    json_str = json.dumps(templates_data)
    old_json_start = '<script id="templates-data" type="application/json">'
    old_json_end = '</script>'
    
    start_idx = base_html.find(old_json_start)
    if start_idx != -1:
        end_idx = base_html.find(old_json_end, start_idx)
        if end_idx != -1:
            new_json = f'{old_json_start}\n{json_str}\n{old_json_end}'
            base_html = base_html[:start_idx] + new_json + base_html[end_idx + len(old_json_end):]

    # Update type→code lookup
    lookup_script = f'<script>var TYPE_TO_CODE = {json.dumps(type_to_code)};</script>'
    old_lookup = base_html.find('<script>var TYPE_TO_CODE = ')
    if old_lookup != -1:
        end_lookup = base_html.find('</script>', old_lookup)
        if end_lookup != -1:
            base_html = base_html[:old_lookup] + lookup_script + base_html[end_lookup + 9:]

    # Write output
    out_path = site_dir / "templates.html"
    out_path.write_text(base_html, encoding="utf-8")
    print(f"  templates.html -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build inspection tool for benchmark site")
    parser.add_argument("--data-dir", required=True, help="Path to data directory (images/ + labels/)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)

    script_dir = Path(__file__).parent
    thumbs_dir = script_dir / "assets" / "inspector" / "thumbs"

    # Use existing templates.html as base instead of archive
    base_html_path = script_dir / "templates.html"
    if not base_html_path.exists():
        print(f"Error: {base_html_path} not found")
        sys.exit(1)

    print("Scanning data directory...")
    templates_data, categories = scan_data_dir(data_dir, script_dir)
    total_pages = sum(t["total_pages"] for t in templates_data)
    total_ann = sum(t["annotation_count"] for t in templates_data)
    print(f"  Found {len(templates_data)} templates, {total_pages} pages, {total_ann} annotations")

    print("\nGenerating thumbnails...")
    n = build_thumbnails(templates_data, data_dir / "images", thumbs_dir)
    print(f"  {n} thumbnails")

    print("\nBuilding templates.html...")
    build_html(templates_data, categories, base_html_path, script_dir)

    print("\nDone! Open site/templates.html in your browser.")


if __name__ == "__main__":
    main()
