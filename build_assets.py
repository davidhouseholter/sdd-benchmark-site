#!/usr/bin/env python3
"""
Build static image assets for the benchmark site.

Generates:
  - site/assets/gallery/{template_type}.jpg           — original thumbnail
  - site/assets/gallery/{template_type}_overlay.jpg    — GT bounding boxes
  - site/assets/gallery/{template_type}_prediction.jpg — NER results (hit/miss)
  - site/assets/examples/{name}.jpg                    — cropped comparison regions

Usage:
    python site/build_assets.py --run synthetic-document-dataset/run1
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


# ── Color map for entity types (GT overlay) ──────────────────────────────────
ENTITY_COLORS = {
    # Standard NER types
    "PERSON": "#a855f7",   # purple
    "ORG": "#3b82f6",      # blue
    "DATE": "#14b8a6",     # teal
    "AMOUNT": "#f97316",   # orange
    "ID": "#eab308",       # yellow
    "PHONE": "#22c55e",    # green
    "ADDRESS": "#ef4444",  # red
    "EMAIL": "#06b6d4",    # cyan
    "LOCATION": "#8b5cf6", # violet
    "QUANTITY": "#d946ef", # fuchsia
    "URL": "#0ea5e9",      # sky
    "DURATION": "#84cc16", # lime
    # Structural / document layout types
    "TABLE_CELL": "#94a3b8",
    "TABLE_HEADER": "#64748b",
    "FIELD_LABEL": "#cbd5e1",
    "CAPTION": "#475569",
    "SECTION_TITLE": "#334155",
    "TEXT": "#9ca3af",
    "TITLE": "#1e293b",
}
DEFAULT_COLOR = "#ef4444"

# ── Colors for prediction overlay (match result) ─────────────────────────────
MATCH_COLORS = {
    "hit": "#22c55e",           # green
    "miss": "#ef4444",          # red
    "type_confusion": "#f59e0b", # amber
}

STANDARD_TYPES = {"PERSON", "ORG", "DATE", "ADDRESS", "AMOUNT", "PHONE", "ID",
                  "LOCATION", "QUANTITY", "EMAIL", "URL", "DURATION"}
STRUCTURAL_TYPES = {"TABLE_CELL", "TABLE_HEADER", "FIELD_LABEL", "TITLE", "SECTION_TITLE", "TEXT", "CAPTION"}


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def draw_overlay(img: Image.Image, annotations: list, width: int, height: int,
                 standard_only: bool = False, structural_only: bool = False) -> Image.Image:
    """Draw bounding boxes colored by entity type on a copy of the image."""
    out = img.copy()
    scale_x = img.width / 1000.0
    scale_y = img.height / 1000.0

    for ann in annotations:
        etype = ann.get("entity_type", "")
        if standard_only and etype not in STANDARD_TYPES:
            continue
        if structural_only and etype not in STRUCTURAL_TYPES:
            continue

        box = ann.get("box_2d", [])
        if len(box) != 4:
            continue
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        color = hex_to_rgb(ENTITY_COLORS.get(etype, DEFAULT_COLOR))

        overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x1, y1, x2, y2], fill=color + (40,), outline=color + (200,), width=2)
        out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")

    return out


def draw_prediction_overlay(img: Image.Image, annotations: list,
                            match_lookup: dict, page: str,
                            width: int, height: int) -> Image.Image:
    """Draw bounding boxes colored by match result (hit/miss/type_confusion).

    Standard entity types: green (hit), red (miss), amber (type confusion).
    Structural types: drawn in neutral gray so all GT is visible.
    """
    out = img.copy()
    scale_x = img.width / 1000.0
    scale_y = img.height / 1000.0

    # Deep copy page matches so we don't mutate the lookup
    page_matches_orig = match_lookup.get(page, {})
    page_matches = {k: list(v) for k, v in page_matches_orig.items()}

    # Only draw standard types — structural types (FIELD_LABEL, TABLE_CELL, etc.)
    # are not evaluated by NER models, so showing them would be misleading.
    for ann in annotations:
        etype = ann.get("entity_type", "")
        if etype not in STANDARD_TYPES:
            continue

        box = ann.get("box_2d", [])
        if len(box) != 4:
            continue

        text = ann.get("text_content", "").strip()
        key = (etype, text)

        # Look up match result
        results = page_matches.get(key, [])
        if results:
            result = results.pop(0)
        else:
            result = "miss"

        color = hex_to_rgb(MATCH_COLORS.get(result, MATCH_COLORS["miss"]))
        if result == "hit":
            fill_alpha, outline_alpha, lw = 50, 220, 2
        elif result == "miss":
            fill_alpha, outline_alpha, lw = 120, 255, 3
        else:  # type_confusion
            fill_alpha, outline_alpha, lw = 100, 240, 3

        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x1, y1, x2, y2],
                               fill=color + (fill_alpha,),
                               outline=color + (outline_alpha,), width=lw)
        out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")

    return out


def load_entity_matches(run_dir: Path) -> dict:
    """Load entity_matches.csv into a nested lookup.

    Returns: { page: { (entity_type, gt_text): [result, ...] } }
    """
    csv_path = run_dir / "entity_matches.csv"
    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found, skipping prediction overlays")
        return {}

    lookup = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            page = row["page"]
            etype = row["entity_type"]
            gt_text = row["gt_entity"].strip()
            result = row["model_result"]
            lookup[page][(etype, gt_text)].append(result)

    return dict(lookup)


def compute_page_stats(match_lookup: dict, page: str) -> dict:
    """Count hits/misses/confusions for a page from the match lookup."""
    stats = {"hits": 0, "misses": 0, "confusions": 0}
    page_matches = match_lookup.get(page, {})
    for results in page_matches.values():
        for r in results:
            if r == "hit":
                stats["hits"] += 1
            elif r == "type_confusion":
                stats["confusions"] += 1
            else:
                stats["misses"] += 1
    return stats


def crop_region(img: Image.Image, box_norm: list, width: int, height: int, pad: int = 15) -> Image.Image:
    """Crop a region from the image using normalized 0-1000 coordinates with padding."""
    scale_x = img.width / 1000.0
    scale_y = img.height / 1000.0
    x1 = max(0, int(box_norm[0] * scale_x) - pad)
    y1 = max(0, int(box_norm[1] * scale_y) - pad)
    x2 = min(img.width, int(box_norm[2] * scale_x) + pad)
    y2 = min(img.height, int(box_norm[3] * scale_y) + pad)
    return img.crop((x1, y1, x2, y2))


def make_thumbnail(img: Image.Image, max_w: int = 400) -> Image.Image:
    """Resize image to fit within max width, preserving aspect ratio."""
    if img.width <= max_w:
        return img.copy()
    ratio = max_w / img.width
    new_h = int(img.height * ratio)
    return img.resize((max_w, new_h), Image.LANCZOS)


def build_gallery(run_dir: Path, out_dir: Path, match_lookup: dict):
    """Generate one thumbnail + GT overlay + prediction overlay per template type."""
    labels_dir = run_dir / "labels"
    images_dir = run_dir / "images"

    # Group label files by template_type
    templates = defaultdict(list)
    for lf in sorted(labels_dir.glob("*.json")):
        with open(lf) as f:
            data = json.load(f)
        ttype = data.get("template_type", "unknown")
        templates[ttype].append((lf, data))

    gallery_dir = out_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for ttype, items in sorted(templates.items()):
        # Pick first page 1 (or first file) as representative
        chosen = None
        for lf, data in items:
            if data.get("page_number", 1) == 1:
                chosen = (lf, data)
                break
        if chosen is None:
            chosen = items[0]

        lf, data = chosen
        stem = lf.stem
        img_path = images_dir / f"{stem}.jpg"
        if not img_path.exists():
            print(f"  skip {ttype}: {img_path.name} not found")
            continue

        img = Image.open(img_path)
        annotations = data.get("annotations", [])
        w = data.get("width", 816)
        h = data.get("height", 1056)

        # 1. Original thumbnail
        thumb = make_thumbnail(img, 400)
        thumb_path = gallery_dir / f"{ttype}.jpg"
        thumb.save(thumb_path, quality=85)

        # 2. GT overlay thumbnail — standard types only (matches prediction overlay)
        overlay_img = draw_overlay(img, annotations, w, h, standard_only=True)
        overlay_thumb = make_thumbnail(overlay_img, 400)
        overlay_path = gallery_dir / f"{ttype}_overlay.jpg"
        overlay_thumb.save(overlay_path, quality=85)

        # 3. Structural overlay — structural types only (no NER entities)
        structural_img = draw_overlay(img, annotations, w, h, structural_only=True)
        structural_thumb = make_thumbnail(structural_img, 400)
        structural_path = gallery_dir / f"{ttype}_structural.jpg"
        structural_thumb.save(structural_path, quality=85)

        # 4. Prediction overlay thumbnail
        if match_lookup:
            pred_img = draw_prediction_overlay(img, annotations, match_lookup, stem, w, h)
            pred_thumb = make_thumbnail(pred_img, 400)
            pred_path = gallery_dir / f"{ttype}_prediction.jpg"
            pred_thumb.save(pred_path, quality=85)

        # Stats
        n_ann = len(annotations)
        n_standard = sum(1 for a in annotations if a.get("entity_type") in STANDARD_TYPES)
        page_stats = compute_page_stats(match_lookup, stem)

        # Aggregate stats across all pages for this template
        total_stats = {"hits": 0, "confusions": 0}
        total_standard = 0
        for item_lf, item_data in items:
            ps = compute_page_stats(match_lookup, item_lf.stem)
            total_stats["hits"] += ps["hits"]
            total_stats["confusions"] += ps["confusions"]
            total_standard += sum(1 for a in item_data.get("annotations", [])
                                  if a.get("entity_type") in STANDARD_TYPES)
        # Misses = GT standard entities not accounted for by hits or confusions
        total_misses = total_standard - total_stats["hits"] - total_stats["confusions"]

        manifest.append({
            "template_type": ttype,
            "stem": stem,
            "annotations": n_ann,
            "standard_entities": n_standard,
            "total_standard": total_standard,
            "pages": data.get("total_pages", 1),
            "count": len(items),
            "hits": total_stats["hits"],
            "misses": total_misses,
            "confusions": total_stats["confusions"],
        })
        print(f"  {ttype}: {thumb_path.name} ({n_ann} ann, {page_stats['hits']} hits)")

    # Save manifest
    with open(gallery_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGallery: {len(manifest)} templates -> {gallery_dir}")
    return manifest


def _find_page1_stem(labels_dir: Path, template_type: str):
    """Find the first page-1 stem for a given template_type."""
    for lf in sorted(labels_dir.glob("*.json")):
        with open(lf) as f:
            data = json.load(f)
        if data.get("template_type") == template_type:
            if data.get("page_number", 1) == 1:
                return lf.stem
    return None


def _generate_example(run_dir, examples_dir, match_lookup, stem, name, crop_box=None):
    """Generate 3-phase images (original, GT overlay, prediction) for one example.

    If crop_box is given, generates cropped region. Otherwise generates full-page thumbnail.
    """
    labels_dir = run_dir / "labels"
    images_dir = run_dir / "images"

    label_path = labels_dir / f"{stem}.json"
    img_path = images_dir / f"{stem}.jpg"

    if not label_path.exists() or not img_path.exists():
        print(f"  {name}: file not found ({stem})")
        return False

    with open(label_path) as f:
        data = json.load(f)

    img = Image.open(img_path)
    annotations = data.get("annotations", [])
    w = data.get("width", 816)
    h = data.get("height", 1056)

    # Draw standard-only GT overlay (matches prediction overlay which also filters to standard)
    overlay_img = draw_overlay(img, annotations, w, h, standard_only=True)
    pred_img = draw_prediction_overlay(img, annotations, match_lookup, stem, w, h)

    if crop_box:
        crop_region(img, crop_box, w, h).save(examples_dir / f"{name}_original.jpg", quality=90)
        crop_region(overlay_img, crop_box, w, h).save(examples_dir / f"{name}_overlay.jpg", quality=90)
        crop_region(pred_img, crop_box, w, h).save(examples_dir / f"{name}_prediction.jpg", quality=90)
    else:
        make_thumbnail(img, 380).save(examples_dir / f"{name}_original.jpg", quality=85)
        make_thumbnail(overlay_img, 380).save(examples_dir / f"{name}_overlay.jpg", quality=85)
        make_thumbnail(pred_img, 380).save(examples_dir / f"{name}_prediction.jpg", quality=85)

    n_std = sum(1 for a in annotations if a.get("entity_type") in STANDARD_TYPES)
    print(f"  {name}: {stem} ({len(annotations)} ann, {n_std} standard)")
    return True


# Example configs: (template_type, name, crop_box_or_None)
# crop_box=None → full-page thumbnail; [x1,y1,x2,y2] → cropped region (0-1000 coords)
EXAMPLE_CONFIGS = [
    ("bank_statement", "header", [100, 55, 900, 130]),
    ("bank_statement", "table", [100, 190, 900, 410]),
    ("invoice", "invoice", None),
    ("cms1500", "cms1500", None),
    ("legal_pleading", "legal_pleading", None),
]


def build_examples(run_dir: Path, out_dir: Path, match_lookup: dict):
    """Generate 3-phase example images for the comparison section (5 examples)."""
    labels_dir = run_dir / "labels"
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    for template_type, name, crop_box in EXAMPLE_CONFIGS:
        stem = _find_page1_stem(labels_dir, template_type)
        if not stem:
            print(f"  {name}: no {template_type} found, skipping")
            continue
        _generate_example(run_dir, examples_dir, match_lookup, stem, name, crop_box)

    print(f"\nExamples: {examples_dir}")


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


def update_index_html(site_dir: Path, manifest: list, run_dir: Path):
    """Inject gallery cards and stats into index.html from manifest data."""
    import re
    index_path = site_dir / "index.html"

    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    # ── Build gallery cards HTML ────────────────────────────
    cards = []
    total_standard = 0
    total_pages = 0
    for item in sorted(manifest, key=lambda x: x["template_type"]):
        ttype = item["template_type"]
        label = _human_label(ttype)
        n_ann = item["annotations"]
        n_std = item["standard_entities"]
        t_std = item["total_standard"]
        hits = item.get("hits", 0)
        misses = item.get("misses", 0)
        confusions = item.get("confusions", 0)
        total = hits + misses + confusions
        total_standard += t_std
        count = item.get("count", 0)
        pages = item.get("pages", 1)
        total_pages += count  # count = number of label files = pages

        # Hit/miss bar widths
        if total > 0:
            hw = round(hits / total * 100, 1)
            cw = round(confusions / total * 100, 1)
            mw = round(100 - hw - cw, 1)
            f1_str = f"{2 * hits / (2 * hits + misses + confusions):.0%}" if (2 * hits + misses + confusions) > 0 else "—"
        else:
            hw, cw, mw = 0, 0, 100
            f1_str = "—"

        pred_meta = f"{hits} hit, {misses} miss" + (f", {confusions} confused" if confusions else "")

        n_structural = n_ann - n_std
        struct_meta = f"{n_structural} structural regions"

        card = (
            f'  <div class="gallery-card" data-original="assets/gallery/{ttype}.jpg" '
            f'data-gt="assets/gallery/{ttype}_overlay.jpg" '
            f'data-structural="assets/gallery/{ttype}_structural.jpg" '
            f'data-pred="assets/gallery/{ttype}_prediction.jpg" '
            f'data-gt-meta="{n_std} standard entities" '
            f'data-structural-meta="{struct_meta}" '
            f'data-pred-meta="{pred_meta}" '
            f'data-template-type="{ttype}">\n'
            f'    <div class="card-img-wrap"><img src="assets/gallery/{ttype}.jpg" alt="{label}"></div>\n'
            f'    <div class="card-label">{label}</div>\n'
            f'    <div class="card-meta">{n_ann} annotations</div>\n'
            f'    <div class="card-details">\n'
            f'      <div class="detail-row"><span>Standard entities</span><span class="detail-val">{t_std}</span></div>\n'
            f'      <div class="detail-row"><span>Docs / Pages</span><span class="detail-val">{count}</span></div>\n'
            f'      <div class="detail-row"><span>Hits</span><span class="detail-val" style="color:var(--green)">{hits}</span></div>\n'
            f'      <div class="detail-row"><span>Misses</span><span class="detail-val" style="color:var(--red)">{misses}</span></div>\n'
            f'      <div class="detail-row"><span>Type confusion</span><span class="detail-val" style="color:var(--amber)">{confusions}</span></div>\n'
            f'      <div class="detail-row"><span>F1 (approx)</span><span class="detail-val">{f1_str}</span></div>\n'
            f'      <div class="hit-miss-bar"><span class="bar-hit" style="width:{hw}%"></span>'
            f'<span class="bar-conf" style="width:{cw}%"></span>'
            f'<span class="bar-miss" style="width:{mw}%"></span></div>\n'
            f'    </div>\n'
            f'    <a class="card-inspect-link" href="templates.html#{ttype}">View in Inspector &rarr;</a>\n'
            f'  </div>'
        )
        cards.append(card)

    gallery_inner = "\n".join(cards)
    gallery_html = f'<div class="gallery" id="gallery">\n{gallery_inner}\n</div>'

    # Replace between GALLERY_START / GALLERY_END markers
    html = re.sub(
        r'<!-- GALLERY_START -->.*?<!-- GALLERY_END -->',
        f'<!-- GALLERY_START -->\n{gallery_html}\n<!-- GALLERY_END -->',
        html,
        flags=re.DOTALL,
    )

    # ── Update gallery description ──────────────────────────
    n_templates = len(manifest)
    html = re.sub(
        r'<!-- GALLERY_DESC_START -->.*?<!-- GALLERY_DESC_END -->',
        f'<!-- GALLERY_DESC_START -->\n<p>All {n_templates} document templates in the benchmark. '
        f'Use the toggle to switch views. Click a card label to expand details.</p>\n<!-- GALLERY_DESC_END -->',
        html,
        flags=re.DOTALL,
    )

    # ── Update stats cards ──────────────────────────────────
    stats_html = (
        '<div class="stat-grid">\n'
        f'  <div class="stat-card"><div class="num">{total_pages:,}</div><div class="label">Pages</div></div>\n'
        f'  <div class="stat-card"><div class="num">{n_templates}</div><div class="label">Templates</div></div>\n'
        f'  <div class="stat-card"><div class="num">7</div><div class="label">Entity Types</div></div>\n'
        f'  <div class="stat-card"><div class="num">{total_standard:,}</div><div class="label">GT Entities</div></div>\n'
        '</div>'
    )
    html = re.sub(
        r'<!-- STATS_START -->.*?<!-- STATS_END -->',
        f'<!-- STATS_START -->\n{stats_html}\n<!-- STATS_END -->',
        html,
        flags=re.DOTALL,
    )

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nUpdated index.html: {n_templates} gallery cards, {total_pages} pages, {total_standard:,} GT entities")


def main():
    parser = argparse.ArgumentParser(description="Build site image assets")
    parser.add_argument("--run", required=True, help="Path to run directory (e.g. synthetic-document-dataset/run1)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)

    # Output to site/assets/ relative to this script
    script_dir = Path(__file__).parent
    out_dir = script_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load NER match results
    print("Loading entity matches...")
    match_lookup = load_entity_matches(run_dir)
    if match_lookup:
        print(f"  Loaded matches for {len(match_lookup)} pages")
    else:
        print("  No match data — prediction overlays will be skipped")

    print("\nBuilding gallery thumbnails...")
    manifest = build_gallery(run_dir, out_dir, match_lookup)

    print("\nBuilding example cutouts...")
    build_examples(run_dir, out_dir, match_lookup)

    # Inject gallery HTML + stats into index.html
    update_index_html(script_dir, manifest, run_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
