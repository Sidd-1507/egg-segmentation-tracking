"""Manual polygon refinement of Task 1 masks, for eggs SAM would not complete even with box + clicks.

Each entry in refinements.json adds (or removes) pixels inside a hand-drawn polygon for one egg:
  {"egg_010": [{"instance": 1, "mode": "add_white",
                "polygon": [[241, 281], [255, 274], ...],
                "note": "egg top behind the stem"}]}

Modes:
  add_white  add polygon pixels that look like eggshell (low saturation, bright); occluding
             leaves/stems stay excluded, matching how SAM treats occluders elsewhere
  add        add every polygon pixel
  remove     remove every polygon pixel

Then rewrites masks/, instances/, overlays/ for the refined images and rebuilds metadata.jsonl,
summary.json and review/ for the whole set. Needs only numpy + Pillow.

  python task1_sam/refine_masks.py --data_dir data/raw --out_dir task1_sam/output --refinements task1_sam/refinements.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

COLORS = [(0, 255, 0), (255, 64, 64), (64, 128, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255)]
MAX_SAT, MIN_VAL = 70, 150  # eggshell in HSV (0-255 scale)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--out_dir", default="task1_sam/output")
    p.add_argument("--refinements", default="task1_sam/refinements.json")
    return p.parse_args()


def raw_images(data_dir):
    out = {}
    for p in Path(data_dir).rglob("*.jpg"):
        m = re.search(r"not_damaged_(\d+)_jpg", p.name)
        out[f"egg_{int(m.group(1)):03d}"] = p
    return out


def polygon_mask(shape, polygon):
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(canvas).polygon([tuple(pt) for pt in polygon], fill=1)
    return np.array(canvas, bool)


def eggshell_pixels(img, max_sat=MAX_SAT, min_val=MIN_VAL):
    hsv = np.array(img.convert("HSV"))
    return (hsv[..., 1] < max_sat) & (hsv[..., 2] > min_val)


def draw_overlay(img, label):
    vis = np.array(img).copy()
    for k in range(1, int(label.max()) + 1):
        m = label == k
        c = np.array(COLORS[(k - 1) % len(COLORS)])
        vis[m] = (0.55 * vis[m] + 0.45 * c).astype(np.uint8)
        inner = np.array(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
        vis[m & ~inner] = c
    return Image.fromarray(vis)


def review_sheets(records, overlay_dir, review_dir, per_sheet=20, cols=5, tile=256):
    review_dir.mkdir(parents=True, exist_ok=True)
    for old in review_dir.glob("sheet_*.jpg"):
        old.unlink()
    for s in range(0, len(records), per_sheet):
        chunk = records[s:s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile, rows * (tile + 24)), "white")
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            x, y = (k % cols) * tile, (k // cols) * (tile + 24)
            sheet.paste(Image.open(overlay_dir / f"{r['stem']}.jpg").resize((tile, tile)), (x, y + 24))
            title = f"{r['stem']} n={r['n_eggs']} " + ",".join(r["flags"])
            d.text((x + 4, y + 6), title[:42], fill=(220, 0, 0) if r["flags"] else (0, 0, 0))
        sheet.save(review_dir / f"sheet_{s // per_sheet + 1:02d}.jpg", quality=92)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    refinements = json.loads(Path(args.refinements).read_text())
    images = raw_images(args.data_dir)
    records = {r["stem"]: r for r in map(json.loads, (out_dir / "metadata.jsonl").read_text().splitlines())}

    for stem, edits in refinements.items():
        img = Image.open(images[stem]).convert("RGB")
        label = np.array(Image.open(out_dir / "instances" / f"{stem}.png"))
        for e in edits:
            region = polygon_mask(label.shape, e["polygon"])
            if e["mode"] == "add_white":
                shell = eggshell_pixels(img, e.get("max_sat", MAX_SAT), e.get("min_val", MIN_VAL))
                label[region & shell & (label == 0)] = e["instance"]
            elif e["mode"] == "add":
                label[region & (label == 0)] = e["instance"]
            elif e["mode"] == "remove":
                label[region & (label == e["instance"])] = 0
            else:
                raise ValueError(f"unknown mode {e['mode']}")

        Image.fromarray(label).save(out_dir / "instances" / f"{stem}.png")
        Image.fromarray((label > 0).astype(np.uint8) * 255).save(out_dir / "masks" / f"{stem}.png")
        draw_overlay(img, label).save(out_dir / "overlays" / f"{stem}.jpg", quality=92)

        rec = records[stem]
        area = label.size
        for inst in rec["instances"]:
            inst["area_frac"] = round(float((label == inst["id"]).sum()) / area, 4)
        rec["refined"] = [{k: e[k] for k in ("instance", "mode", "max_sat", "min_val", "note") if k in e}
                          for e in edits]
        rec["flags"] = [f for f in rec["flags"] if f != "mask_box_mismatch"]
        print(f"{stem}: area_frac -> {[i['area_frac'] for i in rec['instances']]}")

    ordered = [records[k] for k in sorted(records)]
    (out_dir / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ordered))
    review_sheets(ordered, out_dir / "overlays", out_dir / "review")

    counts = {}
    for r in ordered:
        counts[r["n_eggs"]] = counts.get(r["n_eggs"], 0) + 1
    summary_path = out_dir / "summary.json"
    config = json.loads(summary_path.read_text()).get("config", {}) if summary_path.exists() else {}
    flagged = {r["stem"]: r["flags"] for r in ordered if r["flags"]}
    summary = {
        "n_images": len(ordered),
        "n_eggs_total": sum(r["n_eggs"] for r in ordered),
        "egg_count_histogram": {str(k): v for k, v in sorted(counts.items())},
        "prompt_sources": {s: sum(r["prompt_source"] == s for r in ordered)
                           for s in sorted({r["prompt_source"] for r in ordered})},
        "polygon_refined": sorted(refinements),
        "n_flagged": len(flagged),
        "flagged": flagged,
        "config": {**config, "out_dir": str(out_dir), "fixes": config.get("fixes"), "only": None},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
