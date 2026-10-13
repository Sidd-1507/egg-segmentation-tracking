"""Task 1: pseudo ground-truth egg masks with Grounding DINO (text -> boxes) + SAM (boxes -> masks).

Outputs (in --out_dir):
  masks/<stem>.png        binary union mask, 0/255
  instances/<stem>.png    uint8 label map, 0 = background, 1..N = egg id
  overlays/<stem>.jpg     image + coloured masks + boxes + scores
  review/sheet_XX.jpg     contact sheets for visual review (flagged images titled in red)
  metadata.jsonl          one record per image (boxes, scores, areas, flags, prompt source)
  summary.json            counts and the list of flagged images

Manual corrections: put prompts in a JSON file passed with --fixes, e.g.
  {"egg_036": {"boxes": [[100, 130, 480, 350]]},
   "egg_113": {"points": [[150, 450], [380, 315]]},
   "egg_117": {"instances": [{"box": [245, 105, 525, 285],
                              "points": [[300, 190], [480, 185], [437, 200]],
                              "labels": [1, 1, 0]}]}}
"boxes" / "points" give one egg per entry. "instances" combines a box with several clicks for one
egg (label 1 = egg, 0 = not egg), for eggs split by an occluder. Listed images skip the detector.
"""

import argparse
import inspect
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou, nms
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

COLORS = [(0, 255, 0), (255, 64, 64), (64, 128, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--out_dir", default="task1_sam/output")
    p.add_argument("--dino", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--sam", default="facebook/sam-vit-huge")
    p.add_argument("--prompt", default="an egg.")
    p.add_argument("--box_thresh", type=float, default=0.30)
    p.add_argument("--text_thresh", type=float, default=0.25)
    p.add_argument("--nms_iou", type=float, default=0.5)
    p.add_argument("--max_box_frac", type=float, default=0.70, help="drop boxes covering more of the image than this")
    p.add_argument("--min_mask_frac", type=float, default=0.002, help="drop masks smaller than this fraction of the image")
    p.add_argument("--fixes", default=None, help="JSON of manual box/point prompts per stem")
    p.add_argument("--only", default=None,
                   help="comma-separated stems, or 'fixes' for every stem in --fixes; "
                        "re-runs just those and merges them into the existing metadata")
    p.add_argument("--device", default=None)
    return p.parse_args()


def stem_of(path: Path) -> str:
    m = re.search(r"not_damaged_(\d+)_jpg", path.name)
    return f"egg_{int(m.group(1)):03d}" if m else path.stem


def collect_images(data_dir: Path):
    items = sorted(data_dir.rglob("*.jpg"), key=stem_of)
    stems = [stem_of(p) for p in items]
    assert len(stems) == len(set(stems)), "duplicate image ids"
    return items


# ---------------------------------------------------------------- detection

def detect(img, proc, model, args, device):
    inputs = proc(images=img, text=args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    kw = {"target_sizes": [img.size[::-1]], "text_threshold": args.text_thresh}
    params = inspect.signature(proc.post_process_grounded_object_detection).parameters
    kw["threshold" if "threshold" in params else "box_threshold"] = args.box_thresh
    res = proc.post_process_grounded_object_detection(out, inputs.input_ids, **kw)[0]
    return res["boxes"].float().cpu(), res["scores"].float().cpu()


def filter_boxes(boxes, scores, img_area, args):
    if len(boxes) == 0:
        return boxes, scores
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    keep = (wh[:, 0] * wh[:, 1]) / img_area <= args.max_box_frac
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return boxes, scores
    keep = nms(boxes, scores, args.nms_iou)
    boxes, scores = boxes[keep], scores[keep]

    # A box that encloses two or more other boxes is a "group of eggs" detection.
    drop = set()
    for i in range(len(boxes)):
        inside = 0
        for j in range(len(boxes)):
            if i != j and _containment(boxes[j], boxes[i]) > 0.85:
                inside += 1
        if inside >= 2:
            drop.add(i)
    keep = [i for i in range(len(boxes)) if i not in drop]
    return boxes[keep], scores[keep]


def _containment(inner, outer):
    x0, y0 = torch.max(inner[:2], outer[:2])
    x1, y1 = torch.min(inner[2:], outer[2:])
    inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return float(inter / area.clamp(min=1e-6))


# ---------------------------------------------------------------- segmentation

def sam_segment(img, proc, model, device, boxes=None, points=None):
    """Returns list of (bool HxW mask, sam iou score). One mask per box or per point."""
    if boxes is not None:
        inputs = proc(img, input_boxes=[boxes], return_tensors="pt").to(device)
        multimask = False
    else:
        inputs = proc(img, input_points=[[[p] for p in points]], return_tensors="pt").to(device)
        multimask = True  # single click is ambiguous; take SAM's best of 3
    with torch.no_grad():
        out = model(**inputs, multimask_output=multimask)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]  # (n_prompts, n_masks, H, W)
    scores = out.iou_scores.cpu()[0]  # (n_prompts, n_masks)
    best = scores.argmax(dim=1)
    return [(masks[i, best[i]].numpy().astype(bool), float(scores[i, best[i]])) for i in range(masks.shape[0])]


def sam_segment_instances(img, proc, model, device, instances):
    """One SAM call per egg, each with an optional box plus positive/negative clicks."""
    pixels = proc(img, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_image_embeddings(pixels["pixel_values"])
    results = []
    for inst in instances:
        kw = {}
        if "box" in inst:
            kw["input_boxes"] = [[inst["box"]]]
        if "points" in inst:
            kw["input_points"] = [[inst["points"]]]
            kw["input_labels"] = [[inst.get("labels", [1] * len(inst["points"]))]]
        inputs = proc(img, return_tensors="pt", **kw).to(device)
        prompts = {k: inputs[k] for k in ("input_boxes", "input_points", "input_labels") if k in inputs}
        with torch.no_grad():
            out = model(image_embeddings=embeddings, multimask_output=False, **prompts)
        masks = proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]
        results.append((masks[0, 0].numpy().astype(bool), float(out.iou_scores.cpu()[0, 0, 0])))
    return results


def clean_mask(mask, min_part_frac=0.1):
    """Drop specks and fill interior holes.

    Keeps every component at least min_part_frac of the largest one: an occluder (stem, tray rod)
    can cut one egg into two pieces, and both belong to the egg.
    """
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.flatnonzero(areas >= min_part_frac * areas.max())
        m = np.isin(lab, keep).astype(np.uint8)
    h, w = m.shape
    flood = m.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[seed[1], seed[0]] == 0:
            cv2.floodFill(flood, ff, seed, 1)
    return (m | (1 - flood)).astype(bool)


def build_instances(results, img_area, args):
    """Clean masks, drop tiny/duplicate ones, resolve overlaps (higher score wins)."""
    order = sorted(range(len(results)), key=lambda i: -results[i]["score"])
    kept, label = [], None
    for i in order:
        m = clean_mask(results[i]["mask"])
        if m.sum() / img_area < args.min_mask_frac:
            continue
        if label is None:
            label = np.zeros(m.shape, np.uint8)
        overlap = (label > 0) & m
        if overlap.sum() > 0.5 * m.sum():
            continue  # mostly duplicates an already kept egg
        m &= ~overlap
        kept.append({**results[i], "mask": m})
        label[m] = len(kept)
    return kept, label


# ---------------------------------------------------------------- outputs

def overlay(img_np, kept, boxes_used):
    vis = img_np.copy()
    for k, inst in enumerate(kept):
        c = COLORS[k % len(COLORS)]
        vis[inst["mask"]] = (0.55 * vis[inst["mask"]] + 0.45 * np.array(c)).astype(np.uint8)
        cnts, _ = cv2.findContours(inst["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, c, 2)
        ys, xs = np.nonzero(inst["mask"])
        cv2.putText(vis, f"{k + 1}:{inst['score']:.2f}", (int(xs.mean()) - 30, int(ys.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, f"{k + 1}:{inst['score']:.2f}", (int(xs.mean()) - 30, int(ys.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    for b in boxes_used:
        if b is None:
            continue
        x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 1)
    return vis


def review_sheets(records, overlay_dir, review_dir, per_sheet=20, cols=5, tile=256):
    review_dir.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(records), per_sheet):
        chunk = records[s:s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((rows * (tile + 24), cols * tile, 3), 255, np.uint8)
        for k, r in enumerate(chunk):
            im = cv2.imread(str(overlay_dir / f"{r['stem']}.jpg"))
            im = cv2.resize(im, (tile, tile))
            y, x = (k // cols) * (tile + 24), (k % cols) * tile
            sheet[y + 24:y + 24 + tile, x:x + tile] = im
            title = f"{r['stem']} n={r['n_eggs']} " + ",".join(r["flags"])
            color = (0, 0, 220) if r["flags"] else (0, 0, 0)
            cv2.putText(sheet, title[:40], (x + 4, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        cv2.imwrite(str(review_dir / f"sheet_{s // per_sheet + 1:02d}.jpg"), sheet)


def flags_for(rec, kept, img_area):
    f = []
    if rec["prompt_source"] == "center_point_fallback":
        f.append("no_detection")
    if rec["n_eggs"] == 0:
        f.append("empty")
    if rec["n_eggs"] > 4:
        f.append("many_eggs")
    for inst in kept:
        if inst["score"] < 0.85:
            f.append("low_sam_score")
        if inst["mask"].sum() / img_area > 0.6:
            f.append("huge_mask")
        if inst.get("box") is not None:
            x0, y0, x1, y1 = inst["box"]
            fill = inst["mask"].sum() / max((x1 - x0) * (y1 - y0), 1)
            if fill < 0.5 or fill > 1.0:  # an ellipse fills ~0.785 of its box
                f.append("mask_box_mismatch")
    return sorted(set(f))


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    for sub in ["masks", "instances", "overlays"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    fixes = json.loads(Path(args.fixes).read_text()) if args.fixes else {}
    print(f"loaded {len(fixes)} manual fixes: {sorted(fixes)}")

    paths = collect_images(data_dir)
    previous = {}
    if args.only:
        only = set(fixes) if args.only == "fixes" else set(args.only.split(","))
        paths = [p for p in paths if stem_of(p) in only]
        meta = out_dir / "metadata.jsonl"
        if meta.exists():
            previous = {r["stem"]: r for r in map(json.loads, meta.read_text().splitlines())}
        print(f"re-running {len(paths)} images: {sorted(stem_of(p) for p in paths)}")
    need_dino = any(stem_of(p) not in fixes for p in paths)

    print(f"device={device}  dino={args.dino if need_dino else 'not needed'}  sam={args.sam}")
    if need_dino:
        dino_proc = AutoProcessor.from_pretrained(args.dino)
        dino = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino).to(device).eval()
    sam_proc = SamProcessor.from_pretrained(args.sam)
    sam = SamModel.from_pretrained(args.sam).to(device).eval()

    records = []
    for path in tqdm(paths):
        stem = stem_of(path)
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        img_area = h * w
        det_boxes, det_scores = torch.zeros((0, 4)), torch.zeros(0)

        manual_instances = None
        if stem in fixes and "instances" in fixes[stem]:
            manual_instances = fixes[stem]["instances"]
            source, boxes, points = "manual_instances", [i.get("box") for i in manual_instances], None
        elif stem in fixes and "boxes" in fixes[stem]:
            source, boxes, points = "manual_boxes", fixes[stem]["boxes"], None
        elif stem in fixes and "points" in fixes[stem]:
            source, boxes, points = "manual_points", None, fixes[stem]["points"]
        else:
            det_boxes, det_scores = detect(img, dino_proc, dino, args, device)
            det_boxes, det_scores = filter_boxes(det_boxes, det_scores, img_area, args)
            if len(det_boxes):
                source, boxes, points = "grounding_dino", det_boxes.tolist(), None
            else:
                source, boxes, points = "center_point_fallback", None, [[w / 2, h / 2]]

        if manual_instances is not None:
            segs = sam_segment_instances(img, sam_proc, sam, device, manual_instances)
        else:
            segs = sam_segment(img, sam_proc, sam, device, boxes=boxes, points=points)
        results = [{"mask": m, "score": s, "box": boxes[i] if boxes else None,
                    "det_score": float(det_scores[i]) if source == "grounding_dino" else None}
                   for i, (m, s) in enumerate(segs)]
        kept, label = build_instances(results, img_area, args)
        if label is None:
            label = np.zeros((h, w), np.uint8)

        cv2.imwrite(str(out_dir / "masks" / f"{stem}.png"), (label > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(out_dir / "instances" / f"{stem}.png"), label)
        vis = overlay(img_np, kept, boxes or [])
        cv2.imwrite(str(out_dir / "overlays" / f"{stem}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        rec = {
            "stem": stem,
            "file": str(path.relative_to(data_dir)),
            "roboflow_split": path.relative_to(data_dir).parts[0],
            "prompt_source": source,
            "sam_model": args.sam,
            "n_eggs": len(kept),
            "instances": [{
                "id": k + 1,
                "box": [round(v, 1) for v in inst["box"]] if inst["box"] is not None else None,
                "det_score": round(inst["det_score"], 3) if inst["det_score"] is not None else None,
                "sam_score": round(inst["score"], 3),
                "area_frac": round(float(inst["mask"].sum()) / img_area, 4),
            } for k, inst in enumerate(kept)],
        }
        rec["flags"] = flags_for(rec, kept, img_area)
        records.append(rec)

    if previous:
        previous.update({r["stem"]: r for r in records})
        records = sorted(previous.values(), key=lambda r: r["stem"])

    with open(out_dir / "metadata.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    review_sheets(records, out_dir / "overlays", out_dir / "review")

    flagged = {r["stem"]: r["flags"] for r in records if r["flags"]}
    counts = {}
    for r in records:
        counts[r["n_eggs"]] = counts.get(r["n_eggs"], 0) + 1
    summary = {
        "n_images": len(records),
        "egg_count_histogram": dict(sorted(counts.items())),
        "prompt_sources": {s: sum(r["prompt_source"] == s for r in records) for s in {r["prompt_source"] for r in records}},
        "n_flagged": len(flagged),
        "flagged": flagged,
        "config": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
