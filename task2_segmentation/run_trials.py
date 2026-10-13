"""Runs segmentation trials on the DEV split only and logs dev metrics against the Task 1 masks.

  python task2_segmentation/run_trials.py            # every trial in trials.py not yet in the log
  python task2_segmentation/run_trials.py T05 T06    # specific trials (re-run and overwrite)

Writes:
  experiments/log.csv             one row per trial (dev mean/median IoU, Dice, P, R, BF1, instance F1)
  experiments/<id>/per_image.csv  per-image dev metrics
  experiments/<id>/worst.jpg      the 8 worst dev images (green = TP, red = FP, blue = FN)
"""

import csv
import json
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

from common import HERE, boundary_f1, image_paths, instance_metrics, load_gt, load_split, region_metrics
from segment import segment
from trials import TRIALS

EXP = HERE / "experiments"
LOG = EXP / "log.csv"
FIELDS = ["id", "name", "why", "changes", "n", "iou_mean", "iou_median", "dice_mean", "precision", "recall",
          "bf1", "inst_f1", "abs_count_err", "frac_iou_lt_0.5", "sec_per_img", "decision", "decision_reason"]


def error_overlay(bgr, pred, gt):
    vis = (bgr * 0.45).astype(np.uint8)
    vis[pred & gt] = (0, 200, 0)
    vis[pred & ~gt] = (0, 0, 255)
    vis[~pred & gt] = (255, 80, 0)
    return vis


def _eval_one(args):
    stem, cfg = args
    paths = image_paths()
    bgr = cv2.imread(str(paths[stem]))
    t = time.time()
    pred_inst = segment(bgr, cfg)
    dt = time.time() - t
    gt, gt_inst = load_gt(stem)
    pred = pred_inst > 0
    r = region_metrics(pred, gt)
    r.update(instance_metrics(pred_inst, gt_inst))
    r["bf1"] = boundary_f1(pred, gt)
    r["stem"], r["sec"] = stem, dt
    return r


def evaluate(cfg, stems, processes=6):
    with Pool(processes) as pool:
        return pool.map(_eval_one, [(s, cfg) for s in stems])


def summarise(rows):
    a = lambda k: np.array([r[k] for r in rows], float)
    iou = a("iou")
    return {
        "n": len(rows),
        "iou_mean": iou.mean(), "iou_median": np.median(iou), "dice_mean": a("dice").mean(),
        "precision": a("precision").mean(), "recall": a("recall").mean(), "bf1": a("bf1").mean(),
        "inst_f1": a("inst_f1").mean(), "abs_count_err": np.abs(a("count_err")).mean(),
        "frac_iou_lt_0.5": (iou < 0.5).mean(), "sec_per_img": a("sec").mean(),
    }


def worst_sheet(rows, cfg, path, k=8, tile=200):
    paths = image_paths()
    worst = sorted(rows, key=lambda r: r["iou"])[:k]
    tiles = []
    for r in worst:
        bgr = cv2.imread(str(paths[r["stem"]]))
        pred = segment(bgr, cfg) > 0
        gt, _ = load_gt(r["stem"])
        t = cv2.resize(np.hstack([bgr, error_overlay(bgr, pred, gt)]), (2 * tile, tile))
        cv2.putText(t, f"{r['stem']} IoU {r['iou']:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(t, f"{r['stem']} IoU {r['iou']:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(t)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def read_log():
    if not LOG.exists():
        return {}
    with open(LOG) as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def write_log(rows):
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for tid in sorted(rows):
            w.writerow({k: rows[tid].get(k, "") for k in FIELDS})


def sync_decisions(log, by_id):
    for tid, row in log.items():
        if tid in by_id:
            row["decision"] = by_id[tid]["decision"]
            row["decision_reason"] = by_id[tid]["reason"]


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    dev = load_split()["dev"]
    log = read_log()
    by_id = {t["id"]: t for t in TRIALS}
    wanted = sys.argv[1:] or [t["id"] for t in TRIALS if t["id"] not in log]
    if not wanted:  # nothing to run: just refresh decisions from trials.py
        sync_decisions(log, by_id)
        write_log(log)
    for tid in wanted:
        t = by_id[tid]
        rows = evaluate(t["cfg"], dev)
        s = summarise(rows)
        out = EXP / tid
        out.mkdir(exist_ok=True)
        with open(out / "per_image.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (out / "config.json").write_text(json.dumps(t["cfg"], indent=1))
        worst_sheet(rows, t["cfg"], out / "worst.jpg")
        log[tid] = {"id": tid, "name": t["name"], "why": t["why"], "changes": t["changes"],
                    **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in s.items()}}
        sync_decisions(log, by_id)
        write_log(log)
        print(f"{tid} {t['name']:<38s} IoU {s['iou_mean']:.3f} (med {s['iou_median']:.3f})  Dice {s['dice_mean']:.3f}  "
              f"P {s['precision']:.3f} R {s['recall']:.3f}  BF1 {s['bf1']:.3f}  instF1 {s['inst_f1']:.3f}  "
              f"|cnt| {s['abs_count_err']:.2f}  <0.5: {s['frac_iou_lt_0.5']:.0%}  {s['sec_per_img']:.2f}s")


if __name__ == "__main__":
    main()
