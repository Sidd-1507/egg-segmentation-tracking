"""Task 3: score the frozen Task 2 algorithm (T24) against the Task 1 masks.

The eval split is segmented here for the first time; nothing in the algorithm changes afterwards.
Dev predictions were produced by task2_segmentation/final.py during development and are reused.

  python task3_evaluation/evaluate.py

Writes results/{per_image.csv,summary.json} and figures/* next to this file.
"""

import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

TASK2 = Path(__file__).resolve().parents[1] / "task2_segmentation"
sys.path.insert(0, str(TASK2))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (boundary_f1, hd95, image_paths, instance_metrics, load_gt, load_split,
                    region_metrics)
from run_trials import error_overlay
from final import FINAL_CFG, FINAL_ID, overlay
from segment import segment

OUT = Path(__file__).resolve().parent
PRED = TASK2 / "predictions"
RES, FIG = OUT / "results", OUT / "figures"
METRICS = ["iou", "dice", "precision", "recall", "bf1", "hd95", "inst_f1", "abs_count_err"]


def factors(bgr, gt):
    """Properties of the image and the ground-truth egg, for the 'what went wrong' plots."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    h, w = L.shape
    m = max(4, int(0.08 * min(h, w)))
    border = np.ones((h, w), bool)
    border[m:-m, m:-m] = False
    egg = gt.astype(bool)
    def dist(mask):
        return np.hypot(L[mask] - L[border].mean(),
                        np.hypot(a[mask] - a[border].mean(), b[mask] - b[border].mean()))
    return {
        "gt_area_frac": float(egg.mean()),
        "brightness": float(L.mean()),
        "egg_brightness": float(L[egg].mean()) if egg.any() else 0.0,
        "contrast": float(dist(egg).mean()) if egg.any() else 0.0,  # Lab distance, egg vs border
    }


def failure_class(r):
    """Coarse, metric-based label. The report refines the worst cases by looking at them."""
    if r["n_pred"] == 0 or (r["recall"] < 0.2 and r["precision"] < 0.2):
        return "missed"
    if r["count_err"] < 0 and r["iou"] >= 0.45:
        return "eggs merged"
    if r["count_err"] > 0 and r["precision"] < 0.75:
        return "extra false region"
    if r["precision"] < 0.55 and r["recall"] >= 0.35:
        return "leaked into background"
    if r["recall"] < 0.7 and r["precision"] >= 0.7:
        return "partial (shadow/occlusion)"
    if r["iou"] >= 0.7:
        return "good"
    return "moderate error"


def _one(args):
    stem, split, scene, save = args
    bgr = cv2.imread(str(image_paths()[stem]))
    gt, gt_inst = load_gt(stem)
    if split == "eval":
        pred_inst = segment(bgr, FINAL_CFG)
    else:
        pred_inst = cv2.imread(str(PRED / "dev" / "instances" / f"{stem}.png"),
                               cv2.IMREAD_UNCHANGED)
    pred = pred_inst > 0
    row = region_metrics(pred, gt)
    row["bf1"] = boundary_f1(pred, gt)
    row["hd95"] = hd95(pred, gt)
    row.update(instance_metrics(pred_inst, gt_inst))
    row["abs_count_err"] = abs(row["count_err"])
    row.update(factors(bgr, gt))
    row.update(stem=stem, split=split, scene=scene, failure=failure_class(row))
    if save:
        d = PRED / "eval"
        cv2.imwrite(str(d / "instances" / f"{stem}.png"), pred_inst)
        cv2.imwrite(str(d / "overlays" / f"{stem}.jpg"), overlay(bgr, pred_inst))
        err = error_overlay(bgr, pred, gt)
        cv2.imwrite(str(FIG / "errors" / f"{stem}.jpg"), np.hstack([bgr, err]))
    return row


def summarise(rows):
    out = {"n": len(rows), "n_eggs_gt": int(sum(r["n_gt"] for r in rows)),
           "n_eggs_pred": int(sum(r["n_pred"] for r in rows))}
    for k in METRICS:
        v = np.array([r[k] for r in rows], float)
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std())}
    iou = np.array([r["iou"] for r in rows])
    out["frac_iou_ge_0.5"] = float((iou >= 0.5).mean())
    out["frac_iou_ge_0.7"] = float((iou >= 0.7).mean())
    out["failures"] = dict(sorted((k, sum(r["failure"] == k for r in rows))
                                  for k in {r["failure"] for r in rows}))
    return out


def save_csv(rows):
    fields = ["stem", "split", "scene", "failure", "iou", "dice", "precision", "recall", "bf1", "hd95",
              "inst_f1", "n_pred", "n_gt", "count_err", "gt_area_frac", "brightness", "egg_brightness", "contrast"]
    with open(RES / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields})


# ---------------------------------------------------------------- figures

def _hist(ax, rows, split, color):
    v = [r["iou"] for r in rows if r["split"] == split]
    ax.hist(v, bins=np.linspace(0, 1, 11), color=color, alpha=0.55, edgecolor="white", label=f"{split} (n={len(v)})")


def fig_histogram(rows):
    fig, ax = plt.subplots(figsize=(7, 4))
    _hist(ax, rows, "dev", "#4C78A8")
    _hist(ax, rows, "eval", "#E45756")
    ax.set_xlabel("IoU vs Task 1 masks")
    ax.set_ylabel("images")
    ax.set_title("IoU distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "iou_histogram.png", dpi=130)
    plt.close(fig)


def fig_factors(rows):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    specs = [("contrast", "egg vs border contrast (Lab)"), ("gt_area_frac", "egg area / image"),
             ("brightness", "image brightness (L)"), ("n_gt", "number of eggs (ground truth)")]
    for ax, (key, label) in zip(axes.ravel(), specs):
        for split, color, marker in (("dev", "#4C78A8", "o"), ("eval", "#E45756", "s")):
            sub = [r for r in rows if r["split"] == split]
            ax.scatter([r[key] for r in sub], [r["iou"] for r in sub], s=18, alpha=0.7,
                       c=color, marker=marker, label=split)
        ax.set_xlabel(label)
        ax.set_ylabel("IoU")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("Where the algorithm loses IoU")
    fig.tight_layout()
    fig.savefig(FIG / "iou_vs_factors.png", dpi=130)
    plt.close(fig)


def fig_scenes(rows):
    scenes = sorted({r["scene"] for r in rows})
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(scenes))
    for split, color, shift in (("dev", "#4C78A8", -0.18), ("eval", "#E45756", 0.18)):
        means, ns = [], []
        for s in scenes:
            v = [r["iou"] for r in rows if r["scene"] == s and r["split"] == split]
            means.append(np.mean(v) if v else np.nan)
            ns.append(len(v))
        ax.barh(y + shift, [0 if np.isnan(m) else m for m in means], height=0.36, color=color, label=split)
        for yi, (m, n) in enumerate(zip(means, ns)):
            if n:
                ax.text((0 if np.isnan(m) else m) + 0.01, yi + shift, f"n={n}", va="center", fontsize=7, color=color)
    ax.set_yticks(y, scenes, fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("mean IoU")
    ax.set_title("Mean IoU by scene")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "iou_by_scene.png", dpi=130)
    plt.close(fig)


def sheet(rows, name, k=10):
    """Original | error map (green TP, red FP, blue FN) for the worst images of one split."""
    tile = 220
    tiles = []
    for r in rows[:k]:
        bgr = cv2.imread(str(image_paths()[r["stem"]]))
        pred = cv2.imread(str(PRED / r["split"] / "instances" / f"{r['stem']}.png"),
                          cv2.IMREAD_UNCHANGED) > 0
        gt, _ = load_gt(r["stem"])
        pair = cv2.resize(np.hstack([bgr, error_overlay(bgr, pred, gt)]), (2 * tile, tile))
        label = f"{r['stem']} {r['scene']} IoU {r['iou']:.2f} {r['failure']}"
        cv2.putText(pair, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3)
        cv2.putText(pair, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(pair)
    cv2.imwrite(str(FIG / name), np.vstack(tiles))


def main():
    for d in (RES, FIG / "errors"):
        d.mkdir(parents=True, exist_ok=True)
    split = load_split()
    scene = split["scene_of"]
    eval_dir = PRED / "eval"
    (eval_dir / "instances").mkdir(parents=True, exist_ok=True)
    (eval_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (eval_dir / "config.json").write_text(json.dumps({"trial": FINAL_ID, "note": "first and only eval run"}, indent=1))

    jobs = [(s, "dev", scene[s], False) for s in split["dev"]] + [(s, "eval", scene[s], True) for s in split["eval"]]
    with Pool(6) as pool:
        rows = pool.map(_one, jobs)

    save_csv(rows)
    summary = {"algorithm": FINAL_ID, "ground_truth": "task1 SAM masks",
               "dev": summarise([r for r in rows if r["split"] == "dev"]),
               "eval": summarise([r for r in rows if r["split"] == "eval"])}
    # per scene, both splits
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["scene"]][r["split"]].append(r["iou"])
    summary["by_scene"] = {s: {sp: {"n": len(v), "iou_mean": float(np.mean(v))} for sp, v in d.items()}
                           for s, d in sorted(by.items())}
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))

    fig_histogram(rows)
    fig_factors(rows)
    fig_scenes(rows)
    for sp in ("eval", "dev"):
        ordered = sorted((r for r in rows if r["split"] == sp), key=lambda r: r["iou"])
        sheet(ordered, f"worst_{sp}.jpg")
        sheet(list(reversed(ordered)), f"best_{sp}.jpg", k=6)
    print(json.dumps({"dev": summary["dev"], "eval": summary["eval"]}, indent=1))


if __name__ == "__main__":
    main()
