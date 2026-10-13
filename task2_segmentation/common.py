"""Shared helpers for Tasks 2 and 3: image/GT loading and segmentation metrics."""

import json
import re
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data" / "raw"
GT_DIR = ROOT / "task1_sam" / "output"
SPLIT_PATH = HERE / "split.json"


def stem_of(path):
    m = re.search(r"not_damaged_(\d+)_jpg", Path(path).name)
    return f"egg_{int(m.group(1)):03d}"


def image_paths(data_dir=DATA_DIR):
    return {stem_of(p): p for p in sorted(Path(data_dir).rglob("*.jpg"))}


def load_image(stem, paths=None):
    paths = paths or image_paths()
    return cv2.imread(str(paths[stem]))  # BGR uint8


def load_gt(stem):
    """(binary mask bool HxW, instance label map uint8 HxW)."""
    inst = cv2.imread(str(GT_DIR / "instances" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    return inst > 0, inst


def load_split():
    return json.loads(SPLIT_PATH.read_text())


# ---------------------------------------------------------------- metrics

def region_metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    union = tp + fp + fn
    return {
        "iou": tp / union if union else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if union else 1.0,
        "precision": tp / (tp + fp) if tp + fp else float(fn == 0),
        "recall": tp / (tp + fn) if tp + fn else 1.0,
    }


def boundary_f1(pred, gt, tol=3):
    """F1 of boundary pixels, matched within `tol` pixels."""
    def edge(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)

    pe, ge = edge(pred), edge(gt)
    if not pe.any() and not ge.any():
        return 1.0
    if not pe.any() or not ge.any():
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    p_hit = (pe & cv2.dilate(ge.astype(np.uint8), k).astype(bool)).sum() / pe.sum()
    r_hit = (ge & cv2.dilate(pe.astype(np.uint8), k).astype(bool)).sum() / ge.sum()
    return 2 * p_hit * r_hit / (p_hit + r_hit) if p_hit + r_hit else 0.0


def hd95(pred, gt):
    """95th percentile symmetric Hausdorff distance between mask boundaries, in pixels."""
    pred, gt = pred.astype(np.uint8), gt.astype(np.uint8)
    if not pred.any() or not gt.any():
        return float(np.hypot(*pred.shape))
    pe = pred - cv2.erode(pred, np.ones((3, 3), np.uint8))
    ge = gt - cv2.erode(gt, np.ones((3, 3), np.uint8))
    dt_g = cv2.distanceTransform(1 - ge, cv2.DIST_L2, 5)
    dt_p = cv2.distanceTransform(1 - pe, cv2.DIST_L2, 5)
    d = np.concatenate([dt_g[pe > 0], dt_p[ge > 0]])
    return float(np.percentile(d, 95))


def instance_metrics(pred_inst, gt_inst, thr=0.5):
    """Greedy one-to-one matching of instances by IoU; F1 at IoU >= thr and count error."""
    p_ids = [i for i in np.unique(pred_inst) if i]
    g_ids = [i for i in np.unique(gt_inst) if i]
    pairs = []
    for g in g_ids:
        gm = gt_inst == g
        for p in p_ids:
            pm = pred_inst == p
            inter = np.logical_and(gm, pm).sum()
            if inter:
                pairs.append((inter / np.logical_or(gm, pm).sum(), g, p))
    used_g, used_p, tp = set(), set(), 0
    for iou, g, p in sorted(pairs, reverse=True):
        if iou >= thr and g not in used_g and p not in used_p:
            used_g.add(g); used_p.add(p); tp += 1
    fp, fn = len(p_ids) - tp, len(g_ids) - tp
    return {
        "n_pred": len(p_ids), "n_gt": len(g_ids),
        "count_err": len(p_ids) - len(g_ids),
        "inst_f1": 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0,
    }
