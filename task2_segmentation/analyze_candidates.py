"""Diagnostic for the candidate stage (dev only): is the egg in the candidate pool, and where is it ranked?

For every dev image, all candidates of a trial config are matched to the GT instances (work
resolution). Reports the oracle (best possible) IoU per image and the rank of the best candidate,
and dumps candidate features to experiments/<trial>/candidates.csv for designing the ranking.

  python task2_segmentation/analyze_candidates.py T10
"""

import csv
import sys
from multiprocessing import Pool

import cv2
import numpy as np

from common import HERE, image_paths, load_gt, load_split
from segment import egg_score, egg_candidates
from trials import TRIALS


def _one(args):
    stem, cfg = args
    bgr_full = cv2.imread(str(image_paths()[stem]))
    H, W = bgr_full.shape[:2]
    s = cfg["work_size"] / max(H, W)
    size = (round(W * s), round(H * s))
    bgr = cv2.bilateralFilter(cv2.resize(bgr_full, size, interpolation=cv2.INTER_AREA), 7, 40, 7)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    score = egg_score(bgr, lab, cfg)
    cands = egg_candidates(score, cfg, lab)
    _, gt_inst = load_gt(stem)
    gt_inst = cv2.resize(gt_inst, size, interpolation=cv2.INTER_NEAREST)
    gts = [gt_inst == i for i in np.unique(gt_inst) if i]
    rows = []
    order = np.argsort([-q for q, _, _ in cands])
    for rank, k in enumerate(order):
        q, c, f = cands[k]
        best = max(np.logical_and(c, g).sum() / np.logical_or(c, g).sum() for g in gts)
        rows.append({"stem": stem, "rank": rank, "q": q, "gt_iou": best, **f})
    return stem, rows


def main():
    tid = sys.argv[1] if len(sys.argv) > 1 else "T10"
    cfg = {t["id"]: t for t in TRIALS}[tid]["cfg"]
    dev = load_split()["dev"]
    with Pool(6) as pool:
        results = pool.map(_one, [(s, cfg) for s in dev])
    all_rows = [r for _, rows in results for r in rows]
    out = HERE / "experiments" / tid / "candidates.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    oracle, top1, best_rank, n_c = [], [], [], []
    for stem, rows in results:
        n_c.append(len(rows))
        if not rows:
            oracle.append(0); top1.append(0); best_rank.append(-1)
            continue
        ious = [r["gt_iou"] for r in rows]
        oracle.append(max(ious)); top1.append(rows[0]["gt_iou"]); best_rank.append(int(np.argmax(ious)))
    oracle, top1, best_rank = map(np.array, (oracle, top1, best_rank))
    print(f"{tid}: candidates/img median {int(np.median(n_c))}, images with none: {(np.array(n_c) == 0).sum()}")
    print(f"oracle best-candidate IoU: mean {oracle.mean():.3f}, >=0.7 on {(oracle >= 0.7).mean():.0%}")
    print(f"top-ranked candidate IoU:  mean {top1.mean():.3f}, >=0.7 on {(top1 >= 0.7).mean():.0%}")
    print(f"best candidate is rank 0 on {(best_rank == 0).mean():.0%}, in top 3 on {((best_rank >= 0) & (best_rank < 3)).mean():.0%}")
    print("worst oracle:", sorted(zip(oracle.round(2), [s for s, _ in results]))[:12])


if __name__ == "__main__":
    main()
