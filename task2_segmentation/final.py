"""Frozen final algorithm (trial T24) and a runner that writes its predictions.

  python task2_segmentation/final.py --split dev           # predictions for the dev images
  python task2_segmentation/final.py --split eval          # only once the algorithm is frozen (Task 3)
  python task2_segmentation/final.py --image path/to.jpg   # single image -> overlay next to it

Writes predictions/<split>/instances/<stem>.png (uint8 label map) and overlays/<stem>.jpg.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from common import HERE, image_paths, load_split
from segment import segment
from trials import FINAL_ID, TRIALS

FINAL_CFG = {t["id"]: t for t in TRIALS}[FINAL_ID]["cfg"]
COLORS = [(0, 255, 0), (64, 64, 255), (255, 128, 64), (0, 200, 255), (255, 0, 255), (255, 255, 0)]


def overlay(bgr, inst):
    vis = bgr.copy()
    for k in range(1, int(inst.max()) + 1):
        m = inst == k
        c = np.array(COLORS[(k - 1) % len(COLORS)])
        vis[m] = (0.55 * vis[m] + 0.45 * c).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, tuple(int(v) for v in c), 2)
    return vis


def _run(args):
    stem, out = args
    bgr = cv2.imread(str(image_paths()[stem]))
    inst = segment(bgr, FINAL_CFG)
    cv2.imwrite(str(out / "instances" / f"{stem}.png"), inst)
    cv2.imwrite(str(out / "overlays" / f"{stem}.jpg"), overlay(bgr, inst))
    return stem, int(inst.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval", "all"])
    ap.add_argument("--image")
    args = ap.parse_args()
    if args.image:
        bgr = cv2.imread(args.image)
        inst = segment(bgr, FINAL_CFG)
        p = Path(args.image)
        cv2.imwrite(str(p.with_name(p.stem + "_seg.jpg")), overlay(bgr, inst))
        print(f"{inst.max()} egg(s) -> {p.with_name(p.stem + '_seg.jpg')}")
        return
    split = load_split()
    stems = split["dev"] + split["eval"] if args.split == "all" else split[args.split]
    out = HERE / "predictions" / args.split
    (out / "instances").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"trial": FINAL_ID, **FINAL_CFG}, indent=1))
    with Pool(6) as pool:
        res = pool.map(_run, [(s, out) for s in stems])
    print(f"{len(res)} images -> {out}  (eggs predicted: {sum(n for _, n in res)})")


if __name__ == "__main__":
    main()
