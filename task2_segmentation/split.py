"""Leakage-aware 4:1 dev/eval split, at the level of whole scenes.

Leakage here is mostly scene leakage: the same background (cloth, pot, keyboard, marble counter...)
is photographed many times. If one scene lands on both sides, eval rewards memorising that
background instead of segmenting eggs. So every image gets a scene label and whole scenes are
assigned to dev or eval.

Scene labels (scenes.json) were produced in two steps:
  1. `python split.py --propose` clusters images automatically (HSV histogram of the border ring,
     HSV histogram of the whole image, 16x16 layout thumbnail; average-linkage cosine clustering,
     plus merging of near-identical consecutive filename ids) and writes contact sheets.
  2. Every image was reviewed: histogram clusters split one scene across groups (marble counter,
     pink bowl, hands on a dark wall) and merged unrelated grey scenes, so the labels were fixed by
     hand.

Assignment: among random scene subsets (fixed seed), keep those with 20-24 eval images and pick
the one that covers every difficulty type (hand/skin, white-on-white, textured background,
multi-egg), keeps the multi-egg share near the dataset share, and uses many scenes.
"""

import argparse
import json
from collections import Counter

import cv2
import numpy as np

from common import GT_DIR, HERE, SPLIT_PATH, image_paths

SEED = 0
TARGET_EVAL = 22
SCENE_TYPES = {
    "skin": {"hand_dark_wall", "hand_backlit", "hand_light_wall"},
    "white_on_white": {"white_box", "plastic_bag", "fridge_tray", "rice"},
    "texture": {"ash", "leaves", "paisley_cloth", "earphones_carpet", "woven_mat", "checked_cloth",
                "keyboard", "foil_packets"},
    "multi_egg": {"marble_counter", "steel_pot"},
}


# ---------------------------------------------------------------- proposal clustering

def features(img):
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    ring = np.ones((64, 64), np.uint8)
    ring[8:-8, 8:-8] = 0
    bins, ranges = [12, 4, 4], [0, 180, 0, 256, 0, 256]
    h_border = cv2.calcHist([hsv], [0, 1, 2], ring, bins, ranges).ravel()
    h_all = cv2.calcHist([hsv], [0, 1, 2], None, bins, ranges).ravel()
    layout = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA).ravel()
    layout = (layout - layout.mean()) / (layout.std() + 1e-6) / 16.0
    return np.concatenate([np.sqrt(h_border / h_border.sum()) * 1.5, np.sqrt(h_all / h_all.sum()), layout * 0.5])


def propose(cut=0.30, session_cut=0.20):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform

    paths = image_paths()
    stems = sorted(paths)
    ids = np.array([int(s[4:]) for s in stems])
    dist = pdist(np.stack([features(cv2.imread(str(paths[s]))) for s in stems]), "cosine")
    labels = fcluster(linkage(dist, "average"), t=cut, criterion="distance")
    D = squareform(dist)
    parent = {l: l for l in set(labels)}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    order = np.argsort(ids)
    for a, b in zip(order, order[1:]):
        if ids[b] - ids[a] == 1 and D[a, b] < session_cut:
            parent[find(labels[a])] = find(labels[b])
    groups = [find(l) for l in labels]
    clusters = {}
    for s, g in zip(stems, groups):
        clusters.setdefault(int(g), []).append(s)
    clusters = sorted(clusters.values(), key=lambda m: m[0])
    print(f"{len(clusters)} proposed groups, sizes {sorted(map(len, clusters), reverse=True)}")
    contact_sheets({f"cluster {k}": m for k, m in enumerate(clusters)}, set(), "proposal")


def contact_sheets(named_groups, eval_names, prefix, tile=128, cols=8, per_page=12):
    paths = image_paths()
    out = HERE / "split_groups"
    out.mkdir(exist_ok=True)
    for f in out.glob(f"{prefix}_*.jpg"):
        f.unlink()
    rows = []
    for name, members in named_groups.items():
        for start in range(0, len(members), cols):
            chunk = members[start:start + cols]
            strip = np.full((tile + 18, cols * tile, 3), 255, np.uint8)
            tag = "EVAL" if name in eval_names else "dev"
            cv2.putText(strip, f"{name} ({len(members)}) {tag}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 220) if name in eval_names else (0, 0, 0), 1)
            for k, s in enumerate(chunk):
                strip[18:, k * tile:(k + 1) * tile] = cv2.resize(cv2.imread(str(paths[s])), (tile, tile))
                cv2.putText(strip, s[4:], (k * tile + 3, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(strip, s[4:], (k * tile + 3, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            rows.append(strip)
    for k in range(0, len(rows), per_page):
        cv2.imwrite(str(out / f"{prefix}_{k // per_page + 1:02d}.jpg"), np.vstack(rows[k:k + per_page]))


# ---------------------------------------------------------------- final split

def make_split():
    scenes = json.loads((HERE / "scenes.json").read_text())
    scenes.pop("_note", None)
    scene_of = {f"egg_{i:03d}": name for name, ids in scenes.items() for i in ids}
    stems = sorted(image_paths())
    assert set(scene_of) == set(stems), "scenes.json must label every image exactly once"

    meta = {r["stem"]: r for r in map(json.loads, (GT_DIR / "metadata.jsonl").read_text().splitlines())}
    size = Counter(scene_of.values())
    multi = Counter(scene_of[s] for s in stems if meta[s]["n_eggs"] > 1)
    overall_multi = sum(multi.values()) / len(stems)
    names = sorted(scenes)

    rng = np.random.default_rng(SEED)
    best, best_score = None, -np.inf
    for _ in range(200000):
        chosen, total = [], 0
        for name in rng.permutation(names):
            if total + size[name] <= TARGET_EVAL + 2:
                chosen.append(name)
                total += size[name]
        if not TARGET_EVAL - 2 <= total <= TARGET_EVAL + 2:
            continue
        covered = sum(bool(SCENE_TYPES[t] & set(chosen)) for t in SCENE_TYPES)
        if covered < len(SCENE_TYPES):
            continue
        multi_share = sum(multi[n] for n in chosen) / total
        score = len(chosen) - 30 * abs(multi_share - overall_multi) - abs(total - TARGET_EVAL)
        if score > best_score:
            best, best_score = sorted(chosen), score

    eval_set = set(best)
    split = {
        "seed": SEED,
        "method": "whole-scene split; scene labels in scenes.json (auto clustering proposal + manual review)",
        "dev": [s for s in stems if scene_of[s] not in eval_set],
        "eval": [s for s in stems if scene_of[s] in eval_set],
        "eval_scenes": best,
        "dev_scenes": sorted(set(names) - eval_set),
        "scene_of": scene_of,
    }
    SPLIT_PATH.write_text(json.dumps(split, indent=1))

    def describe(part):
        ss = split[part]
        n_multi = sum(meta[s]["n_eggs"] > 1 for s in ss)
        n_eggs = sum(meta[s]["n_eggs"] for s in ss)
        types = [t for t in SCENE_TYPES if SCENE_TYPES[t] & {scene_of[s] for s in ss}]
        return f"{part}: {len(ss)} images, {n_eggs} eggs, {n_multi} multi-egg ({n_multi / len(ss):.0%}), types={types}"

    print(describe("dev"))
    print(describe("eval"))
    print("eval scenes:", best)
    contact_sheets({n: sorted(s for s in stems if scene_of[s] == n) for n in names}, eval_set, "split")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose", action="store_true", help="write automatic cluster proposal sheets")
    args = ap.parse_args()
    propose() if args.propose else make_split()
