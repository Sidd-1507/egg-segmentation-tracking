"""Classical egg segmentation. Every stage is switchable through a config dict so each experiment
in trials.py is the same code with different settings.

Pipeline (full version):
  1. resize to work_size, bilateral denoise, convert to Lab
  2. per-pixel egg score from up to three cues:
       white   - bright and low chroma (eggshell is near-white)
       bg      - colour distance from a k-means model of the border pixels (assumed background)
       smooth  - low local texture (eggshell is smooth; ash, leaves, cloth are not)
  3. threshold the score (Otsu or fixed), optional skin removal, morphology
  4. pick egg-like connected components (solidity, ellipse aspect, area)
  5. optional GrabCut refinement seeded from those components
  6. hole filling, optional ellipse regularisation
  7. optional watershed split of touching eggs
  8. upsample instances to the original size
"""

import cv2
import numpy as np

DEFAULT = dict(
    method="score",            # "otsu_gray" | "hsv_white" | "score"
    work_size=320,
    # cues used by method="score"
    use_white=True, use_bg=False, use_smooth=False,
    chroma_scale=40.0,         # chroma at which the white cue reaches 0
    bg_border=0.06, bg_k=4, bg_l_weight=0.5,
    smooth_win=7, smooth_scale=8.0,
    # hsv_white
    hsv_s_max=60, hsv_v_pct=70,
    # thresholding and cleanup
    thresh="otsu",             # "otsu" or a float in [0, 1]
    skin=False,
    open_r=0, close_r=0,
    # component selection
    select="all",              # "all" | "largest" | "egg" | "multilevel"
    min_area=0.004, max_area=0.65,
    min_solidity=0.85, max_aspect=2.2, rel_area=0.2,
    # multilevel: sweep thresholds, collect egg-shaped blobs at any level, keep the best ones
    ml_levels=16, ml_lo=0.2, ml_hi=0.995, ml_rel_quality=0.6, ml_min_fill=0.75, ml_nms_iou=0.3, ml_ring=0.15,
    # candidate maps and ranking for multilevel
    ml_sources=("score",),     # any of "score", "white", "bg", "L", "edges", "L_ms", "white_ms", "bg_ms"
    ms_sp=12, ms_sr=24,        # mean-shift filter (spatial / colour radius) for the *_ms sources
    ml_open_radii=(1,),        # opening radii per level; larger radii cut thin bridges between egg and background
    ml_min_white=0.0,          # reject candidates whose mean white cue is below this (not eggshell-coloured)
    ml_quality="contrast",     # "contrast" | "edge" | "both" | "ring" | "ring2"
    ring_area_pow=0.25,        # "ring": quality *= area_fraction ** ring_area_pow (prefer whole eggs to fragments)
    edge_pct=80,               # gradient percentile that counts as a strong edge
    area_prior=False,          # down-weight blobs outside the typical egg size (1-35% of the image)
    # refinement
    grabcut=False, gc_iters=5, gc_fg_erode=0.25, gc_bg_margin=0.15,
    gc_guard=False, gc_max_growth=1.6,  # per egg: undo GrabCut if it grows > x or stops being egg-shaped
    fill_holes=False,
    ellipse=False, ellipse_min_fill=0.85,
    watershed=False, ws_peak=0.6,
)


def cfg_with(**kw):
    unknown = set(kw) - set(DEFAULT)
    if unknown:
        raise KeyError(f"unknown config keys {unknown}")
    return {**DEFAULT, **kw}


# ---------------------------------------------------------------- cues

def _norm(x, lo_pct=2, hi_pct=99.5):
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def white_cue(lab, cfg):
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    return _norm(L) * np.clip(1 - chroma / cfg["chroma_scale"], 0, 1)


def bg_cue(lab, cfg):
    h, w = lab.shape[:2]
    b = max(2, int(cfg["bg_border"] * min(h, w)))
    ring = np.ones((h, w), bool)
    ring[b:-b, b:-b] = False
    wts = np.array([cfg["bg_l_weight"], 1.0, 1.0], np.float32)
    samples = (lab[ring] * wts).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)
    _, _, centers = cv2.kmeans(samples, cfg["bg_k"], None, crit, 3, cv2.KMEANS_PP_CENTERS)
    px = (lab.reshape(-1, 3) * wts)[:, None, :]
    d = np.sqrt(((px - centers[None]) ** 2).sum(-1)).min(1).reshape(h, w)
    return _norm(d, 0, 99)


def smooth_cue(lab, cfg):
    L = lab[..., 0]
    k = (cfg["smooth_win"], cfg["smooth_win"])
    mean = cv2.blur(L, k)
    std = np.sqrt(np.maximum(cv2.blur(L * L, k) - mean * mean, 0))
    return np.exp(-std / cfg["smooth_scale"])


def skin_mask(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    return (cr >= 138) & (cr <= 180) & (cb >= 80) & (cb <= 130)


def egg_score(bgr, lab, cfg):
    cues = []
    if cfg["use_white"]:
        cues.append(white_cue(lab, cfg))
    if cfg["use_bg"]:
        cues.append(bg_cue(lab, cfg))
    if cfg["use_smooth"]:
        cues.append(smooth_cue(lab, cfg))
    score = np.ones(lab.shape[:2], np.float32)
    for c in cues:
        score *= c
    return score ** (1.0 / max(len(cues), 1))  # geometric mean keeps the 0-1 scale


# ---------------------------------------------------------------- components

def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def fill_holes(m):
    m = m.astype(np.uint8)
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return (m | (1 - ff[1:-1, 1:-1])).astype(bool)


def shape_stats(comp):
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = comp.sum()
    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = area / hull if hull > 0 else 0
    if len(c) >= 5:
        (cx, cy), (a, b), ang = cv2.fitEllipse(c)
        aspect = max(a, b) / max(min(a, b), 1e-6)
        fill = area / (np.pi * a * b / 4 + 1e-6)
        ellipse = ((cx, cy), (a, b), ang)
    else:
        aspect, fill, ellipse = 99.0, 0.0, None
    return dict(area=area, solidity=solidity, aspect=aspect, fill=fill, ellipse=ellipse)


def components(mask):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [lab == i for i in range(1, n)]


def select_components(mask, score, cfg):
    comps = components(mask)
    total = mask.size
    comps = [c for c in comps if cfg["min_area"] * total <= c.sum() <= cfg["max_area"] * total]
    if not comps:
        return np.zeros_like(mask)
    if cfg["select"] == "all":
        return np.any(comps, axis=0)
    if cfg["select"] == "largest":
        return max(comps, key=lambda c: c.sum())
    # "egg": keep convex, ellipse-like blobs, then drop ones much smaller than the best
    stats = [(c, shape_stats(c)) for c in comps]
    eggs = [(c, s) for c, s in stats if s["solidity"] >= cfg["min_solidity"] and s["aspect"] <= cfg["max_aspect"]]
    if not eggs:  # nothing egg-shaped: fall back to the most egg-like blob
        c, _ = max(stats, key=lambda cs: cs[1]["solidity"] * cs[0].sum() * score[cs[0]].mean())
        return c
    biggest = max(s["area"] for _, s in eggs)
    return np.any([c for c, s in eggs if s["area"] >= cfg["rel_area"] * biggest], axis=0)


def _level_masks(cue, cfg):
    lo, hi = np.percentile(cue, [cfg["ml_lo"] * 100, cfg["ml_hi"] * 100])
    for t in np.linspace(lo, hi, cfg["ml_levels"]):
        m = (cue >= t).astype(np.uint8)
        for r in cfg["ml_open_radii"]:
            yield cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(r))


def _edge_regions(L):
    """Regions enclosed by Canny edges (auto thresholds from the median)."""
    u8 = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    med = np.median(u8)
    edges = cv2.Canny(u8, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    edges = cv2.dilate(edges, disk(1))
    yield (edges == 0).astype(np.uint8)


def multilevel_select(score, cfg, lab=None):
    """Threshold cue maps at many levels (like MSER). At some level the egg usually separates from
    the background as a clean ellipse even when a single global threshold merges them. Each candidate
    blob is scored by ellipse fill * solidity * evidence, where evidence is the cue contrast
    (inside vs a ring outside), the fraction of its outline on strong image edges, or both;
    candidates are kept greedily by quality with overlap suppression."""
    cands = egg_candidates(score, cfg, lab)
    if not cands:
        return np.zeros(score.shape, bool)
    cands.sort(key=lambda qc: -qc[0])
    best_q = cands[0][0]
    kept = []
    for q, c, _ in cands:
        if q < cfg["ml_rel_quality"] * best_q:
            break
        if all(np.logical_and(c, k).sum() / np.logical_or(c, k).sum() < cfg["ml_nms_iou"]
               and np.logical_and(c, k).sum() < 0.5 * c.sum() for k in kept):
            kept.append(c)
    return np.any(kept, axis=0)


def egg_candidates(score, cfg, lab=None):
    """All egg-shaped blobs over every cue map and threshold level, as (quality, mask) pairs."""
    h, w = score.shape
    total = h * w
    maps = {"score": score}
    if lab is not None:
        maps["white"] = white_cue(lab, cfg)
        maps["bg"] = bg_cue(lab, cfg)
        maps["L"] = _norm(lab[..., 0])
        gx = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)
        strong = cv2.dilate((grad >= np.percentile(grad, cfg["edge_pct"])).astype(np.uint8), disk(1)).astype(bool)
        if any(s.endswith("_ms") for s in cfg["ml_sources"]):
            bgr_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            ms = cv2.pyrMeanShiftFiltering(bgr_u8, cfg["ms_sp"], cfg["ms_sr"])
            lab_ms = cv2.cvtColor(ms, cv2.COLOR_BGR2LAB).astype(np.float32)
            maps["L_ms"] = _norm(lab_ms[..., 0])
            maps["white_ms"] = white_cue(lab_ms, cfg)
            maps["bg_ms"] = bg_cue(lab_ms, cfg)

    cands = []
    for src in cfg["ml_sources"]:
        levels = _edge_regions(lab[..., 0]) if src == "edges" else _level_masks(maps[src], cfg)
        cue = score if src == "edges" else maps[src]
        for m in levels:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if not cfg["min_area"] * total <= area <= cfg["max_area"] * total:
                    continue
                comp = labels == i
                s = shape_stats(comp)
                if s["aspect"] > cfg["max_aspect"] or s["fill"] < cfg["ml_min_fill"] or s["solidity"] < cfg["min_solidity"]:
                    continue
                r = max(2, int(cfg["ml_ring"] * np.sqrt(area)))
                ring = cv2.dilate(comp.astype(np.uint8), disk(r)).astype(bool) & ~comp
                contrast = max(cue[comp].mean() - cue[ring].mean(), 0) * 2
                edge = 0.0
                if lab is not None:
                    outline = comp & ~cv2.erode(comp.astype(np.uint8), disk(1)).astype(bool)
                    edge = float(strong[outline].mean())
                frac = area / total
                if lab is not None and maps["white"][comp].mean() < cfg["ml_min_white"]:
                    continue
                shape = min(s["fill"], 1.0) * s["solidity"]
                (cx, cy) = s["ellipse"][0]
                center_dist = float(np.hypot(cx / w - 0.5, cy / h - 0.5))
                if cfg["ml_quality"] in ("ring", "ring2"):
                    # whiter than its surroundings + more unlike the border than its surroundings
                    w_d = maps["white"][comp].mean() - maps["white"][ring].mean()
                    b_d = maps["bg"][comp].mean() - maps["bg"][ring].mean()
                    ev = (max(w_d, 0) + max(b_d, 0)) * frac ** cfg["ring_area_pow"]
                    if cfg["ml_quality"] == "ring2":
                        # plus: outline on strong edges, closer to the centre, shape counted twice
                        ev *= (0.5 + edge) * (1 - center_dist) * shape
                else:
                    evidence = {"contrast": [contrast], "edge": [edge], "both": [contrast, edge]}[cfg["ml_quality"]]
                    ev = float(np.mean(evidence))
                if ev <= 0:
                    continue
                q = shape * ev
                if cfg["area_prior"]:
                    q *= 1.0 if 0.01 <= frac <= 0.35 else 0.6
                feats = dict(src=src, fill=s["fill"], solidity=s["solidity"], aspect=s["aspect"], area=frac,
                             contrast=contrast, edge=edge, center_dist=center_dist,
                             touches_border=bool(comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()))
                if lab is not None:
                    feats.update(white=float(maps["white"][comp].mean()), bg=float(maps["bg"][comp].mean()),
                                 white_ring=float(maps["white"][ring].mean()), bg_ring=float(maps["bg"][ring].mean()))
                cands.append((q, comp, feats))
    return cands


# ---------------------------------------------------------------- refinement

def grabcut_refine(bgr, mask, cfg):
    if not mask.any():
        return mask
    h, w = mask.shape
    size = np.sqrt(mask.sum())
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    far = ~cv2.dilate(mask.astype(np.uint8), disk(max(2, int(cfg["gc_bg_margin"] * size)))).astype(bool)
    gc[far] = cv2.GC_BGD
    gc[mask] = cv2.GC_PR_FGD
    sure = cv2.erode(mask.astype(np.uint8), disk(max(1, int(cfg["gc_fg_erode"] * size / 2)))).astype(bool)
    gc[sure] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(bgr, gc, None, bgd, fgd, cfg["gc_iters"], cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    return (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)


def guarded_grabcut(bgr, mask, cfg):
    """GrabCut each selected egg separately; keep the refined blob only if it is still egg-shaped and
    has not grown more than gc_max_growth times (otherwise it leaked into similar background)."""
    out = np.zeros_like(mask)
    for comp in components(mask):
        refined = grabcut_refine(bgr, comp, cfg)
        parts = [p for p in components(refined) if np.logical_and(p, comp).any()]
        refined = np.any(parts, axis=0) if parts else comp
        s = shape_stats(refined)
        ok = (refined.sum() <= cfg["gc_max_growth"] * comp.sum() and s["solidity"] >= cfg["min_solidity"]
              and s["aspect"] <= cfg["max_aspect"])
        out |= refined if ok else comp
    return out


def ellipse_regularise(inst_mask, cfg):
    s = shape_stats(inst_mask)
    if s["ellipse"] is None or s["fill"] < cfg["ellipse_min_fill"]:
        return inst_mask
    out = np.zeros(inst_mask.shape, np.uint8)
    (cx, cy), (a, b), ang = s["ellipse"]
    cv2.ellipse(out, ((cx, cy), (a, b), ang), 1, -1)
    return out.astype(bool)


def split_touching(bgr, mask, cfg):
    """Watershed on the distance transform; a blob is split only if it has 2+ distinct peaks."""
    labels = np.zeros(mask.shape, np.int32)
    next_id = 1
    for comp in components(mask):
        dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
        peaks = (dist >= cfg["ws_peak"] * dist.max()).astype(np.uint8)
        n, markers = cv2.connectedComponents(peaks)
        if n - 1 < 2:
            labels[comp] = next_id
            next_id += 1
            continue
        markers = markers.astype(np.int32)
        markers[~comp] = 0
        ws_markers = markers.copy()
        ws_markers[~cv2.dilate(comp.astype(np.uint8), disk(2)).astype(bool)] = n  # background marker
        relief = cv2.cvtColor((255 - cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8),
                              cv2.COLOR_GRAY2BGR)
        cv2.watershed(relief, ws_markers)
        for k in range(1, n):
            part = (ws_markers == k) & comp
            if part.any():
                labels[part] = next_id
                next_id += 1
        leftover = comp & (labels == 0)
        if leftover.any():  # watershed lines: give them to the nearest instance
            near = cv2.dilate(labels.astype(np.float32), disk(2)).astype(np.int32)
            labels[leftover] = near[leftover]
    return labels


# ---------------------------------------------------------------- main entry

def segment(bgr_full, cfg):
    """Returns uint8 instance label map at the original resolution (0 = background)."""
    H, W = bgr_full.shape[:2]
    s = cfg["work_size"] / max(H, W)
    bgr = cv2.resize(bgr_full, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    bgr = cv2.bilateralFilter(bgr, 7, 40, 7)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    if cfg["method"] == "otsu_gray":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        score, mask = gray / 255.0, m > 0
    elif cfg["method"] == "hsv_white":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v_thr = np.percentile(hsv[..., 2], cfg["hsv_v_pct"])
        mask = (hsv[..., 1] <= cfg["hsv_s_max"]) & (hsv[..., 2] >= v_thr)
        score = hsv[..., 2] / 255.0
    else:
        score = egg_score(bgr, lab, cfg)
        u8 = (score * 255).astype(np.uint8)
        if cfg["thresh"] == "otsu":
            _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = m > 0
        else:
            mask = score >= cfg["thresh"]

    if cfg["skin"]:
        mask &= ~skin_mask(bgr)
    if cfg["open_r"]:
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disk(cfg["open_r"])).astype(bool)
    if cfg["close_r"]:
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(cfg["close_r"])).astype(bool)

    if cfg["select"] == "multilevel":
        mask = multilevel_select(score, cfg, lab)
    else:
        mask = select_components(mask, score, cfg)
    if cfg["grabcut"] and cfg["gc_guard"]:
        mask = guarded_grabcut(bgr, mask, cfg)
    elif cfg["grabcut"]:
        mask = grabcut_refine(bgr, mask, cfg)
        if cfg["open_r"]:
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disk(cfg["open_r"])).astype(bool)
        mask = select_components(mask, score, {**cfg, "select": "egg"} if cfg["select"] == "multilevel" else cfg)

    labels = split_touching(bgr, mask, cfg) if cfg["watershed"] else cv2.connectedComponents(mask.astype(np.uint8))[1]

    out = np.zeros((H, W), np.uint8)
    for k in range(1, labels.max() + 1):
        inst = labels == k
        if not inst.any():
            continue
        if cfg["fill_holes"]:
            inst = fill_holes(inst)
        if cfg["ellipse"]:
            inst = ellipse_regularise(inst, cfg)
        up = cv2.resize(inst.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        out[up & (out == 0)] = min(k, 255)
    # relabel consecutively
    ids = [i for i in np.unique(out) if i]
    relabeled = np.zeros_like(out)
    for new, old in enumerate(ids, 1):
        relabeled[out == old] = new
    return relabeled
