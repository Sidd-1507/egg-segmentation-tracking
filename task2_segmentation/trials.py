"""Experiment log source. Each trial = one config of segment.py, run on DEV only.
Trials were added one at a time after looking at the previous trial's numbers and worst cases;
"decision" is filled in after the run.
"""

from segment import cfg_with

TRIALS = [
    dict(id="T01", name="Gray + Otsu (baseline)",
         why="Simplest possible baseline: eggs are usually brighter than the background.",
         changes="grayscale, global Otsu, keep every bright pixel",
         cfg=cfg_with(method="otsu_gray"),
         decision="", reason=""),
    dict(id="T02", name="Gray + Otsu + morphology + largest blob",
         why="T01 keeps every bright pixel (cloth, walls, highlights); most images have one egg.",
         changes="open r=2, close r=4, keep largest component",
         cfg=cfg_with(method="otsu_gray", open_r=2, close_r=4, select="largest"),
         decision="", reason=""),
    dict(id="T03", name="HSV white (low S, high V)",
         why="Egg is white: low saturation and bright; gray alone also picks up bright coloured areas.",
         changes="S <= 60 and V >= 70th percentile; open 2, close 4, largest",
         cfg=cfg_with(method="hsv_white", hsv_s_max=60, hsv_v_pct=70, open_r=2, close_r=4, select="largest"),
         decision="", reason=""),
    dict(id="T04", name="Lab white cue + Otsu",
         why="Lab separates lightness from chroma more cleanly than HSV, and the cue is continuous so Otsu adapts per image.",
         changes="score = norm(L) * (1 - chroma/40); Otsu; open 2, close 4, largest",
         cfg=cfg_with(method="score", use_white=True, open_r=2, close_r=4, select="largest"),
         decision="", reason=""),
    dict(id="T05", name="T04 + egg-shape component selection",
         why="T02-T04 worst cases: 'largest bright blob' is often the wall, a label or white cloth, not the egg.",
         changes="select = blobs with solidity >= 0.85 and ellipse aspect <= 2.2, keep those >= 20% of the biggest",
         cfg=cfg_with(method="score", use_white=True, open_r=2, close_r=4, select="egg"),
         decision="", reason=""),
    dict(id="T06", name="T05 + border background model",
         why="White-on-white scenes (bag, tray, white box): egg and background are both white, so the white cue "
             "merges them. Border pixels are mostly background, so 'different from the border colours' separates them.",
         changes="k-means (k=4) on border-ring Lab pixels (L weight 0.5); cue = distance to nearest centre; "
                 "score = geometric mean(white, bg)",
         cfg=cfg_with(method="score", use_white=True, use_bg=True, open_r=2, close_r=4, select="egg"),
         decision="", reason=""),
    dict(id="T07", name="T06 + smoothness cue",
         why="Textured backgrounds (ash, cloth, carpet, keyboard) have pale pixels that pass the white and bg cues; "
             "eggshell is smooth.",
         changes="cue = exp(-local std of L in 7x7 / 8); score = geometric mean(white, bg, smooth)",
         cfg=cfg_with(method="score", use_white=True, use_bg=True, use_smooth=True, open_r=2, close_r=4, select="egg"),
         decision="", reason=""),
    dict(id="T08", name="T06 + multi-level threshold selection",
         why="T06 worst cases: at the single Otsu level the egg merges with cloth/bag/tray into one irregular blob, "
             "the shape filter rejects it and keeps a speck. At some other level the egg is a clean ellipse.",
         changes="16 thresholds between the 20th and 99.5th score percentile; candidates need fill >= 0.75, "
                 "solidity >= 0.85, aspect <= 2.2; quality = fill * solidity * (inside - ring contrast); "
                 "greedy keep with overlap suppression, keep quality >= 0.6 * best",
         cfg=cfg_with(method="score", use_white=True, use_bg=True, select="multilevel"),
         decision="", reason=""),
    dict(id="T09", name="T08 + GrabCut refinement + hole fill",
         why="T08 has precision 0.78 but recall 0.52: it finds the right object but keeps only the bright core "
             "(shaded side of the egg falls below the chosen level). GrabCut grows it using colour models.",
         changes="GrabCut init: selected blobs = probable FG, eroded core = sure FG, beyond a 15% margin = sure BG; "
                 "5 iters; re-select egg-shaped blobs; fill holes per instance",
         cfg=cfg_with(method="score", use_white=True, use_bg=True, select="multilevel", grabcut=True, fill_holes=True),
         decision="", reason=""),
]

_T09 = dict(method="score", use_white=True, use_bg=True, select="multilevel", grabcut=True, fill_holes=True)
_ALL_SOURCES = ("score", "white", "bg", "L", "edges")

TRIALS += [
    dict(id="T10", name="T09 + candidates from 5 maps (contrast rank)",
         why="T09 gives no prediction on 8 dev images (ash, foil, bag, tray): the combined score never isolates the "
             "egg. Cue maps show the egg separates in *some* map (L, white, bg) or as an edge-enclosed region.",
         changes="multilevel candidates from score, white, bg, L and Canny-enclosed regions; same contrast ranking",
         cfg=cfg_with(**_T09, ml_sources=_ALL_SOURCES, ml_quality="contrast"),
         decision="", reason=""),
    dict(id="T11", name="T10 with edge-support ranking",
         why="Contrast in a cue is weak exactly where cues fail; but the egg outline always sits on a strong "
             "luminance edge.",
         changes="quality = fill * solidity * fraction of outline pixels on gradients above the 80th percentile",
         cfg=cfg_with(**_T09, ml_sources=_ALL_SOURCES, ml_quality="edge"),
         decision="", reason=""),
    dict(id="T12", name="T10 with contrast + edge ranking",
         why="Combine both kinds of evidence: contrast rejects background patches, edges reject cue noise.",
         changes="quality evidence = mean(contrast, edge support)",
         cfg=cfg_with(**_T09, ml_sources=_ALL_SOURCES, ml_quality="both"),
         decision="", reason=""),
    dict(id="T13", name="T10 with ring-contrast + size ranking",
         why="Candidate diagnostic on T10: best candidate in the pool averages IoU 0.68 but the top-ranked one 0.51 "
             "(best is ranked first on only 10% of images). Good candidates are larger (median 8% vs 0.9% of the image) "
             "and whiter / more unlike the border than the ring around them (AUC 0.89 / 0.80).",
         changes="quality = fill * solidity * (max(white_in - white_ring, 0) + max(bg_in - bg_ring, 0)) * area^0.25 "
                 "(formula picked offline on dev candidates: top-1 IoU 0.51 -> 0.56)",
         cfg=cfg_with(**_T09, ml_sources=_ALL_SOURCES, ml_quality="ring"),
         decision="", reason=""),
]

_T13 = {**_T09, "ml_sources": _ALL_SOURCES, "ml_quality": "ring"}

TRIALS += [
    dict(id="T14", name="T13 without GrabCut",
         why="T13 worst cases include leaks (IoU 0.2-0.36 with low precision) on cloth, bag, rice and food container: "
             "the right egg is chosen, then GrabCut floods into similar-coloured background. Ablation.",
         changes="grabcut off (candidate + hole fill only)",
         cfg=cfg_with(**{**_T13, "grabcut": False}),
         decision="", reason=""),
    dict(id="T15", name="T13 with tighter GrabCut margin",
         why="Limit how far GrabCut may grow: sure-background starts closer to the candidate.",
         changes="gc_bg_margin 0.15 -> 0.05",
         cfg=cfg_with(**{**_T13, "gc_bg_margin": 0.05}),
         decision="", reason=""),
    dict(id="T16", name="T13 with guarded per-egg GrabCut",
         why="Keep GrabCut's gain where it works (shaded side of the egg) and undo it where it leaks.",
         changes="GrabCut per selected egg; revert to the candidate if it grows > 1.6x, or solidity < 0.85, "
                 "or aspect > 2.2",
         cfg=cfg_with(**{**_T13, "gc_guard": True}),
         decision="", reason=""),
    dict(id="T17", name="T13 + mean-shift candidate maps",
         why="14 dev images have no usable candidate at all (dusty eggs on ash, white-on-white bag/tray/rice). "
             "Mean-shift filtering flattens texture but keeps strong edges, so a dusty egg becomes one uniform blob.",
         changes="extra candidate maps: L, white and bg computed on a mean-shift filtered image (sp=12, sr=24)",
         cfg=cfg_with(**{**_T13, "ml_sources": _ALL_SOURCES + ("L_ms", "white_ms", "bg_ms")}),
         decision="", reason=""),
    dict(id="T18", name="T13 + multi-radius opening of candidates",
         why="In the remaining misses the egg is a thick convex blob joined to background by a thin bridge at "
             "every threshold (egg base touching the white bag, the tray, the rice). A larger opening cuts the bridge.",
         changes="each threshold level opened with radius 1, 4 and 8 px (at 320 px work size)",
         cfg=cfg_with(**{**_T13, "ml_open_radii": (1, 4, 8)}),
         decision="", reason=""),
    dict(id="T19", name="T18 + ranking with edge, centre and shape terms",
         why="T18 raised the candidate-pool oracle from 0.68 to 0.76 but top-1 stayed ~0.56: ranking is now the "
             "bottleneck. On T18 dev candidates, distance from centre (AUC 0.22, i.e. good ones are central), edge "
             "support (0.76) and shape add information to the ring contrast.",
         changes="quality = shape^2 * (white_d + bg_d) * area^0.25 * (0.5 + edge_support) * (1 - centre_distance) "
                 "(offline top-1 on dev candidates 0.56 -> 0.60)",
         cfg=cfg_with(**{**_T13, "ml_open_radii": (1, 4, 8), "ml_quality": "ring2"}),
         decision="", reason=""),
]

_T19 = {**_T13, "ml_open_radii": (1, 4, 8), "ml_quality": "ring2"}

TRIALS += [
    dict(id="T20", name="T19 + watershed split of touching eggs",
         why="Marble-counter images have 2-3 touching eggs; GrabCut can merge them into one blob (count error -1/-2).",
         changes="distance-transform watershed per blob; split only if 2+ peaks above 0.6 * max distance",
         cfg=cfg_with(**{**_T19, "watershed": True}),
         decision="", reason=""),
    dict(id="T21", name="T19 + ellipse regularisation",
         why="Eggs are near-ellipses; replacing each mask by its fitted ellipse could fix ragged GrabCut edges "
             "and shadow bites.",
         changes="replace instance by fitted ellipse when mask fills >= 85% of it",
         cfg=cfg_with(**{**_T19, "ellipse": True}),
         decision="", reason=""),
    dict(id="T22", name="T19, keep more eggs (rel quality 0.4)",
         why="Multi-egg images are under-counted more often than over-counted.",
         changes="ml_rel_quality 0.6 -> 0.4",
         cfg=cfg_with(**{**_T19, "ml_rel_quality": 0.4}),
         decision="", reason=""),
    dict(id="T23", name="T19, keep fewer eggs (rel quality 0.8)",
         why="Opposite direction of T22, to bracket the parameter.",
         changes="ml_rel_quality 0.6 -> 0.8",
         cfg=cfg_with(**{**_T19, "ml_rel_quality": 0.8}),
         decision="", reason=""),
    dict(id="T24", name="T19 + whiteness gate on candidates",
         why="T19 failures are mostly wrong picks (29 of the IoU<0.5 images have precision < 0.5): orange lid, "
             "blue print on the bag, foil. 99% of good dev candidates have mean white cue >= 0.54.",
         changes="drop candidates with mean white cue < 0.5 (offline top-1 0.601 -> 0.609)",
         cfg=cfg_with(**{**_T19, "ml_min_white": 0.5}),
         decision="", reason=""),
]

# Filled in after each run (dev numbers from experiments/log.csv).
DECISIONS = {
    "T01": ("baseline", "Reference point: IoU 0.363; keeps every bright pixel, |count err| 1.47."),
    "T02": ("kept", "IoU 0.363 -> 0.370 and |count err| 1.47 -> 0.47; 'largest blob' often picks wall/cloth instead of egg."),
    "T03": ("kept", "IoU 0.370 -> 0.429: a whiteness prior beats raw brightness."),
    "T04": ("kept", "Mean IoU 0.416 (vs 0.429) but BF1 0.259 -> 0.298 and instance F1 up; a continuous 0-1 cue is needed to fuse more cues."),
    "T05": ("kept", "IoU 0.416 -> 0.439, instance F1 0.347 -> 0.405: shape beats size for choosing the egg."),
    "T06": ("kept", "IoU 0.439 -> 0.484, median 0.308 -> 0.536, |count err| 0.33 -> 0.16: fixes most white-on-white scenes."),
    "T07": ("discarded", "IoU 0.484 -> 0.427 and BF1 0.317 -> 0.047: texture is high exactly on the egg outline, so the mask shrinks."),
    "T08": ("kept", "IoU 0.484 -> 0.492, precision 0.663 -> 0.779; recall drops (bright core only) but it gives clean seeds for GrabCut."),
    "T09": ("kept", "IoU 0.492 -> 0.553, BF1 0.342 -> 0.471: GrabCut recovers the shaded side of the egg."),
    "T10": ("kept", "Mean IoU 0.553 -> 0.549 but median 0.618 -> 0.674 and recall 0.569 -> 0.642; bigger candidate pool needed for better ranking."),
    "T11": ("discarded", "IoU 0.549 -> 0.444: edge-only ranking prefers round background structures (tray cups, bowl rims)."),
    "T12": ("discarded", "IoU 0.500, below contrast-only T10."),
    "T13": ("kept", "IoU 0.549 -> 0.568, median 0.697, recall 0.723: ranking picks whole eggs instead of fragments."),
    "T14": ("discarded", "Ablation: IoU 0.568 -> 0.541 and BF1 0.510 -> 0.403 without GrabCut, so GrabCut stays; leaks come from wrong candidates, not GrabCut."),
    "T15": ("discarded", "IoU 0.568 -> 0.548: tighter margin cuts off real egg as well as leaks."),
    "T16": ("discarded", "IoU 0.568 -> 0.573 but median 0.697 -> 0.659 and |count err| 0.23 -> 0.41; not worth the complexity."),
    "T17": ("discarded", "IoU 0.568 -> 0.553 and 0.73 s/img; candidate oracle only 0.678 -> 0.695."),
    "T18": ("kept", "Mean IoU 0.562 (flat) but candidate oracle 0.678 -> 0.755 with no image below 0.18; enables T19."),
    "T19": ("kept", "IoU 0.562 -> 0.593, median 0.683 -> 0.758, BF1 0.528."),
    "T20": ("discarded", "IoU unchanged (0.593), |count err| 0.18 -> 0.25: splits single eggs with highlights more often than it separates touching eggs."),
    "T21": ("discarded", "IoU 0.593 -> 0.572, BF1 0.528 -> 0.364: SAM masks follow occluders (fingers, leaves, rods), an ellipse cannot."),
    "T22": ("discarded", "IoU 0.593 -> 0.543, instance F1 0.571 -> 0.495: extra false eggs."),
    "T23": ("discarded", "IoU 0.593 -> 0.590, instance F1 up but |count err| 0.18 -> 0.30 (misses real second/third eggs); keep 0.6."),
    "T24": ("kept (FINAL)", "IoU 0.593 -> 0.612, median 0.758, Dice 0.684, P 0.718, R 0.761, BF1 0.545, instance F1 0.613, |count err| 0.16; best on every metric."),
}
for _t in TRIALS:
    _t["decision"], _t["reason"] = DECISIONS.get(_t["id"], ("", ""))

FINAL_ID = "T24"
