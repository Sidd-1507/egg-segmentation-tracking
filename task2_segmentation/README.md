# Task 2 — classical egg segmentation

No neural networks. The final algorithm is trial **T24** in `trials.py`. Every trial and its dev numbers are in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Split (`split.py`, `scenes.json`, `split.json`)

Leakage in this dataset is scene leakage, not duplicate files: the same marble counter, steel pot, cloth or hand is photographed several times. If one of those scenes is on both sides, eval rewards remembering the background.

Clustering on colour histograms (`python task2_segmentation/split.py --propose`) mixed scenes up, so every image was given a scene label by hand in `scenes.json` (22 scenes). Whole scenes are then assigned to eval: with seed 0 the script tries random scene subsets and keeps the one with 20–24 images that covers hands, white-on-white, textured backgrounds and multi-egg images, with a multi-egg share close to the whole dataset.

- **dev**: 87 images, 110 eggs, 14% multi-egg
- **eval**: 23 images, 28 eggs, 22% multi-egg. Scenes: blue_bucket, earphones_carpet, hand_backlit, red_stool, steel_pot, white_box. None of them appear in dev.

All tuning used dev only.

## Algorithm (`segment.py`)

One function, `segment(image, cfg)`. Every stage is a config switch, so each trial is the same code with different settings.

1. Resize to 320 px, bilateral filter, convert to Lab.
2. Two per-pixel cues:
   - **white**: normalised L × (1 − chroma/40). Bright and colourless.
   - **background distance**: k-means (k = 4) on the colours of a 6% border ring; each pixel's distance to the nearest border colour. This is what separates a white egg from a white bag.
3. Candidates: threshold five maps (combined score, white, background, L, Canny-enclosed regions) at 16 levels, open each level with radius 1, 4 and 8 px, and keep egg-shaped blobs (area 0.4–65%, aspect ≤ 2.2, ellipse fill ≥ 0.75, solidity ≥ 0.85, mean white ≥ 0.5).
4. Rank each candidate against a ring around it:
   `q = shape² × (white_in − white_ring + bg_in − bg_ring) × area^0.25 × (0.5 + edge support) × (1 − distance from centre)`.
   Keep candidates within 60% of the best that do not overlap a kept one.
5. GrabCut seeded from the candidates (eroded core = sure egg, 15% margin = probable background), egg-shape check again, fill holes, upsample.

## Experiments

24 trials. Kept steps on the way to T24, dev mean IoU:

| Step | Trial | Dev IoU |
|---|---|---|
| grey + Otsu baseline | T01 | 0.363 |
| HSV white | T03 | 0.429 |
| egg-shape selection | T05 | 0.439 |
| border background model | T06 | 0.484 |
| multi-level candidates | T08 | 0.492 |
| GrabCut + hole fill | T09 | 0.553 |
| ring-contrast ranking | T13 | 0.568 |
| multi-radius opening (best-candidate oracle 0.68 → 0.76) | T18 | 0.562 |
| edge / centre / shape in the rank | T19 | 0.593 |
| whiteness gate | **T24** | **0.612** |

Discarded: smoothness cue (T07, it removes the egg outline), edge-only ranking (T11), no GrabCut (T14), tighter or guarded GrabCut (T15, T16), mean-shift maps (T17), watershed splitting (T20), ellipse fitting (T21), keeping more or fewer eggs (T22, T23). The reasons are in the log.

`analyze_candidates.py` was used to decide what to work on. It checks whether a good candidate exists in the pool (the "oracle") and how it is ranked. At T10 the best candidate averaged IoU 0.68 while the chosen one averaged 0.51, which is why T13 and T19 work on ranking.

T24 on dev: IoU 0.612 (median 0.758), Dice 0.684, precision 0.718, recall 0.761, boundary F1 0.545, instance F1 0.613, mean count error 0.16.

## Files

```
common.py              paths, loading, metrics (IoU, Dice, P/R, boundary F1, HD95, instance F1)
split.py               scene-level split
segment.py             the algorithm
trials.py              all 24 trial configs, with why / what changed / decision
run_trials.py          runs trials on dev -> experiments/log.csv, experiments/TXX/
analyze_candidates.py  candidate-pool diagnostic
make_log.py            experiments/log.csv -> EXPERIMENT_LOG.md
final.py               frozen T24; writes predictions/
experiments/TXX/       per_image.csv, config.json, worst.jpg (green TP, red FP, blue FN)
split_groups/          contact sheets of every scene and which side it went to
predictions/           T24 label maps and overlays for dev and eval
```

## Run

```
python task2_segmentation/split.py
python task2_segmentation/run_trials.py          # trials not yet in the log
python task2_segmentation/run_trials.py T24      # one trial
python task2_segmentation/make_log.py
python task2_segmentation/final.py --split dev
python task2_segmentation/final.py --image some.jpg
```
