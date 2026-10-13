# Task 2 experiment log

All numbers are on the **dev split only** (87 images); the eval split (23 images) was not touched during development. Ground truth = Task 1 SAM masks.
Metrics: per-image IoU / Dice / precision / recall averaged over images, BF1 = boundary F1 (3 px), inst F1 = instance F1 at IoU 0.5, |cnt| = mean absolute egg-count error.

## Summary

| ID | Trial | IoU mean | IoU median | Dice | P | R | BF1 | inst F1 | \|cnt\| | s/img | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T01 | Gray + Otsu (baseline) | 0.363 | 0.289 | 0.461 | 0.408 | 0.680 | 0.229 | 0.282 | 1.47 | 0.03 | baseline |
| T02 | Gray + Otsu + morphology + largest blob | 0.370 | 0.281 | 0.455 | 0.430 | 0.601 | 0.245 | 0.301 | 0.47 | 0.02 | kept |
| T03 | HSV white (low S, high V) | 0.429 | 0.359 | 0.535 | 0.518 | 0.631 | 0.259 | 0.328 | 0.28 | 0.02 | kept |
| T04 | Lab white cue + Otsu | 0.416 | 0.288 | 0.504 | 0.486 | 0.675 | 0.298 | 0.347 | 0.38 | 0.03 | kept |
| T05 | T04 + egg-shape component selection | 0.439 | 0.308 | 0.530 | 0.529 | 0.687 | 0.321 | 0.405 | 0.33 | 0.03 | kept |
| T06 | T05 + border background model | 0.484 | 0.536 | 0.594 | 0.663 | 0.676 | 0.317 | 0.524 | 0.16 | 0.05 | kept |
| T07 | T06 + smoothness cue | 0.427 | 0.480 | 0.529 | 0.595 | 0.577 | 0.047 | 0.450 | 0.33 | 0.05 | discarded |
| T08 | T06 + multi-level threshold selection | 0.492 | 0.535 | 0.587 | 0.779 | 0.518 | 0.342 | 0.506 | 0.23 | 0.08 | kept |
| T09 | T08 + GrabCut refinement + hole fill | 0.553 | 0.618 | 0.629 | 0.782 | 0.569 | 0.471 | 0.584 | 0.21 | 0.22 | kept |
| T10 | T09 + candidates from 5 maps (contrast rank) | 0.549 | 0.674 | 0.617 | 0.679 | 0.642 | 0.492 | 0.569 | 0.25 | 0.40 | kept |
| T11 | T10 with edge-support ranking | 0.444 | 0.333 | 0.513 | 0.516 | 0.633 | 0.423 | 0.409 | 0.44 | 0.39 | discarded |
| T12 | T10 with contrast + edge ranking | 0.500 | 0.544 | 0.569 | 0.586 | 0.653 | 0.458 | 0.481 | 0.32 | 0.43 | discarded |
| T13 | T10 with ring-contrast + size ranking | 0.568 | 0.697 | 0.639 | 0.666 | 0.723 | 0.510 | 0.567 | 0.23 | 0.44 | kept |
| T14 | T13 without GrabCut | 0.541 | 0.624 | 0.627 | 0.666 | 0.732 | 0.403 | 0.538 | 0.36 | 0.23 | discarded |
| T15 | T13 with tighter GrabCut margin | 0.548 | 0.625 | 0.627 | 0.673 | 0.697 | 0.479 | 0.531 | 0.24 | 0.43 | discarded |
| T16 | T13 with guarded per-egg GrabCut | 0.573 | 0.658 | 0.650 | 0.672 | 0.767 | 0.504 | 0.551 | 0.41 | 0.48 | discarded |
| T17 | T13 + mean-shift candidate maps | 0.553 | 0.605 | 0.627 | 0.654 | 0.711 | 0.500 | 0.559 | 0.23 | 0.73 | discarded |
| T18 | T13 + multi-radius opening of candidates | 0.562 | 0.683 | 0.633 | 0.657 | 0.719 | 0.495 | 0.530 | 0.16 | 0.85 | kept |
| T19 | T18 + ranking with edge, centre and shape terms | 0.593 | 0.758 | 0.659 | 0.680 | 0.732 | 0.528 | 0.571 | 0.18 | 0.85 | kept |
| T20 | T19 + watershed split of touching eggs | 0.593 | 0.758 | 0.659 | 0.680 | 0.732 | 0.528 | 0.575 | 0.25 | 0.86 | discarded |
| T21 | T19 + ellipse regularisation | 0.572 | 0.709 | 0.645 | 0.661 | 0.726 | 0.364 | 0.563 | 0.18 | 0.82 | discarded |
| T22 | T19, keep more eggs (rel quality 0.4) | 0.543 | 0.611 | 0.615 | 0.619 | 0.739 | 0.485 | 0.495 | 0.26 | 0.87 | discarded |
| T23 | T19, keep fewer eggs (rel quality 0.8) | 0.590 | 0.692 | 0.662 | 0.717 | 0.701 | 0.521 | 0.608 | 0.30 | 0.79 | discarded |
| T24 | **T19 + whiteness gate on candidates** | 0.612 | 0.758 | 0.684 | 0.718 | 0.761 | 0.545 | 0.613 | 0.16 | 0.79 | kept (FINAL) |

## Trials in detail

### T01 - Gray + Otsu (baseline)
- **Why:** Simplest possible baseline: eggs are usually brighter than the background.
- **Change / parameters:** grayscale, global Otsu, keep every bright pixel
- **Dev result:** IoU 0.363 (median 0.289), Dice 0.461, P 0.408, R 0.680, BF1 0.229, inst F1 0.282, images with IoU < 0.5: 68%
- **Decision:** baseline - Reference point: IoU 0.363; keeps every bright pixel, |count err| 1.47.
- Worst dev cases: `experiments/T01/worst.jpg`, full config: `experiments/T01/config.json`

### T02 - Gray + Otsu + morphology + largest blob
- **Why:** T01 keeps every bright pixel (cloth, walls, highlights); most images have one egg.
- **Change / parameters:** open r=2, close r=4, keep largest component
- **Dev result:** IoU 0.370 (median 0.281), Dice 0.455, P 0.430, R 0.601, BF1 0.245, inst F1 0.301, images with IoU < 0.5: 67%
- **Decision:** kept - IoU 0.363 -> 0.370 and |count err| 1.47 -> 0.47; 'largest blob' often picks wall/cloth instead of egg.
- Worst dev cases: `experiments/T02/worst.jpg`, full config: `experiments/T02/config.json`

### T03 - HSV white (low S, high V)
- **Why:** Egg is white: low saturation and bright; gray alone also picks up bright coloured areas.
- **Change / parameters:** S <= 60 and V >= 70th percentile; open 2, close 4, largest
- **Dev result:** IoU 0.429 (median 0.359), Dice 0.535, P 0.518, R 0.631, BF1 0.259, inst F1 0.328, images with IoU < 0.5: 62%
- **Decision:** kept - IoU 0.370 -> 0.429: a whiteness prior beats raw brightness.
- Worst dev cases: `experiments/T03/worst.jpg`, full config: `experiments/T03/config.json`

### T04 - Lab white cue + Otsu
- **Why:** Lab separates lightness from chroma more cleanly than HSV, and the cue is continuous so Otsu adapts per image.
- **Change / parameters:** score = norm(L) * (1 - chroma/40); Otsu; open 2, close 4, largest
- **Dev result:** IoU 0.416 (median 0.288), Dice 0.504, P 0.486, R 0.675, BF1 0.298, inst F1 0.347, images with IoU < 0.5: 61%
- **Decision:** kept - Mean IoU 0.416 (vs 0.429) but BF1 0.259 -> 0.298 and instance F1 up; a continuous 0-1 cue is needed to fuse more cues.
- Worst dev cases: `experiments/T04/worst.jpg`, full config: `experiments/T04/config.json`

### T05 - T04 + egg-shape component selection
- **Why:** T02-T04 worst cases: 'largest bright blob' is often the wall, a label or white cloth, not the egg.
- **Change / parameters:** select = blobs with solidity >= 0.85 and ellipse aspect <= 2.2, keep those >= 20% of the biggest
- **Dev result:** IoU 0.439 (median 0.308), Dice 0.530, P 0.529, R 0.687, BF1 0.321, inst F1 0.405, images with IoU < 0.5: 56%
- **Decision:** kept - IoU 0.416 -> 0.439, instance F1 0.347 -> 0.405: shape beats size for choosing the egg.
- Worst dev cases: `experiments/T05/worst.jpg`, full config: `experiments/T05/config.json`

### T06 - T05 + border background model
- **Why:** White-on-white scenes (bag, tray, white box): egg and background are both white, so the white cue merges them. Border pixels are mostly background, so 'different from the border colours' separates them.
- **Change / parameters:** k-means (k=4) on border-ring Lab pixels (L weight 0.5); cue = distance to nearest centre; score = geometric mean(white, bg)
- **Dev result:** IoU 0.484 (median 0.536), Dice 0.594, P 0.663, R 0.676, BF1 0.317, inst F1 0.524, images with IoU < 0.5: 46%
- **Decision:** kept - IoU 0.439 -> 0.484, median 0.308 -> 0.536, |count err| 0.33 -> 0.16: fixes most white-on-white scenes.
- Worst dev cases: `experiments/T06/worst.jpg`, full config: `experiments/T06/config.json`

### T07 - T06 + smoothness cue
- **Why:** Textured backgrounds (ash, cloth, carpet, keyboard) have pale pixels that pass the white and bg cues; eggshell is smooth.
- **Change / parameters:** cue = exp(-local std of L in 7x7 / 8); score = geometric mean(white, bg, smooth)
- **Dev result:** IoU 0.427 (median 0.480), Dice 0.529, P 0.595, R 0.577, BF1 0.047, inst F1 0.450, images with IoU < 0.5: 54%
- **Decision:** discarded - IoU 0.484 -> 0.427 and BF1 0.317 -> 0.047: texture is high exactly on the egg outline, so the mask shrinks.
- Worst dev cases: `experiments/T07/worst.jpg`, full config: `experiments/T07/config.json`

### T08 - T06 + multi-level threshold selection
- **Why:** T06 worst cases: at the single Otsu level the egg merges with cloth/bag/tray into one irregular blob, the shape filter rejects it and keeps a speck. At some other level the egg is a clean ellipse.
- **Change / parameters:** 16 thresholds between the 20th and 99.5th score percentile; candidates need fill >= 0.75, solidity >= 0.85, aspect <= 2.2; quality = fill * solidity * (inside - ring contrast); greedy keep with overlap suppression, keep quality >= 0.6 * best
- **Dev result:** IoU 0.492 (median 0.535), Dice 0.587, P 0.779, R 0.518, BF1 0.342, inst F1 0.506, images with IoU < 0.5: 47%
- **Decision:** kept - IoU 0.484 -> 0.492, precision 0.663 -> 0.779; recall drops (bright core only) but it gives clean seeds for GrabCut.
- Worst dev cases: `experiments/T08/worst.jpg`, full config: `experiments/T08/config.json`

### T09 - T08 + GrabCut refinement + hole fill
- **Why:** T08 has precision 0.78 but recall 0.52: it finds the right object but keeps only the bright core (shaded side of the egg falls below the chosen level). GrabCut grows it using colour models.
- **Change / parameters:** GrabCut init: selected blobs = probable FG, eroded core = sure FG, beyond a 15% margin = sure BG; 5 iters; re-select egg-shaped blobs; fill holes per instance
- **Dev result:** IoU 0.553 (median 0.618), Dice 0.629, P 0.782, R 0.569, BF1 0.471, inst F1 0.584, images with IoU < 0.5: 40%
- **Decision:** kept - IoU 0.492 -> 0.553, BF1 0.342 -> 0.471: GrabCut recovers the shaded side of the egg.
- Worst dev cases: `experiments/T09/worst.jpg`, full config: `experiments/T09/config.json`

### T10 - T09 + candidates from 5 maps (contrast rank)
- **Why:** T09 gives no prediction on 8 dev images (ash, foil, bag, tray): the combined score never isolates the egg. Cue maps show the egg separates in *some* map (L, white, bg) or as an edge-enclosed region.
- **Change / parameters:** multilevel candidates from score, white, bg, L and Canny-enclosed regions; same contrast ranking
- **Dev result:** IoU 0.549 (median 0.674), Dice 0.617, P 0.679, R 0.642, BF1 0.492, inst F1 0.569, images with IoU < 0.5: 43%
- **Decision:** kept - Mean IoU 0.553 -> 0.549 but median 0.618 -> 0.674 and recall 0.569 -> 0.642; bigger candidate pool needed for better ranking.
- Worst dev cases: `experiments/T10/worst.jpg`, full config: `experiments/T10/config.json`

### T11 - T10 with edge-support ranking
- **Why:** Contrast in a cue is weak exactly where cues fail; but the egg outline always sits on a strong luminance edge.
- **Change / parameters:** quality = fill * solidity * fraction of outline pixels on gradients above the 80th percentile
- **Dev result:** IoU 0.444 (median 0.333), Dice 0.513, P 0.516, R 0.633, BF1 0.423, inst F1 0.409, images with IoU < 0.5: 57%
- **Decision:** discarded - IoU 0.549 -> 0.444: edge-only ranking prefers round background structures (tray cups, bowl rims).
- Worst dev cases: `experiments/T11/worst.jpg`, full config: `experiments/T11/config.json`

### T12 - T10 with contrast + edge ranking
- **Why:** Combine both kinds of evidence: contrast rejects background patches, edges reject cue noise.
- **Change / parameters:** quality evidence = mean(contrast, edge support)
- **Dev result:** IoU 0.500 (median 0.544), Dice 0.569, P 0.586, R 0.653, BF1 0.458, inst F1 0.481, images with IoU < 0.5: 49%
- **Decision:** discarded - IoU 0.500, below contrast-only T10.
- Worst dev cases: `experiments/T12/worst.jpg`, full config: `experiments/T12/config.json`

### T13 - T10 with ring-contrast + size ranking
- **Why:** Candidate diagnostic on T10: best candidate in the pool averages IoU 0.68 but the top-ranked one 0.51 (best is ranked first on only 10% of images). Good candidates are larger (median 8% vs 0.9% of the image) and whiter / more unlike the border than the ring around them (AUC 0.89 / 0.80).
- **Change / parameters:** quality = fill * solidity * (max(white_in - white_ring, 0) + max(bg_in - bg_ring, 0)) * area^0.25 (formula picked offline on dev candidates: top-1 IoU 0.51 -> 0.56)
- **Dev result:** IoU 0.568 (median 0.697), Dice 0.639, P 0.666, R 0.723, BF1 0.510, inst F1 0.567, images with IoU < 0.5: 41%
- **Decision:** kept - IoU 0.549 -> 0.568, median 0.697, recall 0.723: ranking picks whole eggs instead of fragments.
- Worst dev cases: `experiments/T13/worst.jpg`, full config: `experiments/T13/config.json`

### T14 - T13 without GrabCut
- **Why:** T13 worst cases include leaks (IoU 0.2-0.36 with low precision) on cloth, bag, rice and food container: the right egg is chosen, then GrabCut floods into similar-coloured background. Ablation.
- **Change / parameters:** grabcut off (candidate + hole fill only)
- **Dev result:** IoU 0.541 (median 0.624), Dice 0.627, P 0.666, R 0.732, BF1 0.403, inst F1 0.538, images with IoU < 0.5: 45%
- **Decision:** discarded - Ablation: IoU 0.568 -> 0.541 and BF1 0.510 -> 0.403 without GrabCut, so GrabCut stays; leaks come from wrong candidates, not GrabCut.
- Worst dev cases: `experiments/T14/worst.jpg`, full config: `experiments/T14/config.json`

### T15 - T13 with tighter GrabCut margin
- **Why:** Limit how far GrabCut may grow: sure-background starts closer to the candidate.
- **Change / parameters:** gc_bg_margin 0.15 -> 0.05
- **Dev result:** IoU 0.548 (median 0.625), Dice 0.627, P 0.673, R 0.697, BF1 0.479, inst F1 0.531, images with IoU < 0.5: 45%
- **Decision:** discarded - IoU 0.568 -> 0.548: tighter margin cuts off real egg as well as leaks.
- Worst dev cases: `experiments/T15/worst.jpg`, full config: `experiments/T15/config.json`

### T16 - T13 with guarded per-egg GrabCut
- **Why:** Keep GrabCut's gain where it works (shaded side of the egg) and undo it where it leaks.
- **Change / parameters:** GrabCut per selected egg; revert to the candidate if it grows > 1.6x, or solidity < 0.85, or aspect > 2.2
- **Dev result:** IoU 0.573 (median 0.658), Dice 0.650, P 0.672, R 0.767, BF1 0.504, inst F1 0.551, images with IoU < 0.5: 41%
- **Decision:** discarded - IoU 0.568 -> 0.573 but median 0.697 -> 0.659 and |count err| 0.23 -> 0.41; not worth the complexity.
- Worst dev cases: `experiments/T16/worst.jpg`, full config: `experiments/T16/config.json`

### T17 - T13 + mean-shift candidate maps
- **Why:** 14 dev images have no usable candidate at all (dusty eggs on ash, white-on-white bag/tray/rice). Mean-shift filtering flattens texture but keeps strong edges, so a dusty egg becomes one uniform blob.
- **Change / parameters:** extra candidate maps: L, white and bg computed on a mean-shift filtered image (sp=12, sr=24)
- **Dev result:** IoU 0.553 (median 0.605), Dice 0.627, P 0.654, R 0.711, BF1 0.500, inst F1 0.559, images with IoU < 0.5: 44%
- **Decision:** discarded - IoU 0.568 -> 0.553 and 0.73 s/img; candidate oracle only 0.678 -> 0.695.
- Worst dev cases: `experiments/T17/worst.jpg`, full config: `experiments/T17/config.json`

### T18 - T13 + multi-radius opening of candidates
- **Why:** In the remaining misses the egg is a thick convex blob joined to background by a thin bridge at every threshold (egg base touching the white bag, the tray, the rice). A larger opening cuts the bridge.
- **Change / parameters:** each threshold level opened with radius 1, 4 and 8 px (at 320 px work size)
- **Dev result:** IoU 0.562 (median 0.683), Dice 0.633, P 0.657, R 0.719, BF1 0.495, inst F1 0.530, images with IoU < 0.5: 44%
- **Decision:** kept - Mean IoU 0.562 (flat) but candidate oracle 0.678 -> 0.755 with no image below 0.18; enables T19.
- Worst dev cases: `experiments/T18/worst.jpg`, full config: `experiments/T18/config.json`

### T19 - T18 + ranking with edge, centre and shape terms
- **Why:** T18 raised the candidate-pool oracle from 0.68 to 0.76 but top-1 stayed ~0.56: ranking is now the bottleneck. On T18 dev candidates, distance from centre (AUC 0.22, i.e. good ones are central), edge support (0.76) and shape add information to the ring contrast.
- **Change / parameters:** quality = shape^2 * (white_d + bg_d) * area^0.25 * (0.5 + edge_support) * (1 - centre_distance) (offline top-1 on dev candidates 0.56 -> 0.60)
- **Dev result:** IoU 0.593 (median 0.758), Dice 0.659, P 0.680, R 0.732, BF1 0.528, inst F1 0.571, images with IoU < 0.5: 40%
- **Decision:** kept - IoU 0.562 -> 0.593, median 0.683 -> 0.758, BF1 0.528.
- Worst dev cases: `experiments/T19/worst.jpg`, full config: `experiments/T19/config.json`

### T20 - T19 + watershed split of touching eggs
- **Why:** Marble-counter images have 2-3 touching eggs; GrabCut can merge them into one blob (count error -1/-2).
- **Change / parameters:** distance-transform watershed per blob; split only if 2+ peaks above 0.6 * max distance
- **Dev result:** IoU 0.593 (median 0.758), Dice 0.659, P 0.680, R 0.732, BF1 0.528, inst F1 0.575, images with IoU < 0.5: 40%
- **Decision:** discarded - IoU unchanged (0.593), |count err| 0.18 -> 0.25: splits single eggs with highlights more often than it separates touching eggs.
- Worst dev cases: `experiments/T20/worst.jpg`, full config: `experiments/T20/config.json`

### T21 - T19 + ellipse regularisation
- **Why:** Eggs are near-ellipses; replacing each mask by its fitted ellipse could fix ragged GrabCut edges and shadow bites.
- **Change / parameters:** replace instance by fitted ellipse when mask fills >= 85% of it
- **Dev result:** IoU 0.572 (median 0.709), Dice 0.645, P 0.661, R 0.726, BF1 0.364, inst F1 0.563, images with IoU < 0.5: 40%
- **Decision:** discarded - IoU 0.593 -> 0.572, BF1 0.528 -> 0.364: SAM masks follow occluders (fingers, leaves, rods), an ellipse cannot.
- Worst dev cases: `experiments/T21/worst.jpg`, full config: `experiments/T21/config.json`

### T22 - T19, keep more eggs (rel quality 0.4)
- **Why:** Multi-egg images are under-counted more often than over-counted.
- **Change / parameters:** ml_rel_quality 0.6 -> 0.4
- **Dev result:** IoU 0.543 (median 0.611), Dice 0.615, P 0.619, R 0.739, BF1 0.485, inst F1 0.495, images with IoU < 0.5: 47%
- **Decision:** discarded - IoU 0.593 -> 0.543, instance F1 0.571 -> 0.495: extra false eggs.
- Worst dev cases: `experiments/T22/worst.jpg`, full config: `experiments/T22/config.json`

### T23 - T19, keep fewer eggs (rel quality 0.8)
- **Why:** Opposite direction of T22, to bracket the parameter.
- **Change / parameters:** ml_rel_quality 0.6 -> 0.8
- **Dev result:** IoU 0.590 (median 0.692), Dice 0.662, P 0.717, R 0.701, BF1 0.521, inst F1 0.608, images with IoU < 0.5: 40%
- **Decision:** discarded - IoU 0.593 -> 0.590, instance F1 up but |count err| 0.18 -> 0.30 (misses real second/third eggs); keep 0.6.
- Worst dev cases: `experiments/T23/worst.jpg`, full config: `experiments/T23/config.json`

### T24 - T19 + whiteness gate on candidates
- **Why:** T19 failures are mostly wrong picks (29 of the IoU<0.5 images have precision < 0.5): orange lid, blue print on the bag, foil. 99% of good dev candidates have mean white cue >= 0.54.
- **Change / parameters:** drop candidates with mean white cue < 0.5 (offline top-1 0.601 -> 0.609)
- **Dev result:** IoU 0.612 (median 0.758), Dice 0.684, P 0.718, R 0.761, BF1 0.545, inst F1 0.613, images with IoU < 0.5: 37%
- **Decision:** kept (FINAL) - IoU 0.593 -> 0.612, median 0.758, Dice 0.684, P 0.718, R 0.761, BF1 0.545, instance F1 0.613, |count err| 0.16; best on every metric.
- Worst dev cases: `experiments/T24/worst.jpg`, full config: `experiments/T24/config.json`
