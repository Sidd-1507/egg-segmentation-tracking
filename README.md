# Egg segmentation and tracking

CV intern assignment. Four parts:

1. **SAM ground truth.** The 110-image Roboflow egg dataset has no masks, so I made them with Grounding DINO + SAM, then reviewed every one and fixed three by hand.
2. **Classical segmentation.** A non-deep-learning egg segmenter, built over 24 logged experiments on a leakage-aware dev split.
3. **Evaluation.** The frozen algorithm was run once on the held-out split, with metrics, plots and a failure analysis.
4. **Video tracking.** The two eggs in the [spinning-egg video](https://www.youtube.com/watch?v=Y4JyV94y2_A) are segmented and keep their own ids through hand occlusion and the spin.

- Report (PDF): inside the repo at `report.pdf`
- Tracked video (Google Drive): **https://drive.google.com/file/d/1BbzyooMvdoqHriA8DdoZCqx427-M0TEk/view?usp=sharing**

## Results

| | Dev (87 images) | Eval (23 images, unseen scenes) |
|---|---|---|
| IoU mean / median | 0.612 / 0.758 | 0.379 / 0.417 |
| Dice | 0.684 | 0.478 |
| Precision / recall | 0.718 / 0.761 | 0.543 / 0.605 |
| Boundary F1 (3 px) | 0.545 | 0.417 |
| HD95 (px) | 131 | 250 |
| Instance F1 @ IoU 0.5 | 0.613 | 0.354 |
| Abs. count error | 0.16 | 0.39 |

The eval drop is a scene shift. Every eval scene was absent from dev, and the main failures there are backlit eggs, large pale walls outranking the egg, and GrabCut growing into a pale floor or lid. Details are in [task3_evaluation/README.md](task3_evaluation/README.md).

Video: exactly 2 ids over 1440 frames; 640 frames show both eggs, 142 show one (the other is under the hand), 658 are title cards.

![IoU by scene](task3_evaluation/figures/iou_by_scene.png)

## Repository layout

```
scripts/
  download_data.sh        dataset -> data/raw
  download_video.sh       video   -> task4_tracking/input
task1_sam/                Task 1: Grounding DINO + SAM, manual fixes, output/ = ground truth
task2_segmentation/       Task 2: split, algorithm, 24 trials, experiment log, predictions
task3_evaluation/         Task 3: one-shot eval, results/, figures/, analysis (README)
task4_tracking/           Task 4: detector + tracker, sample frames
requirements.txt
```

Each task folder has its own README with the method and the commands.

## Setup

Python 3.11.

```
pip install -r requirements.txt
bash scripts/download_data.sh
bash scripts/download_video.sh
```

Task 1 needs a GPU (I ran it on Colab with `task1_sam/task1_colab.ipynb`). Its output is committed in `task1_sam/output/`, so Tasks 2–4 run on a CPU without it. torch and transformers are only needed for Task 1.

## Run

```
# Task 1 (GPU)
python task1_sam/run_sam.py --data_dir data/raw --out_dir task1_sam/output --fixes task1_sam/fixes.json
python task1_sam/refine_masks.py --data_dir data/raw --out_dir task1_sam/output --refinements task1_sam/refinements.json

# Task 2
python task2_segmentation/split.py
python task2_segmentation/run_trials.py
python task2_segmentation/make_log.py
python task2_segmentation/final.py --split dev

# Task 3 (eval split, run once)
python task3_evaluation/evaluate.py

# Task 4
python task4_tracking/track.py
```

To try the final segmenter on any image: `python task2_segmentation/final.py --image path/to/egg.jpg` (writes `egg_seg.jpg` next to it).

## Data

- Roboflow Universe, egg v1 by fire-bf7qe, CC BY 4.0: https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1. Roboflow's own train/valid/test folders were ignored and re-split by scene.
- Video: "Raw and Boiled Egg Spinning", MocomiKids, YouTube. It is not redistributed here.

## AI tool usage

I used an AI coding assistant (Google Antigravity / Gemini) to help write and debug Python code across all four tasks — things like fixing off-by-one errors in the evaluation script, debugging the tracker's Hungarian-matching logic, and cleaning up the folder structure. All experiment numbers, trial decisions, and metric values come from real runs on the actual data; I did not generate or guess any numbers. The experiment log records every trial in the order it was run.

