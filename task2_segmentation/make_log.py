"""Turns experiments/log.csv into the Markdown experiment log used in the report.

  python task2_segmentation/make_log.py   ->  task2_segmentation/EXPERIMENT_LOG.md
"""

import csv

from common import HERE, load_split
from trials import FINAL_ID

EXP = HERE / "experiments"


def main():
    rows = list(csv.DictReader(open(EXP / "log.csv")))
    split = load_split()
    lines = [
        "# Task 2 experiment log",
        "",
        f"All numbers are on the **dev split only** ({len(split['dev'])} images); the eval split "
        f"({len(split['eval'])} images) was not touched during development. Ground truth = Task 1 SAM masks.",
        "Metrics: per-image IoU / Dice / precision / recall averaged over images, BF1 = boundary F1 (3 px), "
        "inst F1 = instance F1 at IoU 0.5, |cnt| = mean absolute egg-count error.",
        "",
        "## Summary",
        "",
        "| ID | Trial | IoU mean | IoU median | Dice | P | R | BF1 | inst F1 | \\|cnt\\| | s/img | Decision |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    f = lambda r, k: f"{float(r[k]):.3f}"
    for r in rows:
        name = f"**{r['name']}**" if r["id"] == FINAL_ID else r["name"]
        lines.append(f"| {r['id']} | {name} | {f(r, 'iou_mean')} | {f(r, 'iou_median')} | {f(r, 'dice_mean')} | "
                     f"{f(r, 'precision')} | {f(r, 'recall')} | {f(r, 'bf1')} | {f(r, 'inst_f1')} | "
                     f"{float(r['abs_count_err']):.2f} | {float(r['sec_per_img']):.2f} | {r['decision']} |")
    lines += ["", "## Trials in detail", ""]
    for r in rows:
        lines += [
            f"### {r['id']} - {r['name']}",
            f"- **Why:** {r['why']}",
            f"- **Change / parameters:** {r['changes']}",
            f"- **Dev result:** IoU {f(r, 'iou_mean')} (median {f(r, 'iou_median')}), Dice {f(r, 'dice_mean')}, "
            f"P {f(r, 'precision')}, R {f(r, 'recall')}, BF1 {f(r, 'bf1')}, inst F1 {f(r, 'inst_f1')}, "
            f"images with IoU < 0.5: {float(r['frac_iou_lt_0.5']):.0%}",
            f"- **Decision:** {r['decision']} - {r['decision_reason']}",
            f"- Worst dev cases: `experiments/{r['id']}/worst.jpg`, full config: `experiments/{r['id']}/config.json`",
            "",
        ]
    (HERE / "EXPERIMENT_LOG.md").write_text("\n".join(lines))
    print("wrote", HERE / "EXPERIMENT_LOG.md")


if __name__ == "__main__":
    main()
