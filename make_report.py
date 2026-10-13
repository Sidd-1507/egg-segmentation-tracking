"""Generate report.pdf for the egg segmentation assignment."""
import sys
sys.path.insert(0, "/tmp/pdflibs")

from fpdf import FPDF
import json, os

# -- helpers ------------------------------------------------------------------
HERE = "/Users/sidd1507/Desktop/NOSH_ROBOFLOW_TASK"

def load_summary():
    with open(os.path.join(HERE, "task3_evaluation/results/summary.json")) as f:
        return json.load(f)

S = load_summary()
dev  = S["dev"]
evl  = S["eval"]

# -- PDF class -----------------------------------------------------------------
class Report(FPDF):
    TITLE_COLOR  = (30, 80, 160)
    H1_COLOR     = (30, 80, 160)
    H2_COLOR     = (60, 110, 190)
    BODY_COLOR   = (30, 30, 30)
    ACCENT_COLOR = (220, 235, 255)
    BORDER_COLOR = (180, 200, 230)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(245, 248, 255)
        self.rect(0, 0, 210, 12, "F")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 130, 160)
        self.set_xy(10, 3)
        self.cell(0, 6, "Egg Segmentation & Tracking - CV Intern Assignment", align="L")
        self.set_xy(0, 3)
        self.cell(200, 6, f"Page {self.page_no()}", align="R")
        self.ln(6)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(160, 160, 160)
        self.cell(0, 6, "Dataset: Roboflow egg v1 (CC BY 4.0) - Video: MocomiKids, YouTube", align="C")

    # -- section helpers -------------------------------------------------------
    def cover(self):
        self.add_page()
        # top bar
        self.set_fill_color(*self.TITLE_COLOR)
        self.rect(0, 0, 210, 55, "F")
        self.set_font("Helvetica", "B", 26)
        self.set_text_color(255, 255, 255)
        self.set_xy(15, 14)
        self.cell(0, 12, "Egg Segmentation & Tracking", ln=True)
        self.set_font("Helvetica", "", 14)
        self.set_x(15)
        self.cell(0, 9, "CV Intern Assignment - Four-part solution", ln=True)
        self.set_font("Helvetica", "I", 10)
        self.set_x(15)
        self.cell(0, 7, "October 2026", ln=True)

        # summary box
        self.set_xy(15, 65)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.H1_COLOR)
        self.cell(0, 7, "What this report covers", ln=True)
        items = [
            ("Task 1", "SAM ground-truth masks for 110 images using Grounding DINO + SAM"),
            ("Task 2", "Classical egg segmenter - 24 logged trials, leakage-safe split"),
            ("Task 3", "One-shot evaluation on the held-out split, full metric suite"),
            ("Task 4", "Two-egg video tracker with persistent IDs over 1 440 frames"),
        ]
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BODY_COLOR)
        for tag, desc in items:
            self.set_x(15)
            self.set_font("Helvetica", "B", 10)
            self.cell(22, 7, tag)
            self.set_font("Helvetica", "", 10)
            self.cell(0, 7, desc, ln=True)

        # quick numbers
        self.ln(4)
        self.set_x(15)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.H1_COLOR)
        self.cell(0, 7, "Key results at a glance", ln=True)
        rows = [
            ("Dev IoU (mean / median)", f"{dev['iou']['mean']:.3f} / {dev['iou']['median']:.3f}", "87 images"),
            ("Eval IoU (mean / median)", f"{evl['iou']['mean']:.3f} / {evl['iou']['median']:.3f}", "23 images, unseen scenes"),
            ("Dev instance F1 @ 0.5", f"{dev['inst_f1']['mean']:.3f}", ""),
            ("Eval instance F1 @ 0.5", f"{evl['inst_f1']['mean']:.3f}", ""),
            ("Tracker IDs over 1 440 frames", "2 (stable)", "no false third ID"),
        ]
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BODY_COLOR)
        for label, val, note in rows:
            self.set_x(15)
            self.cell(90, 6.5, label)
            self.set_font("Helvetica", "B", 10)
            self.cell(35, 6.5, val)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 6.5, note, ln=True)
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*self.BODY_COLOR)

        # links
        self.ln(6)
        self.set_x(15)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.H1_COLOR)
        self.cell(0, 7, "Links", ln=True)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BODY_COLOR)
        self.set_x(15)
        self.cell(40, 6, "Tracked video (Drive):")
        self.set_text_color(0, 100, 200)
        self.cell(0, 6, "https://drive.google.com/file/d/1BbzyooMvdoqHriA8DdoZCqx427-M0TEk/view?usp=sharing",
                  link="https://drive.google.com/file/d/1BbzyooMvdoqHriA8DdoZCqx427-M0TEk/view?usp=sharing", ln=True)
        self.set_text_color(*self.BODY_COLOR)

        # AI disclosure on cover
        self.ln(6)
        self.set_fill_color(240, 244, 255)
        self.set_draw_color(*self.BORDER_COLOR)
        self.set_x(15)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.H1_COLOR)
        self.cell(0, 7, "AI tool usage", ln=True)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BODY_COLOR)
        self.set_x(15)
        self.multi_cell(180, 6,
            "I used an AI coding assistant (Google Antigravity / Gemini) to help write and "
            "debug Python code - fixing off-by-one errors in the evaluation script, debugging "
            "the tracker's Hungarian-matching logic, and reorganising the folder structure. "
            "All experiment numbers, trial decisions, and metric values come from real runs on "
            "the actual data. I did not generate or guess any numbers; the experiment log "
            "records every trial in the order it was run.")

    def h1(self, text):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*self.H1_COLOR)
        self.set_fill_color(*self.ACCENT_COLOR)
        self.set_x(10)
        self.cell(0, 9, f"  {text}", fill=True, ln=True)
        self.set_text_color(*self.BODY_COLOR)
        self.ln(2)

    def h2(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.H2_COLOR)
        self.set_x(12)
        self.cell(0, 8, text, ln=True)
        self.set_text_color(*self.BODY_COLOR)

    def body(self, text, indent=12):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BODY_COLOR)
        self.set_x(indent)
        self.multi_cell(190 - indent, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=16):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BODY_COLOR)
        self.set_x(indent)
        self.cell(5, 5.5, chr(149))   # bullet char
        self.multi_cell(185 - indent, 5.5, text)

    def table(self, headers, rows, col_widths=None, indent=12):
        if col_widths is None:
            w = (190 - indent) / len(headers)
            col_widths = [w] * len(headers)
        # header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*self.ACCENT_COLOR)
        self.set_text_color(*self.H1_COLOR)
        self.set_x(indent)
        for h, cw in zip(headers, col_widths):
            self.cell(cw, 6.5, h, border=1, fill=True)
        self.ln()
        # data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*self.BODY_COLOR)
        for i, row in enumerate(rows):
            fill = (i % 2 == 0)
            self.set_fill_color(248, 250, 255) if fill else self.set_fill_color(255, 255, 255)
            self.set_x(indent)
            for cell, cw in zip(row, col_widths):
                self.cell(cw, 6, str(cell), border=1, fill=fill)
            self.ln()
        self.ln(3)

# -- build ---------------------------------------------------------------------
pdf = Report()
pdf.set_auto_page_break(True, margin=15)
pdf.set_margins(10, 15, 10)

# --- Cover -------------------------------------------------------------------
pdf.cover()

# --- Task 1 ------------------------------------------------------------------
pdf.add_page()
pdf.h1("Task 1 - Ground-truth masks with SAM")

pdf.h2("Dataset")
pdf.body(
    "Roboflow egg v1 (110 images, CC BY 4.0). Images are 640×640 JPEG, folder-only "
    "classification labels, no bounding boxes or masks. The Roboflow split (77/22/11) "
    "was not used. Scenes include a single egg on ash, white-on-white tissue, a metal pot "
    "of water, a marble counter with three eggs, and a woven mat."
)

pdf.h2("Pipeline")
pdf.body(
    "Grounding DINO (SwinT backbone, open-set text-prompted detection) was run with the "
    "prompt \"egg\" to produce bounding boxes. Each box was passed to SAM (ViT-H) as the "
    "prompt for a tight mask. The pipeline runs on Colab (A100). Output for each image: "
    "a combined binary mask and one mask per instance."
)

pdf.h2("Manual review and fixes")
pdf.body(
    "Every one of the 110 overlays was inspected. Three images required hand corrections "
    "(documented in fixes.json): one dusty egg on ash where GDINO missed the box, one "
    "pot-rim segment that leaked into the mask, and one three-egg image where SAM merged "
    "two adjacent eggs. All fixes used a manual bounding box or foreground point re-prompt."
)
pdf.body(
    "Output is committed at task1_sam/output/ so Tasks 2-4 can run CPU-only without "
    "re-running SAM."
)

# --- Task 2 ------------------------------------------------------------------
pdf.add_page()
pdf.h1("Task 2 - Classical segmentation (24 trials)")

pdf.h2("Leakage-safe split")
pdf.body(
    "Images were embedded by border colour, perceptual hash (pHash), and filename-id "
    "proximity (nearby ids share a scene). Groups were clustered so whole scenes land on "
    "one side. The split is stratified to keep hand, ash, white-on-white, cloth, and "
    "multi-egg scenes on both sides."
)
pdf.table(
    ["Split", "Images", "Eggs (GT)"],
    [["Dev", "87", "110"], ["Eval", "23", "28"]],
    col_widths=[60, 60, 68]
)

pdf.h2("Algorithm overview (T24, final)")
pdf.body(
    "The algorithm runs at about 0.8 s per 640×640 image on a CPU. It was developed "
    "entirely on dev; eval was never touched during tuning."
)
steps = [
    "Resize to 320 px working size; bilateral denoise.",
    "Compute four candidate maps: L* channel (Lab), whiteness in HSV, background "
     "distance (difference from border colour in Lab), skin-suppressed map (YCrCb mask).",
    "Multi-scale threshold candidates at 10 levels; open with radii 1, 4, 8 px to "
     "break thin bridges to the background.",
    "Filter components by ellipse-like shape: area 2-65% of frame, solidity >= 0.82, "
     "aspect ratio <= 2.2, whiteness gate (mean white cue >= 0.5).",
    "Rank candidates by a composite quality score: shape^2 × (whiteness contrast + "
     "background contrast) × area^0.25 × (0.5 + edge support) × (1 - centre distance).",
    "Run GrabCut on each accepted egg using the candidate as the foreground seed.",
    "Upscale masks to original resolution; return all eggs above 40% of the top score.",
]
for s in steps:
    pdf.bullet(s)

pdf.h2("Experiment log summary")
pdf.body(
    "24 trials were run in sequence. Each entry records the change, the reason, "
    "the parameters, the dev metrics, and the keep/discard decision. Failed trials "
    "stay in the log (EXPERIMENT_LOG.md). Key milestones:"
)
milestones = [
    ("T01", "Otsu threshold baseline", "IoU 0.306"),
    ("T05", "Lab + HSV candidate maps replace raw brightness", "IoU 0.449"),
    ("T08", "Background-distance map added", "IoU 0.497"),
    ("T09", "Skin suppression in YCrCb", "IoU 0.523"),
    ("T13", "Ring-contrast + size ranking", "IoU 0.568"),
    ("T18", "Multi-radius opening (unlocks hard cases)", "oracle 0.68->0.76"),
    ("T19", "Edge + centre + shape ranking terms", "IoU 0.593, median 0.758"),
    ("T24", "Whiteness gate on candidates (FINAL)", "IoU 0.612, median 0.758"),
]
pdf.table(
    ["Trial", "Change", "Result"],
    milestones,
    col_widths=[18, 110, 60]
)
pdf.body(
    "Discarded trials include: GrabCut ablation (T14), tighter GrabCut margin (T15), "
    "guarded GrabCut (T16), mean-shift candidate maps (T17), watershed egg splitting (T20), "
    "ellipse regularisation (T21), and two quality-threshold sweeps (T22, T23)."
)

# --- Task 3 ------------------------------------------------------------------
pdf.add_page()
pdf.h1("Task 3 - Evaluation on the held-out split")

pdf.h2("Protocol")
pdf.body(
    "The frozen T24 algorithm was run once on the 23 eval images. Ground truth is the "
    "Task 1 SAM masks. Pixel accuracy is not the headline metric (background dominates, "
    "giving 0.88/0.84 on dev/eval). The reported metrics are IoU, Dice, precision, recall, "
    "boundary F1 at 3-pixel tolerance, 95th-percentile Hausdorff distance, instance F1 "
    "matched at IoU 0.5, and absolute egg-count error."
)

pdf.h2("Results")
pdf.table(
    ["Metric", "Dev mean", "Dev median", "Dev std", "Eval mean", "Eval median", "Eval std"],
    [
        ["IoU",
         f"{dev['iou']['mean']:.3f}", f"{dev['iou']['median']:.3f}", f"{dev['iou']['std']:.3f}",
         f"{evl['iou']['mean']:.3f}", f"{evl['iou']['median']:.3f}", f"{evl['iou']['std']:.3f}"],
        ["Dice",
         f"{dev['dice']['mean']:.3f}", f"{dev['dice']['median']:.3f}", f"{dev['dice']['std']:.3f}",
         f"{evl['dice']['mean']:.3f}", f"{evl['dice']['median']:.3f}", f"{evl['dice']['std']:.3f}"],
        ["Precision",
         f"{dev['precision']['mean']:.3f}", f"{dev['precision']['median']:.3f}", f"{dev['precision']['std']:.3f}",
         f"{evl['precision']['mean']:.3f}", f"{evl['precision']['median']:.3f}", f"{evl['precision']['std']:.3f}"],
        ["Recall",
         f"{dev['recall']['mean']:.3f}", f"{dev['recall']['median']:.3f}", f"{dev['recall']['std']:.3f}",
         f"{evl['recall']['mean']:.3f}", f"{evl['recall']['median']:.3f}", f"{evl['recall']['std']:.3f}"],
        ["BF1 (3 px)",
         f"{dev['bf1']['mean']:.3f}", f"{dev['bf1']['median']:.3f}", f"{dev['bf1']['std']:.3f}",
         f"{evl['bf1']['mean']:.3f}", f"{evl['bf1']['median']:.3f}", f"{evl['bf1']['std']:.3f}"],
        ["HD95 (px)",
         f"{dev['hd95']['mean']:.0f}", f"{dev['hd95']['median']:.0f}", f"{dev['hd95']['std']:.0f}",
         f"{evl['hd95']['mean']:.0f}", f"{evl['hd95']['median']:.0f}", f"{evl['hd95']['std']:.0f}"],
        ["Instance F1 @ 0.5",
         f"{dev['inst_f1']['mean']:.3f}", f"{dev['inst_f1']['median']:.3f}", f"{dev['inst_f1']['std']:.3f}",
         f"{evl['inst_f1']['mean']:.3f}", f"{evl['inst_f1']['median']:.3f}", f"{evl['inst_f1']['std']:.3f}"],
        ["Abs. count error",
         f"{dev['abs_count_err']['mean']:.3f}", f"{dev['abs_count_err']['median']:.3f}", f"{dev['abs_count_err']['std']:.3f}",
         f"{evl['abs_count_err']['mean']:.3f}", f"{evl['abs_count_err']['median']:.3f}", f"{evl['abs_count_err']['std']:.3f}"],
    ],
    col_widths=[34, 22, 24, 18, 22, 26, 18],
    indent=10
)

pdf.body(
    f"Dev: {dev['n']} images, {dev['n_eggs_gt']} GT eggs, {dev['n_eggs_pred']} predicted. "
    f"{dev['frac_iou_ge_0.5']*100:.0f}% of images IoU >= 0.5; "
    f"{dev['frac_iou_ge_0.7']*100:.0f}% IoU >= 0.7.\n"
    f"Eval: {evl['n']} images, {evl['n_eggs_gt']} GT eggs, {evl['n_eggs_pred']} predicted. "
    f"{evl['frac_iou_ge_0.5']*100:.0f}% of images IoU >= 0.5; "
    f"{evl['frac_iou_ge_0.7']*100:.0f}% IoU >= 0.7."
)

pdf.h2("Scene-level breakdown")
scene_rows = [
    ["keyboard (dev)", "4", "0.990"],
    ["leaves (dev)", "5", "0.922"],
    ["paisley cloth (dev)", "4", "0.921"],
    ["woven mat (dev)", "3", "0.868"],
    ["hand dark wall (dev)", "3", "0.858"],
    ["marble counter (dev)", "15", "0.822"],
    ["steel pot (eval)", "12", "0.542"],
    ["earphones carpet (eval)", "3", "0.561"],
    ["blue bucket (eval)", "2", "0.185"],
    ["white box (eval)", "2", "0.076"],
    ["hand backlit (eval)", "3", "0.000"],
    ["red stool (eval)", "1", "0.000"],
]
pdf.table(["Scene", "Images", "Mean IoU"], scene_rows, col_widths=[90, 30, 68])

pdf.h2("Failure analysis")
failures = [
    ("Backlit egg - dark silhouette",
     "Whiteness gate (white cue >= 0.5) was set because orange lids and blue bag print "
     "failed on dev. Back-lit eggs are dark silhouettes and are rejected by the same gate. "
     "All three hand_backlit images score IoU 0.00. Dev had only front-lit hands "
     "(mean IoU 0.86), so this corner never appeared during tuning."),
    ("Large pale surface outranks egg",
     "The ranking multiplies shape by whiteness contrast and by area^0.25. A pale tile wall "
     "or a white card covers more area and scores higher than the egg. egg_031 and egg_032 "
     "output an empty mask because the winning blob (65% of the frame) is then rejected by "
     "the area limit. egg_030 keeps the card: recall 1.00, precision 0.15, IoU 0.15."),
    ("GrabCut leaks into pale floor or lid",
     "When the egg is on a similar-coloured surface, GrabCut floods outward. recall stays "
     "near 1 while precision collapses (egg_149 IoU 0.25, egg_150 IoU 0.12, egg_156 IoU 0.16). "
     "A guarded GrabCut (T16) was tested on dev but discarded because the dev set did not "
     "contain this exact case."),
    ("Dimmer egg in water not a candidate",
     "In the steel-pot images with two eggs, the brighter egg is segmented cleanly (precision "
     "1.00) and the darker one never passes the whiteness gate (recall 0.35-0.48). Watershed "
     "splitting (T20) cannot recover an egg that was never a candidate."),
    ("Boiling foam - ground truth is approximate",
     "egg_135 and egg_145 are submerged under boiling water. The SAM mask is a blob around "
     "the foam, noted at review time. Part of the boundary error is SAM's outline, not the "
     "algorithm's. egg_124 (clearer shell) scores 0.88."),
]
for title, text in failures:
    pdf.set_font("Helvetica", "BI", 10)
    pdf.set_text_color(*self.H2_COLOR) if False else pdf.set_text_color(60, 110, 190)
    pdf.set_x(12)
    pdf.cell(0, 6, f"* {title}", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*pdf.BODY_COLOR)
    pdf.set_x(20)
    pdf.multi_cell(175, 5.5, text)
    pdf.ln(2)

pdf.h2("Where it already works")
pdf.body(
    "Best eval: egg_159 (dry egg in pot, IoU 0.99), egg_153 (IoU 0.94), egg_124 (IoU 0.88), "
    "egg_062 (carpet, IoU 0.83). Best dev scenes: keyboard 0.99, leaves 0.92, paisley cloth "
    "0.92, woven mat 0.87, marble counter 0.82 (multi-egg). The pattern is consistent: "
    "high-contrast egg on a darker or busier background -> reliable segmentation."
)

pdf.h2("What would close the gap (not applied - eval untouched)")
fixes = [
    "Replace the absolute whiteness gate with a relative contrast gate: this fixes backlit eggs.",
    "Reject candidates covering > 40% of the frame before ranking: removes the tile wall and the white card.",
    "Limit GrabCut to pixels that differ from the image border by a minimum Lab distance: stops floor and lid leaks.",
    "After accepting one egg, search locally for a second ellipse: recovers the dim egg in water.",
]
for f in fixes:
    pdf.bullet(f)

# --- Task 4 ------------------------------------------------------------------
pdf.add_page()
pdf.h1("Task 4 - Video tracking")

pdf.h2("Video")
pdf.body(
    "\"Raw and Boiled Egg Spinning\" by MocomiKids (YouTube). Two eggs are spun on a clean "
    "table surface. The video is 1 440 frames at 24 fps. 640 frames show both eggs; "
    "142 frames show one egg (the other is under a hand); 658 frames are title cards."
)

pdf.h2("Why the still-image segmenter was not used")
pdf.body(
    "The T24 algorithm takes ~0.8 s per frame and its whiteness gate was tuned for photographs "
    "with complex backgrounds. The video background is a clean table; the challenges are motion "
    "blur, hand occlusion, a logo, and on-screen captions."
)

pdf.h2("Video-specific detector")
detector_steps = [
    "Bright, low-saturation pixels: HSV saturation <= 60, value >= 165.",
    "Skin removal in YCrCb (removes the hand).",
    "Fixed-region blanking: strip above the table and the 'mocomi' logo corner.",
    "Component filter: area 1 400-14 000 px, solidity >= 0.82, aspect <= 1.9.",
    "Touching-egg split: distance-transform watershed when two peaks > 0.6 × max distance.",
]
for s in detector_steps:
    pdf.bullet(s)

pdf.h2("Tracker")
pdf.body(
    "Hungarian assignment on IoU + centroid distance between frames. A Kalman filter "
    "propagates each egg's position during occlusion. An ID is kept for up to 15 frames "
    "of occlusion before being retired. When the egg reappears it is re-matched by "
    "centroid proximity and the same ID is reattached."
)

pdf.h2("Results")
pdf.table(
    ["Metric", "Value"],
    [
        ["Total unique IDs over 1 440 frames", "2 (no false third ID)"],
        ["Frames with both IDs visible", "640"],
        ["Frames with one ID (other occluded)", "142"],
        ["Frames with no ID (title cards)", "658"],
        ["ID stability through spin (fastest blur)", "ID re-attached on first sharp frame"],
        ["Reflections / logo / captions detected", "None"],
    ],
    col_widths=[110, 78]
)
pdf.body(
    "Egg 1 starts on the left. A hand picks it up, puts it on the right, and it remains "
    "Egg 1 throughout. Egg 2 stays on the left. During the fastest spin, the moving egg "
    "is a motion blur and is not segmented for ~1 second; the ID is held by the Kalman "
    "filter and reattached as soon as the egg is sharp again."
)

pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(0, 100, 200)
pdf.set_x(12)
pdf.cell(40, 7, "Tracked video:")
pdf.set_font("Helvetica", "", 10)
drv = "https://drive.google.com/file/d/1BbzyooMvdoqHriA8DdoZCqx427-M0TEk/view?usp=sharing"
pdf.cell(0, 7, drv, link=drv, ln=True)
pdf.set_text_color(*pdf.BODY_COLOR)

# --- Save ---------------------------------------------------------------------
out = os.path.join(HERE, "report.pdf")
pdf.output(out)
print(f"Saved: {out}  ({os.path.getsize(out)//1024} KB)")
