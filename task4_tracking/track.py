"""Task 4: segment and track the two eggs in the spinning-egg video.

The still-image algorithm (T24) is the wrong tool here: it takes ~0.8 s/frame and its whiteness
gate was tuned for photographs, not a dark table with hands, reflections and title cards. This
file keeps the same ideas (white, egg-shaped, not skin) in a form that runs at video rate, then
gives each egg a persistent id.

Detection, per frame:
  bright + low-saturation pixels, with skin removed and the corner logo blanked
  keep components that fit an ellipse (solidity, aspect, area)
  if two eggs touch, split the blob where the distance transform has two peaks

Tracking:
  the clip only ever has two eggs, so there are at most two ids
  a Kalman filter predicts the centroid; Hungarian matching uses distance and mask IoU
  a hand can hide an egg for several seconds, and a spun egg can reappear far away,
  so an unmatched detection is given back to the hidden id instead of starting a third one
  while the egg is hidden the id stays at the last place it was seen, and the marker is
  drawn for about a second; the id itself is kept until the egg returns or a title card starts

  python task4_tracking/track.py
  python task4_tracking/track.py --video task4_tracking/input/eggs.mp4 --out task4_tracking/output/tracked.mp4
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent

# detection
MIN_AREA, MAX_AREA = 1400, 14000
MIN_SOLIDITY, MAX_ASPECT = 0.82, 1.9
V_MIN, S_MAX = 165, 60
LOGO_BOX = (0.72, 0.82, 1.0, 1.0)   # x0, y0, x1, y1 as fractions: the "mocomi" bug
TABLE_TOP = 0.34                     # fraction of height; the person is above the table

# tracking
MAX_MATCH_DIST = 90     # px; a normal frame-to-frame match
MAX_AGE = 360           # a hidden egg keeps its id for up to 15 s
EMPTY_RESET = 24        # frames with no egg at all (a title card) before ids are cleared
MIN_HITS = 3            # a new id is drawn only after it is seen this often
COAST_DRAW = 36         # frames the held-id marker stays on screen (~1.5 s); the id is kept longer
FULL_AREA = 3000        # a complete egg; smaller blobs are fingers or a sliver of shell


def skin_mask(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    return (cr >= 138) & (cr <= 180) & (cb >= 80) & (cb <= 130)


def shape_of(comp):
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = float(comp.sum())
    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = area / hull if hull > 0 else 0
    if len(c) >= 5:
        (cx, cy), (a, b), _ = cv2.fitEllipse(c)
        aspect = max(a, b) / max(min(a, b), 1e-6)
    else:
        ys, xs = np.nonzero(comp)
        cx, cy = float(xs.mean()), float(ys.mean())
        aspect = 99.0
    return dict(area=area, solidity=solidity, aspect=aspect, centroid=(cx, cy))


def split_if_two(comp):
    """Return one or two masks. A touching pair has two peaks in the distance transform."""
    dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = (dist >= 0.55 * dist.max()).astype(np.uint8)
    n, markers = cv2.connectedComponents(peaks)
    if n - 1 < 2:
        return [comp]
    markers = markers.astype(np.int32)
    ws = markers.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ws[~cv2.dilate(comp.astype(np.uint8), k).astype(bool)] = n
    relief = cv2.cvtColor((255 - cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8),
                          cv2.COLOR_GRAY2BGR)
    cv2.watershed(relief, ws)
    parts = []
    for i in range(1, n):
        part = (ws == i) & comp
        if part.sum() >= MIN_AREA * 0.6:
            parts.append(part)
    return parts or [comp]


def detect(bgr):
    """List of dicts: mask (full res), centroid (x, y), bbox (x, y, w, h), area."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] <= S_MAX) & (hsv[..., 2] >= V_MIN) & ~skin_mask(bgr)
    m[:int(TABLE_TOP * h), :] = False
    x0, y0, x1, y1 = LOGO_BOX
    m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k).astype(bool)

    out = []
    n, lab = cv2.connectedComponents(m.astype(np.uint8))
    for i in range(1, n):
        comp = lab == i
        if not MIN_AREA <= comp.sum() <= MAX_AREA * 2.2:
            continue
        for part in split_if_two(comp):
            if not MIN_AREA <= part.sum() <= MAX_AREA:
                continue
            s = shape_of(part)
            if s["solidity"] < MIN_SOLIDITY or s["aspect"] > MAX_ASPECT:
                continue
            ys, xs = np.nonzero(part)
            x, y = int(xs.min()), int(ys.min())
            out.append({"mask": part, "centroid": s["centroid"], "area": s["area"],
                        "bbox": (x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1)})
    return out


# ---------------------------------------------------------------- tracking

class Track:
    _next = 1

    def __init__(self, det):
        self.id = Track._next
        Track._next += 1
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        cx, cy = det["centroid"]
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.mask = det["mask"]
        self.bbox = det["bbox"]
        self.area = det["area"]
        self.hits = 1
        self.age = 0          # frames since last match
        self.seen = 1
        self.pred = (cx, cy)

    def predict(self):
        p = self.kf.predict()
        p = np.asarray(p).reshape(-1)
        self.pred = (float(p[0]), float(p[1]))
        return self.pred

    def update(self, det):
        cx, cy = det["centroid"]
        self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        self.mask, self.bbox, self.area = det["mask"], det["bbox"], det["area"]
        self.hits += 1
        self.seen += 1
        self.age = 0

    def centroid(self):
        s = np.asarray(self.kf.statePost).reshape(-1)
        return float(s[0]), float(s[1])

    def hold(self):
        """An occluded egg stays where it was last seen. Open velocity would walk the id off the table."""
        if self.age == 0:
            s = self.kf.statePost
            s[2, 0] = 0
            s[3, 0] = 0
            self.pred = (float(s[0, 0]), float(s[1, 0]))
        self.age += 1


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return inter / np.logical_or(a, b).sum()


def associate(tracks, dets):
    """Returns (matches as (track_idx, det_idx), unmatched track idxs, unmatched det idxs)."""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    cost = np.full((len(tracks), len(dets)), 1e3, np.float32)
    for i, t in enumerate(tracks):
        px, py = t.predict()
        for j, d in enumerate(dets):
            cx, cy = d["centroid"]
            dist = np.hypot(cx - px, cy - py)
            iou = _iou(t.mask, d["mask"])
            if dist > MAX_MATCH_DIST and iou < 0.05:
                continue
            cost[i, j] = dist / MAX_MATCH_DIST + (1 - iou)
    rows, cols = linear_sum_assignment(cost)
    matched_t, matched_d, pairs = set(), set(), []
    for r, c in zip(rows, cols):
        if cost[r, c] >= 1e3:
            continue
        pairs.append((r, c))
        matched_t.add(r)
        matched_d.add(c)
    return pairs, [i for i in range(len(tracks)) if i not in matched_t], [j for j in range(len(dets)) if j not in matched_d]


class Tracker:
    """At most two ids. This video only ever shows two eggs, and a hand hides one of them for
    several seconds, so a detection that reappears far from its last spot is given back to the
    hidden id instead of starting a third one."""

    def __init__(self):
        self.tracks = []
        self.empty = 0

    def step(self, dets):
        if not dets:
            self.empty += 1
            if self.empty >= EMPTY_RESET:
                self.tracks = []
            else:
                for t in self.tracks:
                    t.hold()
            return [t for t in self.tracks if t.hits >= MIN_HITS and t.age <= COAST_DRAW]

        self.empty = 0
        if any(d["area"] >= FULL_AREA for d in dets):
            dets = [d for d in dets if d["area"] >= FULL_AREA * 0.8]

        pairs, miss_t, miss_d = associate(self.tracks, dets)
        for ti, di in pairs:
            self.tracks[ti].update(dets[di])
        # a hidden egg can reappear well outside the frame-to-frame gate (it was spun)
        for di in list(miss_d):
            if not miss_t:
                break
            cx, cy = dets[di]["centroid"]
            ti = min(miss_t, key=lambda i: np.hypot(cx - self.tracks[i].pred[0], cy - self.tracks[i].pred[1]))
            # a sliver of shell between fingers must not drag the id across the table
            if dets[di]["area"] < 0.55 * self.tracks[ti].area:
                continue
            self.tracks[ti].update(dets[di])
            miss_t.remove(ti)
            miss_d.remove(di)
        for ti in miss_t:
            self.tracks[ti].hold()
        for di in miss_d:
            if len(self.tracks) < 2:
                self.tracks.append(Track(dets[di]))
        self.tracks = [t for t in self.tracks if t.age <= MAX_AGE]
        return [t for t in self.tracks if t.hits >= MIN_HITS and t.age <= COAST_DRAW]


COLORS = [(40, 220, 40), (40, 40, 230), (230, 160, 40), (40, 200, 230), (230, 40, 230)]


def draw(bgr, shown):
    vis = bgr.copy()
    for t in shown:
        c = COLORS[(t.id - 1) % len(COLORS)]
        if t.age == 0:
            vis[t.mask] = (0.55 * vis[t.mask] + 0.45 * np.array(c)).astype(np.uint8)
            cnts, _ = cv2.findContours(t.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, c, 2)
            x, y, w, h = t.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), c, 1)
            label_at = (x, max(14, y - 6))
        else:
            # egg hidden (usually by the hand): keep the id at the predicted spot
            cx, cy = int(t.pred[0]), int(t.pred[1])
            cv2.circle(vis, (cx, cy), 22, c, 2)
            label_at = (cx - 24, max(14, cy - 28))
        cv2.putText(vis, f"egg {t.id}", label_at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, f"egg {t.id}", label_at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)
    return vis


def is_title_card(frame):
    """The intro and the 'why this happens' card are green. The table is brown."""
    bot = frame[int(frame.shape[0] * 0.62):]
    hsv = cv2.cvtColor(bot, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 60)
    return float(green.mean()) > 0.5


def run(video, out_path, preview_dir=None):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tracker = Tracker()
    counts = []
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if is_title_card(frame):
            tracker.tracks = []
            tracker.empty = 0
            shown = []
        else:
            shown = tracker.step(detect(frame))
        vis = draw(frame, shown)
        writer.write(vis)
        counts.append(len(shown))
        if preview_dir and frame_i % 120 == 0:
            cv2.imwrite(str(preview_dir / f"out_{frame_i:04d}.jpg"), vis)
        frame_i += 1
    cap.release()
    writer.release()
    counts = np.array(counts)
    summary = {
        "frames": int(frame_i), "fps": fps, "size": [w, h],
        "ids_created": Track._next - 1,
        "frames_with_0": int((counts == 0).sum()),
        "frames_with_1": int((counts == 1).sum()),
        "frames_with_2": int((counts == 2).sum()),
        "frames_with_more": int((counts > 2).sum()),
    }
    print(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(HERE / "input" / "eggs.mp4"))
    ap.add_argument("--out", default=str(HERE / "output" / "tracked.mp4"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    preview = HERE / "frames"
    preview.mkdir(exist_ok=True)
    Track._next = 1
    run(args.video, out, preview)
    encode_h264(out, Path(args.video))


def encode_h264(path, audio_from):
    """mp4v plays poorly in browsers. Re-encode and keep the original soundtrack."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = [ffmpeg, "-y", "-i", str(path), "-i", str(audio_from),
           "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and tmp.exists():
        tmp.replace(path)
        print("wrote", path)
    else:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
