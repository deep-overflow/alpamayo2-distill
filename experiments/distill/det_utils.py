"""Detection prompts, box parsing and IoU matching shared by the G0d probe, its analysis and
the labeling run. No GPU / model imports, so analysis scripts can use it anywhere."""

import ast
import json
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

PROMPTS = {
    "p1_qwen_json": ("Detect all vehicles, pedestrians and cyclists in the Front camera image at "
                     "frame 3. Return every object as JSON: "
                     '[{"bbox_2d": [x1, y1, x2, y2], "label": "<class>"}].'),
    "p2_release_style": ("Find all vehicles, pedestrians and cyclists visible in the Front camera "
                         "at the current frame and return their bounding boxes in JSON."),
    "p3_class_list": ("Locate every car, truck, bus, pedestrian and cyclist in the Front camera, "
                      "frame 3. Output a JSON list with one entry per object, each with "
                      '"bbox_2d" and "label".'),
}
BOX_RE = re.compile(r'"(?:bbox_2d|bbox|box)"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)'
                    r'\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
                    r'(?:\s*,\s*"label"\s*:\s*"([^"]*)")?')
MODEL_W, MODEL_H = 576, 320
DIST_BINS = ((0, 30), (30, 60), (60, 1e9))


def parse_boxes(text):
    """-> list of (x1, y1, x2, y2, label) as written by the model; JSON first, regex fallback."""
    s = text.strip()
    start = min([i for i in (s.find("["), s.find("{")) if i >= 0], default=-1)
    parsed = None
    if start >= 0:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(s[start:])
                break
            except (ValueError, SyntaxError):
                continue
    out = []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if isinstance(parsed, list):
        for it in parsed:
            if not isinstance(it, dict):
                continue
            b = next((it[k] for k in ("bbox_2d", "bbox", "box") if k in it), None)
            if isinstance(b, (list, tuple)) and len(b) == 4:
                try:
                    out.append((*[float(v) for v in b], str(it.get("label", ""))))
                except (TypeError, ValueError):
                    continue
        if out:
            return out, "json"
    out = [(float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)),
            m.group(5) or "") for m in BOX_RE.finditer(s)]
    return out, ("regex" if out else "none")


def to_unit1000(box, mode):
    """Model-written box -> [0, 1000] of width/height, sorted and clipped."""
    x1, y1, x2, y2 = box[:4]
    if mode == "pixel":
        x1, x2 = x1 / MODEL_W * 1000, x2 / MODEL_W * 1000
        y1, y2 = y1 / MODEL_H * 1000, y2 / MODEL_H * 1000
    x1, x2 = sorted((np.clip(x1, 0, 1000), np.clip(x2, 0, 1000)))
    y1, y2 = sorted((np.clip(y1, 0, 1000), np.clip(y2, 0, 1000)))
    return [x1, y1, x2, y2]


def iou_matrix(a, b):
    """(n, 4), (m, 4) -> (n, m) IoU, boxes in the same units."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def match(pred, ref, ignore, thr):
    """-> (tp, fp, matched ref mask) for one image; preds on ignore boxes are not FPs."""
    m = iou_matrix(pred, ref)
    matched_ref = np.zeros(len(ref), bool)
    matched_pred = np.zeros(len(pred), bool)
    if m.size:
        r, c = linear_sum_assignment(-m)
        ok = m[r, c] >= thr
        matched_pred[r[ok]] = True
        matched_ref[c[ok]] = True
    ign = np.zeros(len(pred), bool)
    if len(ignore) and len(pred):
        ign = (iou_matrix(pred, ignore).max(1) >= 0.3) & ~matched_pred
    return int(matched_pred.sum()), int((~matched_pred & ~ign).sum()), matched_ref
