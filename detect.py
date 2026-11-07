"""
Door detector: looks at one street-level image and returns the main entrance door's
horizontal position (0..1), a confidence, and a short reason.

Backends (pick with DOORSIGHT_DETECTOR):
  local   (default) free, offline, NO training: an open-vocabulary detector finds "door",
          "window", "garage door" boxes, then simple geometry decides what is a real entrance door.
  claude  Claude vision (needs ANTHROPIC_API_KEY) — optional second opinion.

All return the same shape:
  {"visible": True, "x": 0..1, "y": 0..1, "width": 0..1, "height": 0..1, "confidence": 0..1, "reason": str}
  or {"visible": False, "reason": str}
"""
import base64
import io
import json
import os
import re

# ----------------------------------------------------------------------------- the door rules
# These are the "a door is a door" rules: no training, just shape.
DOOR_MIN_ASPECT = 1.4      # height / width  (a 6'8" x 3' door is ~2.2; a window is ~1.0)
DOOR_MAX_ASPECT = 4.0
DOOR_MIN_HEIGHT = 0.08     # fraction of image height; smaller than this = too far / not it
DOOR_BOTTOM_MIN = 0.45     # bottom edge must sit in the lower part of the frame (reaches ground/porch)
GARAGE_MAX_ASPECT = 1.3    # wide boxes are garage doors, not entrances


def _rules(box, img_w, img_h):
    """box = (x1,y1,x2,y2) in pixels -> (passes, reason, aspect)."""
    w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    aspect = h / w
    bottom = box[3] / img_h
    if aspect < DOOR_MIN_ASPECT:
        return False, f"too wide (aspect {aspect:.1f}) — window or garage", aspect
    if aspect > DOOR_MAX_ASPECT:
        return False, f"too narrow (aspect {aspect:.1f}) — column/gap", aspect
    if h / img_h < DOOR_MIN_HEIGHT:
        return False, "too small in frame — zoom in", aspect
    if bottom < DOOR_BOTTOM_MIN:
        return False, "floats mid-wall — window", aspect
    return True, f"tall ({aspect:.1f}:1), reaches ground", aspect


def _choose(boxes, img_w, img_h):
    """boxes: list of (label, score, (x1,y1,x2,y2)). Pick the best entrance door."""
    garages = [b for b in boxes if b[0] == "garage door"]
    balconies = [b for b in boxes if b[0] == "balcony"]
    best, why = None, []
    windows = [b for b in boxes if b[0] == "window"]
    cands = []
    for label, score, bx in boxes:
        if label == "door":
            # a 'door' that the model ALSO sees as a window (overlapping box, higher window score) is a window
            w_score = max((w[1] for w in windows if _iou(bx, w[2]) > 0.5), default=0.0)
            if w_score > score:
                why.append("looks more like a window")
                continue
            cands.append((score, bx))
        elif label == "window":
            # rescue: a 'window' that is clearly door-shaped (>=1.8:1) and reaches the ground is probably the door
            ok, _, aspect = _rules(bx, img_w, img_h)
            if ok and aspect >= 1.8 and bx[3] / img_h >= 0.55 and score < 0.35:   # a confident window is a window
                cands.append((score * 0.6 + 0.15, bx))
    for score, bx in cands:
        ok, reason, aspect = _rules(bx, img_w, img_h)
        if not ok:
            why.append(reason)
            continue
        # reject if mostly inside a garage box
        inside_garage = any(_iou(bx, g[2]) > 0.5 for g in garages)
        if inside_garage:
            why.append("inside garage door")
            continue
        # reject upstairs doors: a balcony railing directly under / across the door bottom
        if any(_touches_bottom(bx, b[2]) for b in balconies):
            why.append("door is on a balcony (upstairs)")
            continue
        # score: detector confidence, bonus for door-like aspect ~2.2, bonus for size
        aspect_bonus = max(0.0, 1.0 - abs(aspect - 2.2) / 2.0) * 0.2
        size_bonus = min(0.2, (bx[3] - bx[1]) / img_h)
        s = min(1.0, score + aspect_bonus + size_bonus)
        if best is None or s > best[0]:
            best = (s, bx, reason)
    # apartment signal: several confident doors in the MIDDLE of the frame (neighbours at the edges don't count)
    n_doors = sum(1 for b in boxes if b[0] == "door" and b[1] >= 0.3
                  and 0.25 * img_w <= (b[2][0] + b[2][2]) / 2 <= 0.75 * img_w)
    if best is None:
        return {"visible": False, "reason": "; ".join(why[:3]) or "no door-shaped object", "n_doors": n_doors}
    s, bx, reason = best
    return {"visible": True,
            "x": (bx[0] + bx[2]) / 2 / img_w, "y": (bx[1] + bx[3]) / 2 / img_h,
            "width": (bx[2] - bx[0]) / img_w, "height": (bx[3] - bx[1]) / img_h,
            "box": [round(v) for v in bx], "n_doors": n_doors,
            "confidence": round(float(s), 3), "reason": reason}


def all_candidates(img):
    """Every door-shaped candidate passing the basic rules, each with nearby context features.
    Returns {"doors": [cand...], "n_doors": int}; cand = {x,y,width,height,box,score,features{...}}"""
    L = _load_local()
    W, H = img.size
    boxes = _raw_boxes(img, L)
    windows = [b for b in boxes if b[0] == "window"]
    garages = [b for b in boxes if b[0] == "garage door"]
    balconies = [b for b in boxes if b[0] == "balcony"]
    out = []
    for label, score, bx in boxes:
        if label == "door":
            w_score = max((w[1] for w in windows if _iou(bx, w[2]) > 0.5), default=0.0)
            if w_score > score:
                continue
        elif label == "window":
            ok, _, aspect = _rules(bx, img_w=W, img_h=H)
            if not (ok and aspect >= 1.8 and bx[3] / H >= 0.55 and score < 0.35):
                continue
            score = score * 0.6 + 0.15
        else:
            continue
        ok, reason, aspect = _rules(bx, W, H)
        if not ok or any(_iou(bx, g[2]) > 0.5 for g in garages) or any(_touches_bottom(bx, b[2]) for b in balconies):
            continue
        feats = _context_features(bx, boxes, W, H)
        out.append({"x": (bx[0] + bx[2]) / 2 / W, "y": (bx[1] + bx[3]) / 2 / H,
                    "width": (bx[2] - bx[0]) / W, "height": (bx[3] - bx[1]) / H,
                    "box": [round(v) for v in bx], "score": round(float(score), 3), "aspect": round(aspect, 2),
                    "features": feats})
    # merge near-duplicate boxes (same door found by two prompts)
    out.sort(key=lambda c: -c["score"])
    merged = []
    for c in out:
        if all(_iou(tuple(c["box"]), tuple(m["box"])) < 0.5 for m in merged):
            merged.append(c)
    n_doors = sum(1 for c in merged if c["score"] >= 0.3 and 0.25 <= c["x"] <= 0.75)
    return {"doors": merged, "n_doors": n_doors}


def _near(a, b, W, H, dx=1.5, dy=1.0):
    """Is box b within dx door-widths horizontally and dy door-heights vertically of door box a?"""
    aw, ah = max(1, a[2] - a[0]), max(1, a[3] - a[1])
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return abs(bcx - acx) <= dx * aw + (b[2] - b[0]) / 2 and abs(bcy - acy) <= dy * ah + (b[3] - b[1]) / 2


def _context_features(door, boxes, W, H):
    f = {"steps": 0.0, "porch": 0.0, "mailbox": 0.0, "deck": 0.0, "slider": 0.0}
    for label, score, bx in boxes:
        if label == "steps" and _near(door, bx, W, H, dx=1.5, dy=1.2) and bx[3] >= door[3] - 5:   # steps below/at the door
            f["steps"] = max(f["steps"], score)
        elif label == "porch" and _near(door, bx, W, H, dx=2.0, dy=1.2):
            f["porch"] = max(f["porch"], score)
        elif label == "mailbox" and _near(door, bx, W, H, dx=3.0, dy=1.5):
            f["mailbox"] = max(f["mailbox"], score)
        elif label in ("deck", "patio") and _near(door, bx, W, H, dx=1.5, dy=1.2):
            f["deck"] = max(f["deck"], score)
        elif label == "slider" and _iou(door, bx) > 0.5:
            f["slider"] = max(f["slider"], score)
    return {k: round(v, 2) for k, v in f.items()}


def _raw_boxes(img, L):
    W, H = img.size
    boxes = []
    if L["kind"] == "owlv2":
        torch = L["torch"]
        inputs = L["proc"](text=[OWL_QUERIES], images=img, return_tensors="pt")
        with torch.no_grad():
            out = L["model"](**inputs)
        side = max(W, H)
        pp = getattr(L["proc"], "post_process_grounded_object_detection", None) or L["proc"].post_process_object_detection
        r = pp(out, threshold=0.12, target_sizes=torch.tensor([[side, side]]))[0]
        for score, lab, box in zip(r["scores"], r["labels"], r["boxes"]):
            x1, y1, x2, y2 = (float(v) for v in box)
            if x1 >= W or y1 >= H:
                continue
            boxes.append((OWL_LABELS[int(lab)], float(score), (x1, y1, min(x2, W), min(y2, H))))
    else:
        res = L["model"].predict(img, conf=0.08, verbose=False)[0]
        for b in res.boxes:
            boxes.append((res.names[int(b.cls)], float(b.conf), tuple(float(v) for v in b.xyxy[0])))
    return boxes


def _touches_bottom(door, rail):
    """Railing box overlapping the door horizontally and sitting at/below the door's lower half."""
    overlap_x = min(door[2], rail[2]) - max(door[0], rail[0])
    if overlap_x <= 0:
        return False
    door_mid = (door[1] + door[3]) / 2
    return rail[1] <= door[3] + 5 and rail[3] >= door_mid


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area_a


# ----------------------------------------------------------------------------- local, free, no training
_LOCAL = {}


OWL_QUERIES = ["a front door of a house", "a door", "a sliding glass door", "a window", "a garage door",
               "a balcony railing", "front steps leading to a door", "a porch roof over a door", "a mailbox",
               "a wooden deck", "a patio"]
OWL_LABELS = ["door", "door", "slider", "window", "garage door", "balcony", "steps", "porch", "mailbox", "deck", "patio"]


def _load_local():
    """Prefer OWLv2 (transformers; best zero-shot for 'door'), else YOLO-World (ultralytics)."""
    if _LOCAL:
        return _LOCAL
    if os.environ.get("DOORSIGHT_LOCAL", "owlv2") == "owlv2":
        try:
            import torch
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            name = "google/owlv2-base-patch16-ensemble"
            _LOCAL["kind"] = "owlv2"
            _LOCAL["proc"] = Owlv2Processor.from_pretrained(name)
            _LOCAL["model"] = Owlv2ForObjectDetection.from_pretrained(name).eval()
            _LOCAL["torch"] = torch
            return _LOCAL
        except Exception:
            _LOCAL.clear()
    try:
        from ultralytics import YOLOWorld
        m = YOLOWorld("yolov8s-worldv2.pt")          # downloads once (~50 MB)
        m.set_classes(["door", "window", "garage door"])
        _LOCAL["kind"], _LOCAL["model"] = "yoloworld", m
        return _LOCAL
    except Exception:
        pass
    try:
        import torch
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        name = "google/owlvit-base-patch32"
        _LOCAL["kind"] = "owlvit"
        _LOCAL["proc"] = OwlViTProcessor.from_pretrained(name)
        _LOCAL["model"] = OwlViTForObjectDetection.from_pretrained(name).eval()
        _LOCAL["torch"] = torch
        return _LOCAL
    except Exception as e:
        raise RuntimeError("no local detector available: pip install ultralytics  (or transformers torch)") from e


def detect_local(img):
    L = _load_local()
    W, H = img.size
    boxes = []
    if L["kind"] == "owlv2":
        torch = L["torch"]
        inputs = L["proc"](text=[OWL_QUERIES], images=img, return_tensors="pt")
        with torch.no_grad():
            out = L["model"](**inputs)
        # OWLv2 pads the image to a square; results are relative to the padded size
        side = max(W, H)
        ts = torch.tensor([[side, side]])
        pp = getattr(L["proc"], "post_process_grounded_object_detection", None) or L["proc"].post_process_object_detection
        r = pp(out, threshold=0.12, target_sizes=ts)[0]
        for score, lab, box in zip(r["scores"], r["labels"], r["boxes"]):
            x1, y1, x2, y2 = (float(v) for v in box)
            if x1 >= W or y1 >= H:
                continue
            boxes.append((OWL_LABELS[int(lab)], float(score), (x1, y1, min(x2, W), min(y2, H))))
    elif L["kind"] == "yoloworld":
        res = L["model"].predict(img, conf=0.08, verbose=False)[0]
        names = res.names
        for b in res.boxes:
            boxes.append((names[int(b.cls)], float(b.conf), tuple(float(v) for v in b.xyxy[0])))
    else:
        torch = L["torch"]
        texts = [["a photo of a front door", "a photo of a window", "a photo of a garage door"]]
        inputs = L["proc"](text=texts, images=img, return_tensors="pt")
        with torch.no_grad():
            out = L["model"](**inputs)
        target = torch.tensor([[H, W]])
        r = L["proc"].post_process_object_detection(out, threshold=0.1, target_sizes=target)[0]
        labels = ["door", "window", "garage door"]
        for score, lab, box in zip(r["scores"], r["labels"], r["boxes"]):
            boxes.append((labels[int(lab)], float(score), tuple(float(v) for v in box)))
    return _choose(boxes, W, H)


# ----------------------------------------------------------------------------- optional: Claude vision
PROMPT = """Locate the MAIN ENTRANCE DOOR of the building in this street-level photo (the door a delivery driver walks to).
Only report a door you can actually SEE: tall rectangle (about 2-3x taller than wide), reaching the ground or porch floor, with a frame. Not a window (squarer, floats mid-wall), not a garage door (wide), not a gate.
Return ONLY JSON: {"visible": true/false, "x": <0..1 horizontal centre>, "y": <0..1 vertical centre>, "width": <0..1>, "height": <0..1>, "confidence": <0..1>, "reason": "<short>"}"""


def detect_claude(img, model=None):
    import anthropic
    client = anthropic.Anthropic()
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
    msg = client.messages.create(
        model=model or os.environ.get("DOORSIGHT_MODEL", "claude-sonnet-4-5"), max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": PROMPT}]}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not d.get("visible"):
        return {"visible": False, "reason": d.get("reason", "")}
    return {"visible": True, "x": float(d["x"]), "y": float(d.get("y", 0.5)), "width": float(d.get("width", 0.05)),
            "height": float(d.get("height", 0.2)), "confidence": float(d.get("confidence", 0.5)),
            "reason": d.get("reason", "")}


def detect_boxes(img, queries, labels, threshold=0.15):
    """Generic zero-shot boxes with OWLv2 (any prompts). Returns [(label, score, (x1,y1,x2,y2))]."""
    L = _load_local()
    if L["kind"] != "owlv2":
        return []
    torch = L["torch"]
    W, H = img.size
    inputs = L["proc"](text=[queries], images=img, return_tensors="pt")
    with torch.no_grad():
        out = L["model"](**inputs)
    side = max(W, H)
    pp = getattr(L["proc"], "post_process_grounded_object_detection", None) or L["proc"].post_process_object_detection
    r = pp(out, threshold=threshold, target_sizes=torch.tensor([[side, side]]))[0]
    boxes = []
    for score, lab, box in zip(r["scores"], r["labels"], r["boxes"]):
        x1, y1, x2, y2 = (float(v) for v in box)
        if x1 >= W or y1 >= H:
            continue
        boxes.append((labels[int(lab)], float(score), (x1, y1, min(x2, W), min(y2, H))))
    return boxes


# ----------------------------------------------------------------------------- house numbers (optional)
_OCR = {}


def read_house_numbers(img, expect_len=None):
    """Return the list of 2-5 digit numbers visible in the image (needs `pip install easyocr`); [] if unavailable.
    The bottom strip (Google watermark '© 2026 Google') is cropped off and year-like numbers are ignored."""
    if "r" not in _OCR:
        try:
            import easyocr
            _OCR["r"] = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            _OCR["r"] = None
    if _OCR["r"] is None:
        return []
    import numpy as np
    W, H = img.size
    crop = img.crop((0, 0, W, int(H * 0.90)))          # drop the watermark strip
    try:
        res = _OCR["r"].readtext(np.array(crop), allowlist="0123456789", detail=1)
    except Exception:
        return []
    nums = []
    for box, text, conf in res:
        t = re.sub(r"\D", "", text)
        if not (2 <= len(t) <= 5 and conf >= 0.5):
            continue
        if re.match(r"^(19|20)\d\d\d?$", t):            # a year (2026, 20266 ...) is never a house number
            continue
        if expect_len and abs(len(t) - expect_len) > 1:
            continue
        xs = [pt[0] for pt in box]; ys = [pt[1] for pt in box]
        nums.append({"text": t, "box": (min(xs), min(ys), max(xs), max(ys))})
    return nums


BACKENDS = {"local": detect_local, "claude": detect_claude}


def detect(img, backend=None):
    backend = backend or os.environ.get("DOORSIGHT_DETECTOR", "local")
    return BACKENDS[backend](img)
