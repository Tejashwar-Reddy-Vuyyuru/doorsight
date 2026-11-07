"""
Aerial eye — free satellite tiles (Esri World Imagery, no key) + zero-shot detection of
walkway / porch / driveway, then: which side of the footprint does the path touch?

Returns {"side_bearing": deg, "side": "N|NE|...", "confidence": 0..1, "evidence": "aerial.jpg",
         "entry_point": (lat,lng)} or None.
"""
import io
import math

import requests
from PIL import Image, ImageDraw

import geometry as G

TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
Z = 20  # ~0.15 m/px at mid-latitudes
UA = {"User-Agent": "DoorSight/0.1 (research prototype)"}


def _tile_xy(lat, lng, z):
    n = 2 ** z
    x = (lng + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def _xy_latlng(x, y, z):
    n = 2 ** z
    lng = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lng


def fetch_aerial(lat, lng, size_tiles=3):
    """3x3 tile mosaic centred on the point. Returns (PIL.Image, (x0,y0) tile origin)."""
    cx, cy = _tile_xy(lat, lng, Z)
    x0, y0 = int(cx) - size_tiles // 2, int(cy) - size_tiles // 2
    mosaic = Image.new("RGB", (256 * size_tiles, 256 * size_tiles))
    for dy in range(size_tiles):
        for dx in range(size_tiles):
            r = requests.get(TILE_URL.format(z=Z, x=x0 + dx, y=y0 + dy), headers=UA, timeout=30)
            if r.ok:
                mosaic.paste(Image.open(io.BytesIO(r.content)).convert("RGB"), (dx * 256, dy * 256))
    return mosaic, (x0, y0)


def _to_px(lat, lng, origin):
    x, y = _tile_xy(lat, lng, Z)
    return (x - origin[0]) * 256, (y - origin[1]) * 256


def _to_ll(px, py, origin):
    return _xy_latlng(origin[0] + px / 256, origin[1] + py / 256, Z)


AERIAL_QUERIES = ["a paved footpath leading to a house entrance seen from above",
                  "a front porch seen from above", "a driveway seen from above", "front steps seen from above"]
AERIAL_LABELS = ["walkway", "front porch", "driveway", "steps"]


def _boxes(img):
    """[(label, score, (x1,y1,x2,y2))] from OWLv2; falls back to YOLO-World if OWLv2 is unavailable."""
    from detect import detect_boxes
    out = detect_boxes(img, AERIAL_QUERIES, AERIAL_LABELS, threshold=0.12)
    if out:
        return out
    from ultralytics import YOLOWorld
    m = YOLOWorld("yolov8s-worldv2.pt")
    m.set_classes(AERIAL_LABELS)
    res = m.predict(img, conf=0.1, verbose=False)[0]
    return [(res.names[int(b.cls)], float(b.conf), tuple(float(v) for v in b.xyxy[0])) for b in res.boxes]


def analyze(lat, lng, footprint, outdir):
    img, origin = fetch_aerial(lat, lng)
    poly_px = [_to_px(p[0], p[1], origin) for p in footprint]
    c = G.centroid(footprint)
    cpx = _to_px(c[0], c[1], origin)
    try:
        boxes = _boxes(img)
    except Exception:
        return None
    best = None
    endpoints = []   # where walkways / driveways meet the building: [{"kind","lat","lng","score"}]
    dr = ImageDraw.Draw(img)
    dr.polygon(poly_px, outline=(255, 255, 0), width=3)
    for label, score, (x1, y1, x2, y2) in boxes:
        if (x2 - x1) > 200 or (y2 - y1) > 200:      # > ~30 m: not a walkway/porch, skip
            continue
        if label in ("walkway", "driveway", "steps", "front porch") and score >= 0.12:
            # endpoint = corner/edge point of the box closest to the building centre
            ex = min(max(cpx[0], x1), x2); ey = min(max(cpx[1], y1), y2)
            el = _to_ll(ex, ey, origin)
            if G.dist_m(el, c) <= 20:
                endpoints.append({"kind": label, "lat": el[0], "lng": el[1], "score": round(score, 2)})
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        # feature must be close to the building (within ~12 m = 80 px) but not centred inside it
        d_px = math.hypot(bx - cpx[0], by - cpx[1])
        if d_px > 120 or G.point_in_polygon(_to_ll(bx, by, origin), footprint) and label != "front porch":
            continue
        w = {"front porch": 1.0, "steps": 0.9, "walkway": 0.8, "driveway": 0.4}[label]
        s = score * w
        dr.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        dr.text((x1, y1 - 12), f"{label} {score:.2f}", fill=(0, 255, 0))
        if best is None or s > best[0]:
            best = (s, label, (bx, by))
    path = outdir / "aerial.jpg"
    img.save(path, quality=85)
    if best is None:
        return {"side_bearing": None, "side": None, "confidence": 0.0, "evidence": path.name, "entry_point": None,
                "endpoints": endpoints}
    s, label, (bx, by) = best
    fl = _to_ll(bx, by, origin)
    bearing = G.bearing_to(c[0], c[1], fl[0], fl[1])
    # entry point = where the ray from centroid toward the feature exits the footprint
    hit = G.ray_polygon_intersection(c, bearing, footprint)
    entry = (hit[0], hit[1]) if hit else None
    return {"side_bearing": round(bearing, 1), "side": compass(bearing), "confidence": round(min(1.0, s), 2),
            "feature": label, "evidence": path.name, "entry_point": entry, "endpoints": endpoints}


def compass(b):
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((b + 22.5) % 360) // 45)]


def side_agrees(door_pt, footprint, side_bearing, tol=60):
    """Does a street-level door point lie on the aerial-predicted side of the building?"""
    if side_bearing is None or door_pt is None:
        return None
    c = G.centroid(footprint)
    b = G.bearing_to(c[0], c[1], door_pt[0], door_pt[1])
    diff = abs(((b - side_bearing + 180) % 360) - 180)
    return diff <= tol
