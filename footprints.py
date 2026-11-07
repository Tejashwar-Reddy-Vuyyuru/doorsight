#!/usr/bin/env python3
"""
Real building outlines from Microsoft's US Building Footprints (free, ODbL licence).

Step 1 (once): cut your area out of the big state file into a small local file
  python footprints.py extract Missouri.geojson
  python footprints.py extract Missouri.geojson --bbox 38.53 -90.47 38.64 -90.35   # custom box

  -> writes footprints_local.geojson next to this script (a few MB for one ZIP)

Step 2: DoorSight and regrade.py pick it up automatically (lookup() below).

  python footprints.py check 38.587735 -90.425272   # which outline does a point get?
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "footprints_local.geojson"
DEFAULT_BBOX = (38.53, -90.47, 38.64, -90.35)    # ZIP 63122 (Kirkwood / Glendale) with margin
CELL = 0.002                                      # grid cell for the lookup index (~200 m)
MAX_NEAREST_M = 15.0                              # use the nearest outline only if this close

_index = None


# ----------------------------------------------------------------------------- extract
def _rings(geom):
    """Outer rings of a Polygon / MultiPolygon as lists of (lat, lng)."""
    if not geom:
        return []
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        return []
    return [[(c[1], c[0]) for c in p[0]] for p in polys if p and p[0]]


def _in_bbox(ring, bbox):
    s, w, n, e = bbox
    return any(s <= lat <= n and w <= lng <= e for lat, lng in ring)


def _features(path):
    """Stream features from a big GeoJSON file without loading it all.
    Microsoft's files have one feature per line; falls back to a full load otherwise."""
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if len(first) > 5_000_000:                   # whole file on one line
            f.seek(0)
            for feat in json.load(f).get("features", []):
                yield feat
            return
        f.seek(0)
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{") or '"geometry"' not in line:
                continue
            try:
                feat = json.loads(line)
            except json.JSONDecodeError:
                continue
            if feat.get("type") == "Feature":
                yield feat


def extract(src, out, bbox):
    kept, seen = [], 0
    for feat in _features(src):
        seen += 1
        for ring in _rings(feat.get("geometry")):
            if _in_bbox(ring, bbox):
                kept.append({"type": "Feature", "properties": {},
                             "geometry": {"type": "Polygon",
                                          "coordinates": [[[lng, lat] for lat, lng in ring]]}})
        if seen % 250_000 == 0:
            print(f"  scanned {seen:,} buildings, kept {len(kept):,}", flush=True)
    Path(out).write_text(json.dumps({"type": "FeatureCollection", "features": kept}))
    print(f"done: scanned {seen:,}, kept {len(kept):,} outlines in {bbox} -> {out}")


# ----------------------------------------------------------------------------- lookup
def _dist_m(a, b):
    k = 111_320.0
    return math.hypot((a[0] - b[0]) * k, (a[1] - b[1]) * k * math.cos(math.radians(a[0])))


def _inside(pt, ring):
    lat, lng = pt
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        yi, xi = ring[i]
        yj, xj = ring[j]
        if (yi > lat) != (yj > lat) and lng < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _edge_dist(pt, ring):
    """Shortest distance (m) from a point to the outline's edges."""
    best = float("inf")
    lat0 = pt[0]
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 111_320.0
    px, py = pt[1] * kx, pt[0] * ky
    for i in range(len(ring)):
        (a_lat, a_lng), (b_lat, b_lng) = ring[i], ring[(i + 1) % len(ring)]
        ax, ay, bx, by = a_lng * kx, a_lat * ky, b_lng * kx, b_lat * ky
        dx, dy = bx - ax, by - ay
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def _load(path=LOCAL):
    global _index
    if _index is not None:
        return _index
    _index = {}
    if not Path(path).exists():
        return _index
    for feat in json.loads(Path(path).read_text()).get("features", []):
        for ring in _rings(feat.get("geometry")):
            if ring and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue
            cells = {(int(lat // CELL), int(lng // CELL)) for lat, lng in ring}
            for c in cells:
                _index.setdefault(c, []).append(ring)
    return _index


def available():
    return bool(_load())


def lookup(lat, lng):
    """Outline containing the point, else the nearest one within MAX_NEAREST_M. None if no local file."""
    idx = _load()
    if not idx:
        return None
    ci, cj = int(lat // CELL), int(lng // CELL)
    near = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            near += idx.get((ci + di, cj + dj), [])
    if not near:
        return None
    for ring in near:
        if _inside((lat, lng), ring):
            return list(ring)
    ring = min(near, key=lambda r: _edge_dist((lat, lng), r))
    return list(ring) if _edge_dist((lat, lng), ring) <= MAX_NEAREST_M else None


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Microsoft building outlines for DoorSight")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("src")
    e.add_argument("--out", default=str(LOCAL))
    e.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    c = sub.add_parser("check")
    c.add_argument("lat", type=float)
    c.add_argument("lng", type=float)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.src, a.out, tuple(a.bbox))
    else:
        ring = lookup(a.lat, a.lng)
        if ring is None:
            print("no outline found here" if available() else f"no {LOCAL.name} yet - run extract first")
            sys.exit(1)
        lats = [p[0] for p in ring]; lngs = [p[1] for p in ring]
        w = _dist_m((lats[0], min(lngs)), (lats[0], max(lngs))); h = _dist_m((min(lats), lngs[0]), (max(lats), lngs[0]))
        print(f"outline with {len(ring)} corners, about {w:.0f} m x {h:.0f} m, "
              f"point is {'inside' if _inside((a.lat, a.lng), ring) else 'outside'}")


if __name__ == "__main__":
    main()
