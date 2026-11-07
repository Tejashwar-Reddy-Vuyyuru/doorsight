#!/usr/bin/env python3
"""
ZIP census — count the REAL hard cases in a ZIP before (or after) running DoorSight.

  python census.py 63122                 # counts + writes census_63122.csv (one row per building)
  python doorsight.py batch census_63122.csv --tier hard   # (or just feed the csv to the app)

Pulls every building footprint and road in the ZIP from OpenStreetMap (free) and labels each building:
  corner      - within 25 m of two differently-named roads  (door could face either street)
  multiunit   - building=apartments/residential-large (>600 m2) / terrace  (doors face lots/courtyards)
  setback     - nearest road > 35 m away  (flag lot, long driveway, likely hidden door)
  easy        - none of the above
"""
import csv
import math
import sys

import requests

import geometry as G

UA = {"User-Agent": "DoorSight/0.1 (research prototype)"}


def overpass(q):
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, headers=UA, timeout=180)
    r.raise_for_status()
    return r.json()["elements"]


def fetch(zip_code):
    q = f"""[out:json][timeout:180];
area["postal_code"="{zip_code}"]->.z;
(
  way["building"](area.z);
  way["highway"~"^(residential|tertiary|secondary|primary|unclassified|living_street|service)$"](area.z);
);
out tags geom;"""
    els = overpass(q)
    bld, roads = [], []
    for e in els:
        if "geometry" not in e:
            continue
        pts = [(g["lat"], g["lon"]) for g in e["geometry"]]
        t = e.get("tags", {})
        if "building" in t:
            if len(pts) >= 4:
                bld.append({"id": e["id"], "poly": pts[:-1] if pts[0] == pts[-1] else pts, "tags": t})
        elif "highway" in t:
            roads.append({"name": t.get("name", f"unnamed-{e['id']}"), "pts": pts})
    return bld, roads


def area_m2(poly):
    c = G.centroid(poly)
    xy = [G.local_xy(p[0], p[1], c[0], c[1]) for p in poly]
    s = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]; x2, y2 = xy[(i + 1) % len(xy)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def seg_dist(p, a, b):
    """metres from point p to segment a-b (all lat/lng)."""
    ax, ay = G.local_xy(a[0], a[1], p[0], p[1]); bx, by = G.local_xy(b[0], b[1], p[0], p[1])
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0 if L2 == 0 else max(0, min(1, -(ax * dx + ay * dy) / L2))
    return math.hypot(ax + t * dx, ay + t * dy)


def nearby_roads(c, roads, within):
    """names of distinct roads within `within` metres of centroid c, plus nearest distance."""
    names, nearest = set(), 1e9
    for r in roads:
        pts = r["pts"]
        # quick reject: centroid outside the road's bounding box (+ margin)
        m = (within + 50) / 111000.0
        if not (min(p[0] for p in pts) - m <= c[0] <= max(p[0] for p in pts) + m and
                min(p[1] for p in pts) - m * 1.3 <= c[1] <= max(p[1] for p in pts) + m * 1.3):
            continue
        d = min(seg_dist(c, pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        nearest = min(nearest, d)
        if d <= within:
            names.add(r["name"])
    return names, nearest


def classify(b, roads):
    t, poly = b["tags"], b["poly"]
    c = G.centroid(poly)
    a = area_m2(poly)
    names, nearest = nearby_roads(c, roads, within=25)
    multi = t.get("building") in {"apartments", "terrace", "dormitory"} or (t.get("building") in {"residential", "yes"} and a > 600)
    if multi:
        kind = "multiunit"
    elif len(names) >= 2:
        kind = "corner"
    elif nearest > 35:
        kind = "setback"
    else:
        kind = "easy"
    addr = " ".join(x for x in (t.get("addr:housenumber"), t.get("addr:street")) if x)
    return {"osm_id": b["id"], "kind": kind, "lat": round(c[0], 7), "lng": round(c[1], 7),
            "area_m2": round(a), "roads_near": len(names), "nearest_road_m": round(nearest, 1),
            "building": t.get("building", ""), "address": addr}


def main(zip_code):
    print(f"fetching OSM buildings + roads for ZIP {zip_code} ...")
    bld, roads = fetch(zip_code)
    print(f"{len(bld)} buildings, {len(roads)} road segments")
    rows = [classify(b, roads) for b in bld]
    counts = {}
    for r in rows:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    print("\n=== Real counts in ZIP", zip_code, "===")
    for k in ("easy", "corner", "setback", "multiunit"):
        print(f"{k:10}{counts.get(k, 0):>7}")
    with_addr = sum(1 for r in rows if r["address"])
    print(f"\n{with_addr} buildings carry an address in OSM (the rest need a geocode join)")
    out = f"census_{zip_code}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"written {out}  -> filter by kind and feed to doorsight batch / the app")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "63122")
