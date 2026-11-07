"""Generate a random sample of residential addresses in a ZIP for testing DoorSight.
    python make_addresses.py 63122 100        # -> houses.csv (address column, truth columns empty)
Uses OpenStreetMap residential streets (free) + Google reverse geocoding (needs GOOGLE_MAPS_KEY; ~200 calls).
"""
import csv
import math
import os
import random
import sys
import time

import requests

from providers import _overpass, UA

KEY = os.environ.get("GOOGLE_MAPS_KEY", "")
# Kirkwood, MO bounding box (south, west, north, east); add more ZIPs here if needed
BBOX = {"63122": (38.555, -90.445, 38.615, -90.375)}


def sample_points(zip_code, n_wanted):
    s, w, n, e = BBOX[zip_code]
    q = f'[out:json][timeout:60];way["highway"="residential"]({s},{w},{n},{e});out geom;'
    ways = [el for el in _overpass(q)["elements"] if "geometry" in el]
    pts = []
    for way in ways:
        g = way["geometry"]
        for i in range(len(g) - 1):
            a, b = g[i], g[i + 1]
            seg = math.hypot((b["lat"] - a["lat"]) * 111000, (b["lon"] - a["lon"]) * 111000 * math.cos(math.radians(a["lat"])))
            steps = max(1, int(seg / 50))
            for k in range(steps):
                t = (k + 0.5) / steps
                lat, lng = a["lat"] + t * (b["lat"] - a["lat"]), a["lon"] + t * (b["lon"] - a["lon"])
                # push 18 m sideways off the road into the lots (random side)
                brg = math.degrees(math.atan2((b["lon"] - a["lon"]) * math.cos(math.radians(lat)), b["lat"] - a["lat"]))
                side = random.choice((90, -90))
                d = 18 / 111000
                pts.append((lat + d * math.cos(math.radians(brg + side)),
                            lng + d * math.sin(math.radians(brg + side)) / math.cos(math.radians(lat))))
    random.shuffle(pts)
    return pts[: n_wanted * 3]


def reverse(lat, lng):
    r = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                     params={"latlng": f"{lat},{lng}", "result_type": "street_address", "key": KEY}, timeout=20).json()
    for res in r.get("results", []):
        if res["geometry"].get("location_type") == "ROOFTOP":
            return res["formatted_address"]
    return None


def main(zip_code, n):
    if not KEY:
        sys.exit("set GOOGLE_MAPS_KEY first")
    pts = sample_points(zip_code, n)
    print(f"{len(pts)} candidate points on residential streets; reverse-geocoding...")
    seen, rows = set(), []
    for lat, lng in pts:
        addr = reverse(lat, lng)
        if addr and zip_code in addr and addr not in seen:
            seen.add(addr)
            rows.append(addr.replace(", USA", ""))
            print(f"  {len(rows):3d}  {rows[-1]}")
        if len(rows) >= n:
            break
        time.sleep(0.05)
    with open("houses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["address", "true_lat", "true_lng"])
        for a in rows:
            w.writerow([a, "", ""])
    print(f"\nwritten houses.csv with {len(rows)} addresses")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "63122", int(sys.argv[2]) if len(sys.argv) > 2 else 100)
