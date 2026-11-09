#!/usr/bin/env python3
"""Compare Google's entrance data with DoorSight's verified pins.

    python google_entrances.py                 # uses results/high_checklist_judged.csv
    python google_entrances.py --file other.csv

For each shipped DoorSight pin, asks Google's Geocoding API v4 (SearchDestinations)
for the address's entrances, then writes results/google_entrances.csv for hand-checking:
did Google return an entrance, how far is it from DoorSight's verified pin, and a map link.
Needs GOOGLE_MAPS_KEY with the Geocoding API enabled. About one call per address.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

URL = "https://geocode.googleapis.com/v4/geocode/destinations"
FIELDS = "destinations.primary,destinations.entrances"
RESULTS = Path(__file__).resolve().parent / "results"


def dist_m(a, b):
    lat = math.radians((a[0] + b[0]) / 2)
    dy = (a[0] - b[0]) * 111_320
    dx = (a[1] - b[1]) * 111_320 * math.cos(lat)
    return math.hypot(dx, dy)


def ask_google(address, key):
    r = requests.post(
        URL,
        headers={"Content-Type": "application/json", "X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELDS},
        json={"addressQuery": {"addressQuery": address}},
        timeout=30,
    )
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    return r.json(), None


def pick_entrance(data, near):
    """Return (best entrance (lat, lng), number of entrances, preferred?) or (None, 0, False)."""
    ents = []
    for d in (data or {}).get("destinations", []):
        for e in d.get("entrances", []) or []:
            loc = e.get("location") or {}
            if "latitude" not in loc:
                continue
            tags = e.get("tags") or e.get("entranceTags") or []
            ents.append(((loc["latitude"], loc["longitude"]), "PREFERRED" in tags))
    if not ents:
        return None, 0, False
    preferred = [e for e in ents if e[1]] or ents
    # if several, take the preferred one closest to DoorSight's pin (most favourable to Google)
    best = min(preferred, key=lambda e: dist_m(e[0], near))
    return best[0], len(ents), best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RESULTS / "high_checklist_judged.csv"))
    args = ap.parse_args()

    key = os.environ.get("GOOGLE_MAPS_KEY", "")
    if not key:
        sys.exit("Set GOOGLE_MAPS_KEY first.")

    rows = list(csv.DictReader(open(args.file, encoding="utf-8-sig")))
    print(f"{len(rows)} DoorSight pins from {args.file}")

    out, raw = [], {}
    for i, r in enumerate(rows, 1):
        addr = r["address"]
        ds = (float(r["lat"]), float(r["lng"]))
        data, err = ask_google(addr, key)
        raw[addr] = data if data is not None else {"error": err}
        if err:
            print(f"[{i}/{len(rows)}] {addr}: ERROR {err}")
            if i == 1:
                sys.exit("First call failed. Check that the Geocoding API is enabled for this key.")
            g, n, pref = None, 0, False
        else:
            g, n, pref = pick_entrance(data, ds)
        d = round(dist_m(g, ds), 1) if g else ""
        print(f"[{i}/{len(rows)}] {addr}: " + (f"{n} entrance(s), {d} m from DoorSight" if g else "no entrance"))
        out.append({
            "address": addr,
            "doorsight_lat": ds[0], "doorsight_lng": ds[1],
            "google_entrances": n,
            "google_preferred": "yes" if pref else ("no" if g else ""),
            "google_lat": g[0] if g else "", "google_lng": g[1] if g else "",
            "dist_to_doorsight_m": d,
            "google_map_link": f"https://www.google.com/maps/search/?api=1&query={g[0]:.6f},{g[1]:.6f}" if g else "",
            "google_correct (y/n)": "",
        })
        time.sleep(0.2)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "google_entrances_raw.json").write_text(json.dumps(raw, indent=1))
    path = RESULTS / "google_entrances.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    got = [o for o in out if o["google_lat"] != ""]
    near = [o for o in got if o["dist_to_doorsight_m"] <= 5]
    print("\n=== Summary ===")
    print(f"Google returned an entrance: {len(got)} of {len(out)}")
    print(f"  within 5 m of DoorSight's verified pin: {len(near)} of {len(got)}")
    print(f"written {path}  (check each Google pin by hand: fill 'google_correct (y/n)')")


if __name__ == "__main__":
    main()
