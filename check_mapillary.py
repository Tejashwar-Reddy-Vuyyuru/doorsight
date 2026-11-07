"""Diagnose Mapillary access: is the token set, does it work, is there coverage near a point?
    python check_mapillary.py                      # checks downtown Kirkwood
    python check_mapillary.py 38.5833 -90.4067     # checks a specific lat lng
"""
import os
import sys

import requests

tok = os.environ.get("MAPILLARY_TOKEN", "")
print("token set:", "YES" if tok else "NO  <-- run  $env:MAPILLARY_TOKEN=\"MLY|...\"  in this window")
if tok:
    print("token starts with:", tok[:6], "| length:", len(tok))
    if not tok.startswith("MLY|"):
        print("WARNING: Mapillary tokens start with MLY| — check you copied the Access Token, not App ID")

lat, lng = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) == 3 else (38.5833, -90.4067)
for radius in (100, 500, 2000):
    d = radius / 111000.0
    bbox = f"{lng - d},{lat - d},{lng + d},{lat + d}"
    r = requests.get("https://graph.mapillary.com/images",
                     params={"access_token": tok, "bbox": bbox, "limit": 50,
                             "fields": "id,is_pano,captured_at"}, timeout=30)
    if r.status_code != 200:
        print(f"radius {radius} m: HTTP {r.status_code} -> {r.text[:200]}")
        break
    data = r.json().get("data", [])
    panos = sum(1 for x in data if x.get("is_pano"))
    print(f"radius {radius:5d} m: {len(data):3d} images ({panos} panoramas)")
