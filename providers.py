"""
Data sources: geocoding (Nominatim), building footprints (OSM Overpass),
street-level imagery (Mapillary free, Google Street View fallback).
All functions return plain dicts/tuples; no heavy deps beyond `requests` + Pillow.
"""
import io
import math
import os
import time

import requests
from PIL import Image

from geometry import bearing_to, dist_m, point_in_polygon, centroid

UA = {"User-Agent": "DoorSight/0.1 (research prototype)"}
MAPILLARY_TOKEN = os.environ.get("MAPILLARY_TOKEN", "")
GOOGLE_KEY = os.environ.get("GOOGLE_MAPS_KEY", "")


# ----------------------------------------------------------------------------- geocoding
def geocode(address):
    """Address -> (lat, lng). Google if key present, else Nominatim (free, 1 req/s)."""
    if GOOGLE_KEY:
        r = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                         params={"address": address, "key": GOOGLE_KEY}, timeout=20).json()
        res = r.get("results") or []
        if res and not res[0].get("partial_match") and res[0]["geometry"].get("location_type") in ("ROOFTOP", "RANGE_INTERPOLATED"):
            loc = res[0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
        if res:
            raise ValueError(f"address not precise enough (Google: {res[0]['geometry'].get('location_type')}): {address}")
    rr = requests.get("https://nominatim.openstreetmap.org/search",
                      params={"q": address, "format": "json", "limit": 1}, headers=UA, timeout=20)
    time.sleep(1.1)
    r = rr.json() if rr.status_code == 200 and rr.text.lstrip().startswith("[") else []
    if not r:
        raise ValueError(f"could not geocode: {address}")
    return float(r[0]["lat"]), float(r[0]["lon"])


_ROAD_CACHE = {}


def road_name_at(lat, lng):
    """Name of the road at a point (Google reverse geocode 'route'; Nominatim fallback). '' if unknown."""
    key = (round(lat, 5), round(lng, 5))
    if key in _ROAD_CACHE:
        return _ROAD_CACHE[key]
    name = ""
    try:
        if GOOGLE_KEY:
            r = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                             params={"latlng": f"{lat},{lng}", "result_type": "route", "key": GOOGLE_KEY}, timeout=20).json()
            for res in r.get("results", []):
                for comp in res.get("address_components", []):
                    if "route" in comp.get("types", []):
                        name = comp["long_name"]; break
                if name:
                    break
        if not name:
            rr = requests.get("https://nominatim.openstreetmap.org/reverse",
                              params={"lat": lat, "lon": lng, "format": "json", "zoom": 16}, headers=UA, timeout=20)
            time.sleep(1.1)
            if rr.status_code == 200 and rr.text.lstrip().startswith("{"):
                name = (rr.json().get("address") or {}).get("road", "")
    except Exception:
        pass
    _ROAD_CACHE[key] = name
    return name


def norm_street(s):
    """'W Argonne Dr' / 'West Argonne Drive' -> 'w argonne dr' so names compare."""
    s = s.lower().replace(".", "")
    rep = {"street": "st", "avenue": "ave", "drive": "dr", "road": "rd", "lane": "ln", "court": "ct", "boulevard": "blvd",
           "place": "pl", "circle": "cir", "terrace": "ter", "north": "n", "south": "s", "east": "e", "west": "w"}
    return " ".join(rep.get(w, w) for w in s.split())


# ----------------------------------------------------------------------------- footprints
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]


def _overpass(q):
    """POST to Overpass with retries across mirrors; raises if all fail."""
    last = None
    for attempt in range(4):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=60)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                return r.json()
            last = f"HTTP {r.status_code} from {url}"
        except Exception as e:
            last = str(e)
        time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"Overpass unavailable: {last}")


def building_footprint(lat, lng, radius=40):
    """Nearest OSM building polygon to the point -> list of (lat,lng)."""
    q = f'[out:json][timeout:25];way["building"](around:{radius},{lat},{lng});out geom;'
    r = _overpass(q)
    polys = []
    for el in r.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        poly = [(g["lat"], g["lon"]) for g in el["geometry"]]
        if len(poly) >= 4:
            polys.append(poly[:-1] if poly[0] == poly[-1] else poly)
    if not polys:
        return None
    inside = [p for p in polys if point_in_polygon((lat, lng), p)]
    if inside:
        return inside[0]
    return min(polys, key=lambda p: dist_m((lat, lng), centroid(p)))


def synthetic_footprint(lat, lng, size_m=16.0):
    """Square of size_m centred on the point, as a list of (lat,lng). Used when OSM has no outline."""
    h = size_m / 2
    return [_offset(*_offset(lat, lng, 0, -h), 270, h), _offset(*_offset(lat, lng, 0, -h), 90, h),
            _offset(*_offset(lat, lng, 0, h), 90, h), _offset(*_offset(lat, lng, 0, h), 270, h)]


# ----------------------------------------------------------------------------- imagery: Mapillary
def mapillary_images(lat, lng, radius=60, limit=20):
    """Street-level images near a point. Returns list of dicts sorted by distance."""
    if not MAPILLARY_TOKEN:
        return []
    d = radius / 111000.0
    bbox = f"{lng - d},{lat - d},{lng + d},{lat + d}"
    r = requests.get("https://graph.mapillary.com/images",
                     params={"access_token": MAPILLARY_TOKEN, "bbox": bbox, "limit": limit,
                             "fields": "id,computed_geometry,computed_compass_angle,compass_angle,"
                                       "is_pano,thumb_2048_url,captured_at,camera_type"},
                     timeout=30).json()
    out = []
    for im in r.get("data", []):
        g = im.get("computed_geometry") or {}
        if not g.get("coordinates"):
            continue
        ilng, ilat = g["coordinates"]
        out.append({
            "provider": "mapillary",
            "id": im["id"],
            "lat": ilat, "lng": ilng,
            "heading": im.get("computed_compass_angle", im.get("compass_angle", 0.0)) or 0.0,
            "is_pano": bool(im.get("is_pano")),
            "url": im.get("thumb_2048_url"),
            "captured_at": im.get("captured_at"),
            "dist": dist_m((lat, lng), (ilat, ilng)),
        })
    out.sort(key=lambda x: x["dist"])
    return out


def mapillary_view(img, target_bearing, fov):
    """
    Return a pinhole-like crop of a Mapillary image looking at target_bearing with given fov.
    Panos: crop equirectangular. Flat images: crop around the bearing if within the ~70 deg field.
    Returns (PIL.Image, view_heading, view_fov) or None.
    """
    data = requests.get(img["url"], timeout=60).content
    im = Image.open(io.BytesIO(data)).convert("RGB")
    W, H = im.size
    if img["is_pano"]:
        # x=0.5 looks along heading; 360 deg across width
        rel = ((target_bearing - img["heading"] + 180) % 360) - 180
        cx = (0.5 + rel / 360.0) * W
        half_w = (fov / 360.0) * W / 2
        # vertical: keep middle band, aspect ~1:1
        band = half_w
        box = (int(cx - half_w), int(H / 2 - band), int(cx + half_w), int(H / 2 + band))
        if box[0] < 0 or box[2] > W:  # wrap-around: shift the pano
            im = im.transform(im.size, Image.AFFINE, (1, 0, (cx - W / 2), 0, 1, 0))
            box = (int(W / 2 - half_w), box[1], int(W / 2 + half_w), box[3])
        return im.crop(box), target_bearing, fov
    # flat image: assume ~70 deg horizontal field of view
    FLAT_FOV = 70.0
    rel = ((target_bearing - img["heading"] + 180) % 360) - 180
    if abs(rel) > FLAT_FOV / 2 - 5:
        return None
    half = math.tan(math.radians(FLAT_FOV / 2))
    cx = (0.5 + math.tan(math.radians(rel)) / (2 * half)) * W
    frac = math.tan(math.radians(fov / 2)) / half
    half_w = frac * W / 2
    box = (max(0, int(cx - half_w)), max(0, int(H / 2 - half_w)),
           min(W, int(cx + half_w)), min(H, int(H / 2 + half_w)))
    return im.crop(box), target_bearing, fov


# ----------------------------------------------------------------------------- imagery: Google Street View
def google_pano_near(lat, lng):
    if not GOOGLE_KEY:
        return None
    r = requests.get("https://maps.googleapis.com/maps/api/streetview/metadata",
                     params={"location": f"{lat},{lng}", "source": "outdoor", "radius": 60,
                             "key": GOOGLE_KEY}, timeout=20).json()
    if r.get("status") != "OK":
        return None
    return {"provider": "google", "id": r["pano_id"], "lat": r["location"]["lat"],
            "lng": r["location"]["lng"], "is_pano": True, "captured_at": r.get("date"),
            "dist": dist_m((lat, lng), (r["location"]["lat"], r["location"]["lng"]))}


def google_view(pano, target_bearing, fov, pitch=5):
    """Fetch a Street View image at a heading/fov. Returns (PIL.Image, heading, fov)."""
    r = requests.get("https://maps.googleapis.com/maps/api/streetview",
                     params={"pano": pano["id"], "size": "640x640", "heading": round(target_bearing, 1),
                             "fov": fov, "pitch": pitch, "key": GOOGLE_KEY}, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB"), target_bearing, fov


# ----------------------------------------------------------------------------- unified
MIN_CAM_DIST = 8.0   # metres; a camera closer than this to the geocode gives useless bearings
MAX_AGE_YEARS = 5    # ignore panos this much older than the newest available one
PREFER = os.environ.get("DOORSIGHT_PREFER", "google" if GOOGLE_KEY else "mapillary")


def candidate_cameras(lat, lng, footprint, max_cams=2):
    """Pick up to max_cams camera positions that can see the building.
    Google: gather panos at the address and stepped along the street, drop any closer than MIN_CAM_DIST,
    prefer the NEWEST imagery (old panos show repainted/rebuilt houses), and keep two that are >= 8 m apart.
    DOORSIGHT_PREFER=mapillary puts Mapillary first instead."""
    c = centroid(footprint) if footprint else (lat, lng)
    cams = []
    if PREFER != "google":
        for img in mapillary_images(lat, lng):
            if MIN_CAM_DIST < img["dist"] < 60:
                cams.append(img)
            if len(cams) >= max_cams:
                break
    if len(cams) < max_cams and GOOGLE_KEY:
        pool, seen = [], set()
        g0 = google_pano_near(lat, lng)
        if g0:
            b = bearing_to(g0["lat"], g0["lng"], c[0], c[1])
            probes = [(g0["lat"], g0["lng"])] + [_offset(g0["lat"], g0["lng"], (b + off) % 360, d)
                                                for d in (12, 24) for off in (90, -90)]
        else:
            probes = [(lat, lng)]
        for pl in probes:
            g = google_pano_near(*pl)
            if g and g["id"] not in seen:
                seen.add(g["id"])
                g["dist"] = dist_m((lat, lng), (g["lat"], g["lng"]))
                if MIN_CAM_DIST <= g["dist"] <= 45:
                    pool.append(g)
        # newest first; drop panos more than MAX_AGE_YEARS older than the newest
        pool.sort(key=lambda x: x.get("captured_at") or "", reverse=True)
        if pool:
            newest = int((pool[0].get("captured_at") or "0")[:4] or 0)
            pool = [g for g in pool if int((g.get("captured_at") or "0")[:4] or 0) >= newest - MAX_AGE_YEARS]
        for g in pool:
            if all(dist_m((g["lat"], g["lng"]), (k["lat"], k["lng"])) >= 8 for k in cams):
                cams.append(g)
            if len(cams) >= max_cams:
                break
    if len(cams) < max_cams and PREFER == "google":
        for img in mapillary_images(lat, lng):
            if MIN_CAM_DIST < img["dist"] < 60 and img["is_pano"]:
                cams.append(img)
            if len(cams) >= max_cams:
                break
    return cams[:max_cams]


def view(cam, target_bearing, fov):
    if cam["provider"] == "google":
        return google_view(cam, target_bearing, fov)
    return mapillary_view(cam, target_bearing, fov)


def _offset(lat, lng, bearing, metres):
    dlat = metres * math.cos(math.radians(bearing)) / 111000.0
    dlng = metres * math.sin(math.radians(bearing)) / (111000.0 * math.cos(math.radians(lat)))
    return lat + dlat, lng + dlng
