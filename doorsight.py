#!/usr/bin/env python3
"""
DoorSight — find a building's main entrance door from street-level photos.

  python doorsight.py one "123 Main St, Kirkwood, MO 63122"
  python doorsight.py batch addresses.csv          # column: address  (optional: true_lat,true_lng)

Pipeline per address:
  geocode -> OSM footprint -> cameras that see it -> zoom cascade (wide -> medium -> full zoom)
  -> door pixel -> bearing -> ray x footprint = door point -> cross-camera agreement -> confidence
Evidence (zoomed crops) and a JSON result are written to results/<slug>/.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import geometry as G
import providers as P
import aerial as A
from detect import detect, read_house_numbers, all_candidates
import rank as R
from PIL import Image

RESULTS = Path(__file__).resolve().parent / "results"
ZOOMS = [90, 40, 18]          # wide -> medium -> full zoom (deg FOV)
AGREE_TOL_M = 5.0
SWEEP = 0.3                    # full-zoom sweep as a fraction of fov (0.3*18 = ~5 deg)
TRI_MAX_FROM_ADDR_M = 12.0     # triangulated door must be this close to the address point
SAME_DOOR_TOL = 70.0           # mean-RGB distance; above this the two cameras saw different doors
SAME_CTX_TOL = 60.0
SYNTH_FOOTPRINT_M = 16.0       # stand-in building size when OSM has no outline
BUDGET = os.environ.get("DOORSIGHT_BUDGET", "1") != "0"   # fewer Google image calls per building


def log(m):
    print(f"[doorsight] {m}", flush=True)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def zoom_cascade(cam, footprint, outdir, tag, house_number=None, max_levels=None):
    """Look at the building from one camera, zooming in on the door. Returns a sighting or None.
    Wide view only aims the camera. The sweep at full zoom is narrow (+-SWEEP*fov) and only happens
    when the medium zoom already saw a door, so the camera cannot wander to a neighbour's house."""
    c = G.centroid(footprint)
    bearing = G.bearing_to(cam["lat"], cam["lng"], c[0], c[1])
    levels = ZOOMS[:max_levels] if max_levels else ZOOMS
    sighting = None
    numbers_seen = []
    for level, fov in enumerate(levels):
        last = level == len(ZOOMS) - 1
        tries = [0, -SWEEP, SWEEP] if (last and sighting is not None) else [0]
        found = False
        for off in tries:
            b = (bearing + off * fov) % 360
            v = P.view(cam, b, fov)
            if v is None:
                log(f"  {tag}: camera can't look toward {b:.0f}deg")
                return sighting
            img, heading, used_fov = v
            path = outdir / f"{tag}_z{level}_fov{fov}{'' if off == 0 else ('_L' if off < 0 else '_R')}.jpg"
            img.save(path, quality=88)
            d = detect(img)
            if level == 1 and house_number:
                numbers_seen = read_house_numbers(img, expect_len=len(house_number))
            if d and d.get("visible"):
                found = True
                bearing = G.pixel_to_bearing(d["x"], heading, used_fov)
                hit = G.ray_polygon_intersection((cam["lat"], cam["lng"]), bearing, footprint)
                log(f"  {tag} zoom{level} fov{fov}: door at x={d['x']:.2f} conf={d['confidence']:.2f} -> bearing {bearing:.1f}deg")
                sighting = {
                    "camera": {k: cam.get(k) for k in ("provider", "id", "lat", "lng", "dist", "captured_at", "road", "on_street")},
                    "level": level, "fov": fov, "bearing": round(bearing, 2),
                    "det_confidence": d["confidence"], "reason": d["reason"],
                    "evidence": path.name, "box": d.get("box"),
                    "n_doors_medium": d.get("n_doors", 0) if level == 1 else (sighting or {}).get("n_doors_medium", 0),
                    "door": {"lat": hit[0], "lng": hit[1], "range_m": round(hit[2], 1)} if hit else None,
                }
                break
            log(f"  {tag} zoom{level} fov{fov}{'' if off == 0 else ' sweep'}: no door ({(d or {}).get('reason','')})")
        if not found and last:
            break
    if sighting is not None and house_number:
        sighting["numbers_seen"] = numbers_seen
        if numbers_seen:
            if house_number in numbers_seen:
                sighting["number_match"] = True
                log(f"  {tag}: house number {house_number} visible -> confirmed")
            elif not any(n.endswith(house_number[-2:]) for n in numbers_seen):
                sighting["number_match"] = False
                log(f"  {tag}: saw house number(s) {numbers_seen}, expected {house_number} -> WRONG BUILDING, rejected")
                return None
    return sighting


def collect_doors(cam, footprint, outdir, tag, house_number=None, max_zoom_in=3):
    """Human-style pass from one camera: wide shot to aim, medium shot to list EVERY door with its context
    (steps, porch, mailbox, deck, number beside it), then zoom on the top few to confirm and refine bearing.
    Returns a list of candidates."""
    c = G.centroid(footprint)
    aim = G.bearing_to(cam["lat"], cam["lng"], c[0], c[1])
    out = []
    # medium shot (fov 40) is where context is visible
    v = P.view(cam, aim, ZOOMS[1])
    if v is None:
        return out
    img, heading, fov = v
    path = outdir / f"{tag}_z1_fov{ZOOMS[1]}.jpg"
    img.save(path, quality=88)
    res = all_candidates(img)
    numbers = read_house_numbers(img, expect_len=len(house_number)) if house_number else []
    Wd, Hd = img.size
    cands = res["doors"][:max_zoom_in]
    log(f"  {tag}: {len(res['doors'])} door candidate(s) at medium zoom; numbers seen: {[n['text'] for n in numbers] or '-'}")
    for k, cnd in enumerate(cands):
        bearing = G.pixel_to_bearing(cnd["x"], heading, fov)
        # number beside this door?
        near = None
        for n in numbers:
            nb = n["box"]; db = cnd["box"]
            if abs((nb[0] + nb[2]) / 2 - (db[0] + db[2]) / 2) <= 2.5 * (db[2] - db[0]) and nb[1] <= db[3]:
                near = n["text"]; break
        # confirm at full zoom
        conf, evidence, box = cnd["score"], path.name, cnd["box"]
        z = P.view(cam, bearing, ZOOMS[2])
        if z is not None:
            zimg, zh, zf = z
            zpath = outdir / f"{tag}_z2_c{k}.jpg"
            zimg.save(zpath, quality=88)
            zres = all_candidates(zimg)
            best = max((d for d in zres["doors"] if abs(d["x"] - 0.5) < 0.25), key=lambda d: d["score"], default=None)
            if best:
                bearing = G.pixel_to_bearing(best["x"], zh, zf)
                conf = max(conf, best["score"]); evidence = zpath.name; box = best["box"]
                for fk, fv in best["features"].items():
                    cnd["features"][fk] = max(cnd["features"].get(fk, 0), fv)
            else:
                conf *= 0.7   # not re-found at full zoom
        out.append({"cam": {k2: cam.get(k2) for k2 in ("provider", "id", "lat", "lng", "dist", "captured_at", "road", "on_street")},
                    "bearing": round(bearing, 2), "score": round(conf, 3), "features": cnd["features"],
                    "number_near": near, "evidence": evidence, "box": box, "n_doors_medium": res["n_doors"]})
        log(f"  {tag} door#{k}: conf={conf:.2f} bearing={bearing:.1f} feats={ {a: b for a, b in cnd['features'].items() if b} } number={near or '-'}")
    return out


def same_door(s1, s2, outdir):
    """Do two sightings show the same door? Compare colour of the door box and its surroundings."""
    try:
        import numpy as np
        feats = []
        for s in (s1, s2):
            if not s.get("box"):
                return None
            im = np.asarray(Image.open(outdir / s["evidence"]).convert("RGB").resize((640, 640)), dtype=float)
            x1, y1, x2, y2 = s["box"]
            door = im[max(0, y1):y2, max(0, x1):x2]
            if door.size == 0:
                return None
            pad = 40
            ctx = im[max(0, y1 - pad):min(640, y2 + pad), max(0, x1 - pad):min(640, x2 + pad)]
            feats.append((door.reshape(-1, 3).mean(0), ctx.reshape(-1, 3).mean(0)))
        d_door = float(np.linalg.norm(feats[0][0] - feats[1][0]))
        d_ctx = float(np.linalg.norm(feats[0][1] - feats[1][1]))
        return {"door_colour_diff": round(d_door, 1), "context_diff": round(d_ctx, 1),
                "same": d_door < SAME_DOOR_TOL and d_ctx < SAME_CTX_TOL}
    except Exception:
        return None


def triangulate(sightings, footprint):
    """Cross the bearing rays of the two best sightings. Returns (lat,lng) or None."""
    good = [s for s in sightings if s.get("bearing") is not None]
    if len(good) < 2:
        return None
    good.sort(key=lambda s: (-s["level"], -s["det_confidence"]))
    a, b = good[0], good[1]
    x = G.ray_ray_intersection((a["camera"]["lat"], a["camera"]["lng"]), a["bearing"],
                               (b["camera"]["lat"], b["camera"]["lng"]), b["bearing"])
    if not x:
        return None
    lat, lng, r1, r2 = x
    c = G.centroid(footprint)
    if not (4 <= r1 <= 60 and 4 <= r2 <= 60) or G.dist_m((lat, lng), c) > TRI_MAX_FROM_ADDR_M:
        return None
    return lat, lng


def near_side_fix(pt, members, footprint):
    """A door can't be behind the house from the camera that photographed it."""
    c = G.centroid(footprint)
    for m in members:
        cam = (m["cam"]["lat"], m["cam"]["lng"])
        if G.dist_m(cam, pt) > G.dist_m(cam, c) + 1.5:
            hit = G.ray_polygon_intersection(cam, m["bearing"], footprint)
            if hit:
                return (hit[0], hit[1]), "pin was behind the house from the camera; moved to the near wall"
    return pt, None


def grade_ranked(cands, footprint, air, house_number, footprint_source="osm"):
    """Pick the main door among all candidates and decide the tier."""
    best, runner, doors = R.choose(cands, footprint, air, house_number)
    multiunit = any(c.get("n_doors_medium", 0) >= 3 for c in cands)
    if not best:
        if air and air.get("entry_point") and air.get("confidence", 0) >= 0.5:
            e = air["entry_point"]
            return {"tier": "SIDE", "lat": e[0], "lng": e[1], "side": air["side"],
                    "why": f"no door visible from street; aerial {air['feature']} says {air['side']} side", "doors": []}
        return {"tier": "NONE", "why": "no door seen from any camera, no aerial clue", "doors": []}
    pt, note = near_side_fix(best["pt"], best["members"], footprint)
    behind = False
    if footprint_source in ("osm", "microsoft"):   # real outline -> put the pin on a wall the camera can see
        cams = [(m["cam"]["lat"], m["cam"]["lng"]) for m in best["members"]]
        sn = G.snap_to_facing_edge(pt, footprint, cams, max_m=10.0)
        if sn:
            pt = (sn[0], sn[1])
        # final sanity check: the pin may not sit farther from every camera than the house's middle
        c = G.centroid(footprint)
        behind = all(G.dist_m(cam, pt) > G.dist_m(cam, c) + 1.0 for cam in cams)
    f = best["features"]
    margin = best["score"] - (runner["score"] if runner else -9)
    why = [f"main-door score {best['score']}" + (f" vs runner-up {runner['score']}" if runner else ""),
           "features: " + ", ".join(f"{k}={v}" for k, v in f.items())]
    if note:
        why.append(note)
    # tier logic
    strong = best["two_cams"] or f.get("number_match") or (f.get("walkway", 0) >= 0.5 and f.get("on_street"))
    if f.get("number_mismatch"):
        tier = "LOW"; why.append("number beside door does not match address")
    elif strong and margin >= R.MARGIN and best["score"] >= 2.0:
        tier = "HIGH"
    elif best["score"] >= 1.5:
        tier = "MEDIUM"; why.append("no clear winner" if margin < R.MARGIN else "single weak confirmation")
    else:
        tier = "LOW"
    if tier == "HIGH" and behind:
        tier = "MEDIUM"; why.append("pin would sit on the far side of the house from the cameras")
    if tier == "HIGH" and not multiunit and not any(m["cam"].get("on_street") for m in best["members"]):
        tier = "MEDIUM"; why.append("no camera on the address's own street")
    if multiunit and tier == "HIGH" and not f.get("number_match"):
        tier = "MEDIUM"; why.append("multi-unit building, number not confirmed")
    return {"tier": tier, "lat": pt[0], "lng": pt[1], "score": best["score"], "why": "; ".join(why),
            "evidence_best": [m["evidence"] for m in best["members"]],
            "doors": [{"lat": d["pt"][0], "lng": d["pt"][1], "score": d["score"], "two_cams": d["two_cams"],
                       "features": d["features"], "evidence": [m["evidence"] for m in d["members"]]} for d in doors]}


def run_one(address, true_pt=None):
    outdir = RESULTS / slug(address)
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"== {address}")
    lat, lng = P.geocode(address)
    footprint, footprint_source = None, "osm"
    try:
        import footprints as FP            # Microsoft outlines, if footprints_local.geojson exists
        footprint = FP.lookup(lat, lng)
        if footprint:
            footprint_source = "microsoft"
            log(f"  Microsoft building outline ({len(footprint)} corners)")
    except Exception as e:
        log(f"  local outline lookup failed ({str(e)[:60]})")
    if not footprint:
        try:
            footprint = P.building_footprint(lat, lng)
        except Exception as e:
            log(f"  OSM lookup failed ({str(e)[:60]}) -> falling back")
            footprint = None
    if not footprint:
        # OSM has no outline here (common in US suburbs): use a box around the rooftop geocode.
        footprint = P.synthetic_footprint(lat, lng, size_m=SYNTH_FOOTPRINT_M)
        footprint_source = "synthetic"
        log(f"  no OSM footprint -> using {SYNTH_FOOTPRINT_M} m box around geocode")
    cams = P.candidate_cameras(lat, lng, footprint)
    log(f"  footprint ok, {len(cams)} camera(s): {[c['provider'] for c in cams]}")
    try:
        air = A.analyze(lat, lng, footprint, outdir)
        if air:
            log(f"  aerial: side={air['side']} conf={air['confidence']} ({air.get('feature','-')})")
    except Exception as e:
        log(f"  aerial failed: {e}"); air = None
    m = re.match(r"\s*(\d+)", address)
    house_number = m.group(1) if m else None
    street = P.norm_street(re.sub(r"^\s*\d+[A-Za-z]?\s+", "", address.split(",")[0]))
    for cam in cams:
        road = P.road_name_at(cam["lat"], cam["lng"])
        cam["road"] = road
        cam["on_street"] = bool(road) and P.norm_street(road) == street
        log(f"  camera on '{road or '?'}' -> {'address street' if cam['on_street'] else 'OTHER road (back/side door risk)'}")
    cands = []
    for i, cam in enumerate(cams):
        cands += collect_doors(cam, footprint, outdir, tag=f"cam{i}_{cam['provider']}", house_number=house_number)
    sightings = cands
    verdict = grade_ranked(cands, footprint, air, house_number, footprint_source)
    res = {"address": address, "geocode": {"lat": lat, "lng": lng}, "footprint": footprint,
           "footprint_source": footprint_source, "candidates": cands, "aerial": air, **verdict}
    if true_pt and verdict.get("lat") is not None:
        res["error_m"] = round(G.dist_m(true_pt, (verdict["lat"], verdict["lng"])), 1)
    (outdir / "result.json").write_text(json.dumps(res, indent=2))
    msg = f"  -> {verdict['tier']}"
    if verdict.get("lat") is not None:
        msg += f"  {verdict['lat']:.6f},{verdict['lng']:.6f}"
    if "error_m" in res:
        msg += f"  (error {res['error_m']} m)"
    log(msg + f"  [{verdict['why']}]")
    return res


def cmd_one(a):
    r = run_one(a.address)
    print(json.dumps({k: r.get(k) for k in ("address", "tier", "lat", "lng", "why")}, indent=2))


def cmd_batch(a):
    rows = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8")))
    results = []
    for row in rows:
        addr = row.get("address") or row.get("Address")
        tp = None
        if row.get("true_lat") and row.get("true_lng"):
            tp = (float(row["true_lat"]), float(row["true_lng"]))
        try:
            results.append(run_one(addr, tp))
        except Exception as e:  # keep the batch going
            log(f"  FAILED: {e}")
            results.append({"address": addr, "tier": "ERROR", "why": str(e)})
    out = RESULTS / "batch_results.json"
    out.write_text(json.dumps(results, indent=2))
    summarize(results)
    log(f"written {out}")


def summarize(results):
    tiers = {}
    for r in results:
        tiers.setdefault(r.get("tier", "?"), []).append(r)
    print("\n=== Summary ===")
    print(f"{'tier':8}{'count':>6}{'with truth':>12}{'<=5m':>8}{'<=10m':>8}{'median err':>12}")
    for t in ("HIGH", "MEDIUM", "LOW", "SIDE", "NONE", "ERROR"):
        rs = tiers.get(t, [])
        if not rs:
            continue
        errs = sorted(r["error_m"] for r in rs if "error_m" in r)
        n5 = sum(e <= 5 for e in errs)
        n10 = sum(e <= 10 for e in errs)
        med = f"{errs[len(errs)//2]:.1f} m" if errs else "-"
        print(f"{t:8}{len(rs):>6}{len(errs):>12}{n5:>8}{n10:>8}{med:>12}")
    print()


def main():
    p = argparse.ArgumentParser(description="DoorSight — doors from street-level photos")
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("one"); o.add_argument("address"); o.set_defaults(func=cmd_one)
    b = sub.add_parser("batch"); b.add_argument("csv"); b.set_defaults(func=cmd_batch)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
