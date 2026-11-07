"""
Main-door ranker — the "human" step.

Every door seen from every camera becomes a candidate. Each candidate is scored on what a person
looks for when deciding which door is the main entrance:

  + the walking path from the street / driveway ends at it      (aerial walkway endpoint nearby)
  + steps up to it, a porch roof or canopy over it              (street-level context boxes)
  + the house number is right beside it                          (OCR near the door box)
  + a mailbox / package box beside it
  + it faces the address's own street (camera on that street)
  + two different cameras found the same door                    (triangulated cluster)
  - sliding glass door, deck or patio at it                      (back door)
  - a visible number beside it that does NOT match the address  (wrong building / unit)

The best-scoring candidate wins. If the runner-up is close, the answer is MEDIUM, not HIGH.
"""
import geometry as G

W = {
    "det": 1.0, "steps": 0.8, "porch": 0.8, "mailbox": 0.5, "deck": -1.0, "slider": -1.2,
    "number_match": 1.2, "number_mismatch": -1.5, "on_street": 0.6, "two_cams": 1.0,
    "walkway": 0.9, "driveway": 0.3, "aerial_side": 0.4,
}
CLUSTER_M = 4.0      # candidates from two cameras closer than this are the same door
MERGE_M = 4.0        # two finished doors closer than this are the same physical door (merge them)
MARGIN = 0.6         # winner must beat runner-up by this much to be HIGH


def locate(cand, footprint):
    """Map point for a single-camera candidate: ray x footprint, else nearest point on ray to the house."""
    cam = (cand["cam"]["lat"], cand["cam"]["lng"])
    hit = G.ray_polygon_intersection(cam, cand["bearing"], footprint)
    if hit:
        return (hit[0], hit[1]), "ray_hit"
    c = G.centroid(footprint)
    npx = G.ray_nearest_point(cam, cand["bearing"], c, max_m=25.0)
    if npx:
        return (npx[0], npx[1]), "ray_nearest"
    return None, None


def cluster(cands, footprint, max_from_addr=12.0):
    """Group candidates across cameras into doors. Returns list of doors:
    {"pt": (lat,lng), "members": [cand...], "two_cams": bool, "method": str}"""
    doors = []
    used = set()
    c = G.centroid(footprint)
    # pairs from different cameras whose rays cross near the address
    for i, a in enumerate(cands):
        for j, b in enumerate(cands):
            if j <= i or a["cam"]["id"] == b["cam"]["id"] or i in used or j in used:
                continue
            x = G.ray_ray_intersection((a["cam"]["lat"], a["cam"]["lng"]), a["bearing"],
                                       (b["cam"]["lat"], b["cam"]["lng"]), b["bearing"])
            if not x:
                continue
            lat, lng, r1, r2 = x
            if 4 <= r1 <= 60 and 4 <= r2 <= 60 and G.dist_m((lat, lng), c) <= max_from_addr:
                doors.append({"pt": (lat, lng), "members": [a, b], "two_cams": True, "method": "triangulated"})
                used.update((i, j))
                break
    for i, a in enumerate(cands):
        if i in used:
            continue
        pt, how = locate(a, footprint)
        if not pt:
            continue
        # fold into an existing door if very close
        for d in doors:
            if G.dist_m(d["pt"], pt) <= CLUSTER_M:
                d["members"].append(a)
                break
        else:
            doors.append({"pt": pt, "members": [a], "two_cams": False, "method": how})
    return merge_close(doors)


def merge_close(doors, tol=None):
    """The same physical door can come out as several 'doors': each camera often lists the
    same door 2-3 times, and every cross-camera pairing is triangulated separately.
    Merge doors whose points are within MERGE_M, so a door can't tie with its own duplicate."""
    tol = MERGE_M if tol is None else tol
    merged = []
    # strongest first, so the merged point stays on the best-supported position
    for d in sorted(doors, key=lambda d: (-d["two_cams"], -len(d["members"]))):
        for m in merged:
            if G.dist_m(m["pt"], d["pt"]) <= tol:
                m["members"] += [x for x in d["members"] if x not in m["members"]]
                m["two_cams"] = m["two_cams"] or d["two_cams"]
                m["merged"] = m.get("merged", 1) + 1
                break
        else:
            merged.append(dict(d))
    return merged


def score(door, air, house_number):
    m = door["members"]
    f = {k: 0.0 for k in W}
    f["det"] = max(x["score"] for x in m)
    for k in ("steps", "porch", "mailbox", "deck", "slider"):
        f[k] = max(x["features"].get(k, 0.0) for x in m)
    f["on_street"] = 1.0 if any(x["cam"].get("on_street") for x in m) else 0.0
    f["two_cams"] = 1.0 if door["two_cams"] else 0.0
    nm = [x.get("number_near") for x in m if x.get("number_near")]
    if nm:
        if any(n == house_number or n.endswith(house_number) for n in nm):
            f["number_match"] = 1.0
        else:
            f["number_mismatch"] = 1.0
    if air:
        for e in air.get("endpoints", []) or []:
            d = G.dist_m(door["pt"], (e["lat"], e["lng"]))
            if e["kind"] in ("walkway", "steps", "front porch") and d <= 6:
                f["walkway"] = max(f["walkway"], e["score"] * 2)   # 0.12..0.5 -> up to ~1
            elif e["kind"] == "driveway" and d <= 8:
                f["driveway"] = max(f["driveway"], e["score"] * 2)
        if air.get("side_bearing") is not None and air.get("confidence", 0) >= 0.3:
            # does this door sit on the side the aerial eye picked?
            f["aerial_side"] = 1.0 if _on_side(door["pt"], air, m[0]["footprint_centroid"]) else 0.0
    total = sum(W[k] * min(1.0, f[k]) for k in W)
    door["features"] = {k: round(v, 2) for k, v in f.items() if v}
    door["score"] = round(total, 2)
    return total


def _on_side(pt, air, c):
    b = G.bearing_to(c[0], c[1], pt[0], pt[1])
    diff = abs(((b - air["side_bearing"] + 180) % 360) - 180)
    return diff <= 60


def choose(cands, footprint, air, house_number):
    """Rank all candidates. Returns (best_door, runner_up, all_doors) — best may be None."""
    if not cands:
        return None, None, []
    c = G.centroid(footprint)
    for x in cands:
        x["footprint_centroid"] = c
    doors = cluster(cands, footprint)
    for d in doors:
        score(d, air, house_number)
    doors.sort(key=lambda d: -d["score"])
    best = doors[0] if doors else None
    runner = doors[1] if len(doors) > 1 else None
    return best, runner, doors
