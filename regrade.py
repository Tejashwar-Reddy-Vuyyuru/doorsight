#!/usr/bin/env python3
"""
Re-grade a finished batch WITHOUT new image calls.

Reads results/batch_results.json (the doors, cameras and aerial clues already found),
re-runs the ranking + tier logic from rank.py / doorsight.py, and shows what changed.
Use it to test ranking fixes for free before spending Street View calls.

  python regrade.py                       # compare old tiers vs new tiers
  python regrade.py --merge 3 4 5         # try several duplicate-merge distances (metres)
  python regrade.py --write               # also save results/batch_results_regraded.json
  python regrade.py --footprints          # swap the guessed 16 m boxes for real Microsoft outlines
"""
import argparse
import copy
import json
import sys
import types
from collections import Counter
from pathlib import Path

# The heavy modules (detector, Street View, aerial tiles) aren't needed to re-grade.
for name in ("providers", "aerial", "detect"):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        if name == "detect":
            stub.detect = stub.read_house_numbers = stub.all_candidates = None
        sys.modules[name] = stub

import rank as R          # noqa: E402
import doorsight as D     # noqa: E402

HERE = Path(__file__).resolve().parent
TIERS = ["HIGH", "MEDIUM", "LOW", "SIDE", "NONE", "ERROR"]


def norm(a):
    return " ".join(a.replace(", MO,", ", MO").split()).lower()


def regrade(results, merge_m, use_outlines=False):
    R.MERGE_M = merge_m
    FP = None
    if use_outlines:
        import footprints as FP
        if not FP.available():
            sys.exit("no footprints_local.geojson yet - run: python footprints.py extract Missouri.geojson")
    out = []
    for r in results:
        if r.get("tier") == "ERROR" or "footprint" not in r:
            out.append({**r, "old_tier": r.get("tier")})   # nothing to re-grade
            continue
        cands = copy.deepcopy(r.get("candidates") or [])
        m = D.re.match(r"\s*(\d+)", r["address"])
        house_number = m.group(1) if m else None
        footprint, src = r["footprint"], r.get("footprint_source", "osm")
        if FP and src == "synthetic":
            ring = FP.lookup(r["geocode"]["lat"], r["geocode"]["lng"])
            if ring:
                footprint, src = ring, "microsoft"
        v = D.grade_ranked(cands, footprint, r.get("aerial"), house_number, src)
        out.append({**r, **v, "footprint": footprint, "footprint_source": src,
                    "old_tier": r.get("tier"), "old_lat": r.get("lat"), "old_lng": r.get("lng")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(HERE / "results" / "batch_results.json"))
    ap.add_argument("--review", default=str(HERE / "results" / "review.json"))
    ap.add_argument("--merge", type=float, nargs="*", default=[4.0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--footprints", action="store_true", help="use real Microsoft outlines")
    ap.add_argument("--checked", help="your filled checklist CSV: report how far each checked pin moved")
    a = ap.parse_args()

    results = json.loads(Path(a.file).read_text())
    review = {}
    if Path(a.review).exists():
        review = {norm(k): v for k, v in json.loads(Path(a.review).read_text()).items()}

    print(f"before: {dict(Counter(r.get('tier') for r in results))}")
    for mm in a.merge:
        new = regrade(results, mm, a.footprints)
        if a.footprints:
            src = Counter(r.get("footprint_source") for r in new)
            print(f"outlines used: {dict(src)}")
        c = Counter(r["tier"] for r in new)
        print(f"\n=== merge distance {mm} m ===")
        print("after:  " + ", ".join(f"{t} {c[t]}" for t in TIERS if c[t]))
        moved = [r for r in new if r.get("old_tier") != r["tier"]]
        print(f"{len(moved)} address(es) changed tier:")
        for r in moved:
            rv = review.get(norm(r["address"]), "not reviewed")
            print(f"  {r['old_tier']:>6} -> {r['tier']:<6}  {r['address'][:45]:45}  earlier eye-review: {rv}")
        high = [r for r in new if r["tier"] == "HIGH"]
        rv = Counter(review.get(norm(r["address"]), "not reviewed") for r in high)
        print(f"HIGH pins vs earlier eye-review: right {rv['right']}, wrong {rv['wrong']}, "
              f"not reviewed {rv['not reviewed']}")
        if a.checked:
            report_moves(new, a.checked)
        if a.write:
            p = Path(a.file).with_name("batch_results_regraded.json")
            p.write_text(json.dumps(new, indent=2))
            print(f"written {p}")
            write_checklist(high, review, Path(a.file).with_name("high_checklist.csv"))


def report_moves(new, checked_csv):
    """For houses you already checked: how far did each pin move, and did it stay HIGH?"""
    import csv
    import geometry as G
    rows = {norm(r["address"]): r for r in csv.DictReader(open(checked_csv, newline="", encoding="utf-8-sig"))}
    by = {norm(r["address"]): r for r in new}
    print("\nyour checked pins after this re-grade:")
    print(f"  {'address':40} {'you said':10} {'tier now':8} {'moved':>7}")
    for k, row in rows.items():
        r = by.get(k)
        if not r:
            continue
        extra = row.get(None) or [v for k, v in row.items() if k and k.strip() == ""] or [""]
        note = " ".join(x for x in (extra if isinstance(extra, list) else [extra]) if x)
        said = ((row.get("correct (y/n)") or "").strip() + " " + note.strip()).strip()
        moved = "-"
        if r.get("lat") is not None and row.get("lat"):
            moved = f"{G.dist_m((float(row['lat']), float(row['lng'])), (r['lat'], r['lng'])):.1f} m"
        print(f"  {r['address'][:40]:40} {said[:10]:10} {r['tier']:8} {moved:>7}")


def write_checklist(high, review, path):
    """One row per HIGH pin: open the map link (satellite, pin dropped) and mark the last column."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["address", "lat", "lng", "map_link", "earlier_review", "correct (y/n)"])
        for r in high:
            link = f"https://www.google.com/maps/search/?api=1&query={r['lat']:.6f},{r['lng']:.6f}"
            w.writerow([r["address"], f"{r['lat']:.6f}", f"{r['lng']:.6f}", link,
                        review.get(norm(r["address"]), ""), ""])
    print(f"written {path}  ({len(high)} HIGH pins to check)")


if __name__ == "__main__":
    main()
