#!/usr/bin/env python3
"""
DoorSight web app — run locally:  python app.py   then open http://localhost:8000

- Paste addresses or upload a CSV; jobs run in the background with zero clicks.
- Map shows every pin coloured by tier, with the street-level zoom crop and aerial crop as evidence.
- Export HIGH (or HIGH+MEDIUM) pins as CSV / GeoJSON.

Pure standard library server (no framework needed) so it runs anywhere Python runs.
"""
import csv
import io
import json
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import doorsight as D

ROOT = Path(__file__).resolve().parent
STATE = {"jobs": {}, "results": {}, "review": {}}   # job_id -> status ; address -> result ; address -> verdict
LOCK = threading.Lock()
REVIEW_FILE = None  # set in main


def _load_existing():
    for p in (D.RESULTS).glob("*/result.json"):
        try:
            r = json.loads(p.read_text())
            STATE["results"][r["address"]] = r
        except Exception:
            pass
    if REVIEW_FILE and REVIEW_FILE.exists():
        try:
            STATE["review"] = json.loads(REVIEW_FILE.read_text())
        except Exception:
            pass


def _verdict_for(addr, r):
    """A stored verdict only counts if the pin hasn't moved since it was reviewed."""
    v = STATE["review"].get(addr)
    if not v:
        return None
    if isinstance(v, str):
        return v
    if r.get("lat") is None or v.get("lat") is None:
        return v.get("verdict")
    import math
    d = math.hypot((r["lat"] - v["lat"]) * 111000, (r["lng"] - v["lng"]) * 111000 * math.cos(math.radians(r["lat"])))
    return v.get("verdict") if d <= 2.0 else None


def _precision():
    """Per tier: reviewed, right, wrong, unsure, precision = right / (right + wrong)."""
    out = {}
    for addr, r in STATE["results"].items():
        t = r.get("tier")
        v = _verdict_for(addr, r)
        d = out.setdefault(t, {"total": 0, "right": 0, "wrong": 0, "unsure": 0})
        d["total"] += 1
        if v in d:
            d[v] += 1
    for t, d in out.items():
        n = d["right"] + d["wrong"]
        d["precision"] = round(100 * d["right"] / n, 1) if n else None
    return out


def _run_job(job_id, addresses):
    for i, addr in enumerate(addresses):
        with LOCK:
            STATE["jobs"][job_id].update({"current": addr, "done": i})
        try:
            r = D.run_one(addr)
        except Exception as e:
            r = {"address": addr, "tier": "ERROR", "why": str(e)}
        with LOCK:
            STATE["results"][addr] = r
    with LOCK:
        STATE["jobs"][job_id].update({"done": len(addresses), "status": "finished", "current": None})


def _pins(tiers):
    out = []
    for r in STATE["results"].values():
        if r.get("tier") in tiers and r.get("lat") is not None and _verdict_for(r["address"], r) != "wrong":
            out.append(r)
    return out


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self.path = "/ui.html"
            return super().do_GET()
        if u.path == "/api/results":
            with LOCK:
                slim = [{k: r.get(k) for k in ("address", "tier", "lat", "lng", "why", "spread_m", "side", "aerial_agrees", "error_m")}
                        for r in STATE["results"].values()]
                for s, r in zip(slim, STATE["results"].values()):
                    ev = r.get("evidence_best") or [x.get("evidence") for x in r.get("sightings", []) if x.get("evidence")]
                    s["evidence"] = [f"/results/{D.slug(r['address'])}/{e}" for e in ev if e]
                    if (r.get("aerial") or {}).get("evidence"):
                        s["aerial_img"] = f"/results/{D.slug(r['address'])}/{r['aerial']['evidence']}"
                    s["verdict"] = _verdict_for(r["address"], r)
                return self._json({"results": slim, "jobs": STATE["jobs"], "precision": _precision()})
        if u.path == "/api/export":
            q = parse_qs(u.query)
            tiers = set(q.get("tiers", ["HIGH"])[0].split(","))
            fmt = q.get("fmt", ["csv"])[0]
            pins = _pins(tiers)
            if fmt == "geojson":
                gj = {"type": "FeatureCollection", "features": [
                    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lng"], p["lat"]]},
                     "properties": {"address": p["address"], "tier": p["tier"], "why": p["why"]}} for p in pins]}
                body = json.dumps(gj, indent=1).encode(); ctype = "application/geo+json"; name = "doors.geojson"
            else:
                buf = io.StringIO(); w = csv.writer(buf)
                w.writerow(["address", "door_lat", "door_lng", "tier", "why"])
                for p in pins:
                    w.writerow([p["address"], f"{p['lat']:.7f}", f"{p['lng']:.7f}", p["tier"], p["why"]])
                body = buf.getvalue().encode(); ctype = "text/csv"; name = "doors.csv"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
            return
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode("utf-8", "ignore")
        if u.path == "/api/run":
            try:
                body = json.loads(raw)
                text = body.get("addresses", "")
            except json.JSONDecodeError:
                text = raw
            addrs = []
            for line in text.splitlines():
                line = line.strip().strip('"')
                if not line or line.lower().startswith("address"):
                    continue
                addrs.append(line.split('","')[0].strip('"'))
            if not addrs:
                return self._json({"error": "no addresses"}, 400)
            job_id = uuid.uuid4().hex[:8]
            with LOCK:
                STATE["jobs"][job_id] = {"status": "running", "total": len(addrs), "done": 0, "current": None}
            threading.Thread(target=_run_job, args=(job_id, addrs), daemon=True).start()
            return self._json({"job": job_id, "count": len(addrs)})
        if u.path == "/api/review":
            body = json.loads(raw)
            with LOCK:
                r = STATE["results"].get(body["address"], {})
                STATE["review"][body["address"]] = {"verdict": body["verdict"], "lat": r.get("lat"), "lng": r.get("lng"),
                                                    "tier": r.get("tier")}
                REVIEW_FILE.write_text(json.dumps(STATE["review"], indent=1))
            return self._json({"ok": True, "precision": _precision()})
        self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    D.RESULTS.mkdir(exist_ok=True)
    REVIEW_FILE = D.RESULTS / "review.json"
    _load_existing()
    print("DoorSight running at http://localhost:8000  (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
