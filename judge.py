#!/usr/bin/env python3
"""
Main-door judge: a final yes/no check on every HIGH pin by Claude vision.

DoorSight's detector finds real doors well, but sometimes picks a back door, side door or
the door facing the wrong street. For each HIGH pin this shows Claude the street photo of
the house plus the close-up with the chosen door boxed in red, and asks one question:
"is this the main front entrance a visitor would use?"
  yes     -> stays HIGH
  no      -> MEDIUM  (reason saved)
  unsure  -> MEDIUM

Works on results you already have (uses the saved photos, no new Street View calls).
Answers are cached in results/<house>/judge.json, so re-running costs nothing.

  python judge.py                                          # judge, write judged results + checklist
  python judge.py --checked results\\high_checklist_checked.csv   # also compare with your own y/n

Needs: pip install anthropic   and   ANTHROPIC_API_KEY set.
Model: DOORSIGHT_JUDGE_MODEL (default claude-sonnet-5-5).
"""
import argparse
import base64
import csv
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import regrade as RG          # re-grade first (duplicate merge + real outlines if available)
import doorsight as D
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL = os.environ.get("DOORSIGHT_JUDGE_MODEL", "claude-sonnet-5-5")

PROMPT_VERSION = 2
PROMPT = """You are checking a delivery-pin dataset for houses.

Image 1 is a street-level photo of the house at: {address}
Image(s) 2+ are close-ups. The RED BOX marks the door our system chose as the main entrance.

Question: is the red-boxed door the MAIN FRONT ENTRANCE of this house - the door a visitor or
delivery driver would walk up to from the street?

First decide which side of the house the photo shows. Signs you are looking at the BACK or SIDE:
- the door is on a raised wooden deck or patio, reached by deck stairs
- a backyard fence (chain-link, privacy fence) stands between the camera and the house
- open lawn, patio furniture, grills, play sets, sheds, pools
- no walkway or path leading from the street to the door
- no mailbox, house number, porch light or front porch near the door
Signs of the FRONT: a walkway or steps from the street or driveway, a covered porch or stoop,
house number or mailbox nearby, sidelights or a storm door, the garage and driveway facing the same street.

Answer "no" if the box is on: a back or deck door, a patio or sliding glass door, a garage door,
a side or basement door, a window, a gate, or a door of a different building - or if the photo
shows the back or side of the house.
Answer "unsure" if the box is unclear, the door is hidden (trees, fence), or the photos do not
show enough to decide. Do not guess. Answer "yes" only when the photo shows the street-facing front.

Return ONLY JSON: {{"side_shown": "front" | "back" | "side" | "unclear", "main_entrance": "yes" | "no" | "unsure", "reason": "<one short sentence>"}}"""


def _jpeg_b64(img, max_side=1024):
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _boxed(path, box):
    img = Image.open(path).convert("RGB")
    if box:
        x1, y1, x2, y2 = [float(v) for v in box]
        W, H = img.size
        if max(x2, y2) <= 1.0:                      # normalised box
            x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
        pad = 6
        d = ImageDraw.Draw(img)
        d.rectangle([max(0, x1 - pad), max(0, y1 - pad), min(W - 1, x2 + pad), min(H - 1, y2 + pad)],
                    outline=(255, 0, 0), width=max(3, W // 150))
    return img


def evidence_images(r):
    """Context shot + boxed close-ups of the winning door. Returns list of PIL images."""
    folder = RESULTS / D.slug(r["address"])
    best = set(r.get("evidence_best") or [])
    members = [c for c in r.get("candidates") or [] if c.get("evidence") in best]
    imgs = []
    cams = sorted({e.split("_z")[0] for e in best})
    for cam in cams[:1]:                             # one medium-zoom context shot
        ctx = folder / f"{cam}_z1_fov40.jpg"
        if ctx.exists():
            imgs.append(Image.open(ctx).convert("RGB"))
    for m in members[:2]:                            # up to two close-ups, door boxed
        p = folder / m["evidence"]
        if p.exists():
            imgs.append(_boxed(p, m.get("box")))
    return imgs


def ask_claude(address, imgs, client):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": _jpeg_b64(im)}} for im in imgs]
    content.append({"type": "text", "text": PROMPT.format(address=address)})
    msg = client.messages.create(model=MODEL, max_tokens=600,
                                 messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    # read the answer even if the reply is wrapped in ``` or cut off part-way
    a = re.search(r'"main_entrance"\s*:\s*"(yes|no|unsure)"', text, re.I)
    rsn = re.search(r'"reason"\s*:\s*"([^"]*)', text)
    ans = a.group(1).lower() if a else "unsure"
    side = re.search(r'"side_shown"\s*:\s*"(front|back|side|unclear)"', text, re.I)
    side = side.group(1).lower() if side else "unclear"
    if ans == "yes" and side != "front":          # a yes about a back/side view is not a yes
        ans = "unsure"
    return {"answer": ans, "side": side, "reason": rsn.group(1) if rsn else "", "parsed": bool(a),
            "raw": text[:400], "model": MODEL, "prompt_version": PROMPT_VERSION}


def judge(results, client, refresh=False):
    n_new = 0
    for r in results:
        if r.get("tier") != "HIGH":
            continue
        cache = RESULTS / D.slug(r["address"]) / "judge.json"
        j = None
        if cache.exists() and not refresh:
            j = json.loads(cache.read_text())
            if j.get("evidence") != r.get("evidence_best"):   # pin changed since last judged
                j = None
            elif j.get("answer") not in ("yes", "no") and not j.get("parsed"):
                j = None                                      # earlier reply wasn't read properly: ask again
            elif j.get("prompt_version") != PROMPT_VERSION and j.get("answer") == "yes":
                j = None                                      # judged with an older question: ask again
        if j is None:
            imgs = evidence_images(r)
            if not imgs:                               # not cached: photos may appear later
                r["judge"] = {"answer": "unsure", "reason": "no saved photos for this house"}
                r["tier"] = "MEDIUM"
                print(f"  unsure  {r['address'][:45]:45} no saved photos for this house")
                continue
            else:
                try:
                    j = ask_claude(r["address"], imgs, client)
                    n_new += 1
                except Exception as e:                 # keep going; don't cache failures
                    print(f"  judge failed for {r['address']}: {str(e)[:100]}")
                    r["judge"] = {"answer": "error", "reason": str(e)[:100]}
                    r["tier"] = "MEDIUM"
                    continue
            j["evidence"] = r.get("evidence_best")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(j, indent=2))
        r["judge"] = j
        print(f"  {j['answer']:7} {r['address'][:45]:45} {(j.get('reason') or j.get('raw',''))[:70]}")
        if j["answer"] != "yes":
            r["tier"] = "MEDIUM"
            r["why"] = (r.get("why", "") + f"; judge: {j['answer']} - {j.get('reason','')}").strip("; ")
    return n_new


def compare(results, checked_csv):
    rows = {RG.norm(r["address"]): r for r in csv.DictReader(open(checked_csv, newline="", encoding="utf-8-sig"))}
    by = {RG.norm(r["address"]): r for r in results}
    tally = Counter()
    print("\njudge vs your own check (HIGH pins you checked):")
    for k, row in rows.items():
        r = by.get(k)
        if not r or "judge" not in r:
            continue
        you = (row.get("correct (y/n)") or "").strip().lower()[:1]
        j = r["judge"]["answer"]
        tally[(you, j)] += 1
        flag = ""
        if you == "n" and j == "yes":
            flag = "  <- judge MISSED a wrong pin"
        elif you == "y" and j != "yes":
            flag = "  <- judge dropped a good pin"
        print(f"  you {you or '?'}  judge {j:7} {r['address'][:45]}{flag}")
    wrong = tally[("n", "yes")] + tally[("n", "no")] + tally[("n", "unsure")]
    caught = tally[("n", "no")] + tally[("n", "unsure")]
    good = sum(v for (y, _), v in tally.items() if y == "y")
    kept = tally[("y", "yes")]
    print(f"\n  wrong pins caught: {caught} of {wrong}    good pins kept: {kept} of {good}")
    kept_total = tally[("y", "yes")] + tally[("n", "yes")]
    if kept_total:
        print(f"  precision of what stays HIGH: {tally[('y', 'yes')]} of {kept_total} = "
              f"{100 * tally[('y', 'yes')] / kept_total:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RESULTS / "batch_results.json"))
    ap.add_argument("--checked", help="your filled checklist CSV, to compare")
    ap.add_argument("--refresh", action="store_true", help="ignore cached answers")
    a = ap.parse_args()
    try:
        import anthropic
    except ImportError:
        sys.exit("run first:  pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic()

    results = json.loads(Path(a.file).read_text())
    import footprints as FP
    results = RG.regrade(results, 4.0, use_outlines=FP.available())
    before = Counter(r["tier"] for r in results)
    print(f"HIGH pins to judge: {before['HIGH']}  (model {MODEL})")
    n = judge(results, client, a.refresh)
    after = Counter(r["tier"] for r in results)
    print(f"\nnew judge calls: {n}   HIGH before {before['HIGH']} -> after {after['HIGH']}")

    out = Path(a.file).with_name("batch_results_judged.json")
    out.write_text(json.dumps(results, indent=2))
    high = [r for r in results if r["tier"] == "HIGH"]
    RG.write_checklist(high, {}, Path(a.file).with_name("high_checklist_judged.csv"))
    print(f"written {out}")
    if a.checked:
        compare(results, a.checked)


if __name__ == "__main__":
    main()
