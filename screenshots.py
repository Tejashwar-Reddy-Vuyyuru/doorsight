#!/usr/bin/env python3
"""
Screenshot mode — no APIs, no keys, no training.

Drop Street View screenshots (jpg/png) into a folder and run:
    python screenshots.py shots/
For every image you get: door or not, where it is in the frame, and an annotated copy
with the door boxed, in shots/annotated/.

Optional: name files like  "<anything>__h123.jpg" where 123 is the compass heading you were
facing when you took the screenshot; it is echoed in the output so you can place the door.
"""
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from detect import detect


def main(folder):
    folder = Path(folder)
    out = folder / "annotated"
    out.mkdir(exist_ok=True)
    rows = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        img = Image.open(p).convert("RGB")
        d = detect(img) or {"visible": False, "reason": "detector error"}
        m = re.search(r"__h(\d+)", p.stem)
        heading = int(m.group(1)) if m else None
        row = {"file": p.name, "heading": heading, **d}
        rows.append(row)
        if d.get("visible"):
            W, H = img.size
            x, y, w, h = d["x"] * W, d["y"] * H, d["width"] * W, d["height"] * H
            dr = ImageDraw.Draw(img)
            dr.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline=(0, 255, 0), width=4)
            dr.text((x - w / 2, max(0, y - h / 2 - 14)), f"door {d['confidence']:.2f}", fill=(0, 255, 0))
            img.save(out / p.name)
            print(f"{p.name:40} DOOR  x={d['x']:.2f}  conf={d['confidence']:.2f}  {d['reason']}")
        else:
            print(f"{p.name:40} no door  ({d.get('reason','')})")
    (folder / "doors.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{sum(1 for r in rows if r.get('visible'))}/{len(rows)} images show a door. Details: {folder/'doors.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python screenshots.py <folder-of-screenshots>")
    main(sys.argv[1])
