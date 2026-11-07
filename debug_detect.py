r"""Show what the detector sees in one saved evidence image.
    python debug_detect.py results\<address-folder>\cam0_google_z2_fov18.jpg
Prints every box the zero-shot model proposes (any class, very low threshold), and whether the
image itself looks like a real photo or a blank/'no imagery' tile.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

path = Path(sys.argv[1])
img = Image.open(path).convert("RGB")
W, H = img.size
# is it a blank / grey 'no imagery' image?
small = img.resize((32, 32))
px = list(small.getdata())
spread = max(max(p) - min(p) for p in px)
var = sum(sum(p) for p in px) / len(px)
print(f"image {W}x{H}  mean brightness {var/3:.0f}  colour spread {spread}  -> {'LOOKS BLANK/GREY (no imagery?)' if spread < 25 else 'looks like a real photo'}")

import detect
L = detect._load_local()
print("detector backend:", L["kind"])
d = detect.detect_local(img)
print("app verdict:", d)
dr = ImageDraw.Draw(img)
if d.get("visible"):
    x, y, w, h = d["x"] * W, d["y"] * H, d["width"] * W, d["height"] * H
    dr.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline=(0, 255, 0), width=4)
    dr.text((x - w / 2, max(0, y - h / 2 - 14)), f"DOOR {d['confidence']:.2f}", fill=(0, 255, 0))
out = path.with_name(path.stem + "_debug.jpg")
img.save(out)
print("annotated ->", out)
