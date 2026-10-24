"""Chroma key + mask transfer.
  python tools/chroma_key.py key   in.png out.png            # remove flat #FF00FF
  python tools/chroma_key.py mask  day_raw.png night_raw.png out.png   # night cut with the day image's key mask
"""
import sys
import numpy as np
from PIL import Image, ImageFilter

KEY = np.array([255, 0, 255], float)
T0, T1 = 70, 150  # RGB distance: below T0 fully key colour, above T1 fully opaque


def alpha_from_key(rgb):
    d = np.linalg.norm(rgb.astype(float) - KEY, axis=2)
    a = np.clip((d - T0) / (T1 - T0), 0, 1)
    # drop leftover magenta-tinted pixels within 5 px of the keyed area
    f = rgb.astype(int)
    tint = (f[..., 0] - f[..., 1] > 50) & (f[..., 2] - f[..., 1] > 30)
    hole = Image.fromarray(((a < 0.5) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(11))
    return np.where(tint & (np.array(hole) > 0), 0, a)


def despill(rgb, a):
    a3 = np.maximum(a, 1e-3)[..., None]
    out = (rgb.astype(float) - (1 - a3) * KEY) / a3
    return np.clip(out, 0, 255)


def save(rgb, a, path):
    out = np.dstack([rgb, a * 255]).round().astype(np.uint8)
    Image.fromarray(out, "RGBA").save(path)


def main():
    mode = sys.argv[1]
    if mode == "key":
        rgb = np.array(Image.open(sys.argv[2]).convert("RGB"))
        a = alpha_from_key(rgb)
        save(despill(rgb, a), a, sys.argv[3])
    elif mode == "mask":
        day = np.array(Image.open(sys.argv[2]).convert("RGB"))
        night = Image.open(sys.argv[3]).convert("RGB")
        if night.size != (day.shape[1], day.shape[0]):
            night = night.resize((day.shape[1], day.shape[0]), Image.LANCZOS)
        a = alpha_from_key(day)
        save(np.array(night), a, sys.argv[4])
    else:
        sys.exit(__doc__)


main()
