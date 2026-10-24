import sys
from pathlib import Path
from PIL import Image

src = Path(sys.argv[1] if len(sys.argv) > 1 else "assets_src/cutouts")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/assets/img")
max_w = int(sys.argv[3]) if len(sys.argv) > 3 else 2400
dst.mkdir(parents=True, exist_ok=True)
total = 0
for p in sorted(src.glob("*.png")):
    im = Image.open(p).convert("RGBA")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    out = dst / (p.stem + ".webp")
    im.save(out, "WEBP", quality=82, method=6)
    kb = out.stat().st_size / 1024
    total += kb
    print(f"{out.name}: {im.width}x{im.height}, {kb:.0f} KB")
print(f"total {total:.0f} KB")