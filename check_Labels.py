import random
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent                      # папка, где лежит скрипт
DST = BASE if (BASE / "images").is_dir() else BASE / "Waste-YOLO"
SPLIT = "train"

out = BASE / "label_check"
out.mkdir(exist_ok=True)

files = sorted((DST / "images" / SPLIT).glob("*.jpg"))
if not files:
    raise SystemExit(f"Не найдено ни одной картинки в {DST / 'images' / SPLIT}")

random.seed(42)
for p in random.sample(files, min(6, len(files))):
    im = Image.open(p).convert("RGB")
    dr = ImageDraw.Draw(im)
    W, H = im.size
    lbl = DST / "labels" / SPLIT / (p.stem + ".txt")
    n = 0
    for line in lbl.read_text().splitlines():
        c, xc, yc, w, h = map(float, line.split())
        dr.rectangle([(xc - w/2)*W, (yc - h/2)*H, (xc + w/2)*W, (yc + h/2)*H],
                     outline=(255, 0, 0), width=4)
        n += 1
    im.save(out / f"{p.stem}_check.jpg")
    print(p.name, "boxes:", n)

print(f"Превью с рамками: {out}")