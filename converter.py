"""
Конвертер датасета Waste-Dumpsites-DroneImagery (COCO -> YOLOv8).

Вход : waste_dataset/{train,valid,test}/annotations_coco.json + *.jpg
Выход: Waste-YOLO/{images,labels}/{train,val,test}/...

Особенности:
 - имена Roboflow из json (NAME_jpg.rf.<hash>.jpg) автоматически
   приводятся к исходным именам файлов на диске (NAME.jpg);
 - битые/вырожденные боксы пропускаются;
 - кадры без объектов получают пустой .txt (фоновые примеры);
 - повторный запуск безопасен (файлы не копируются заново).
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent      # папка, где лежит converter.py
SRC = HERE / "waste_dataset"                # папка со сплитами train/valid/test
DST = HERE / "Waste-YOLO"                   # куда складываем YOLO-датасет
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

# Roboflow: "DJI_0821_1_2_TC_jpg.rf.abcd1234....jpg" -> "DJI_0821_1_2_TC.jpg"
RF_RE = re.compile(r"^(.+)_jpg\.rf\.[0-9a-fA-F]+\.jpg$", re.IGNORECASE)


def clean_name(name: str) -> str:
    """Убирает roboflow-суффикс _jpg.rf.<hash> из имени файла."""
    base = Path(name).name
    m = RF_RE.match(base)
    return (m.group(1) + ".jpg") if m else base


def find_ann_file(split: str) -> Path:
    """Ищет annotations_coco.json сплита: сначала в SRC, иначе rglob по проекту."""
    direct = SRC / split / "annotations_coco.json"
    if direct.exists():
        return direct
    candidates = sorted(HERE.rglob(f"{split}/annotations_coco.json"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Не найден annotations_coco.json для сплита '{split}'. "
        f"Распакуйте RAR в папку проекта: {HERE}"
    )


def main() -> None:
    names: list[str] = []
    grand_img = grand_box = 0

    for src_split, dst_split in SPLIT_MAP.items():
        ann_file = find_ann_file(src_split)
        src_dir = ann_file.parent
        coco = json.loads(ann_file.read_text(encoding="utf-8"))
        print(f"[{src_split}] {ann_file}")
        print(f"    images={len(coco['images'])}, boxes={len(coco['annotations'])}, "
              f"cats={[c['name'] for c in coco['categories']]}")

        # --- классы ---
        for c in coco["categories"]:
            if c["name"] not in names:
                names.append(c["name"])
        cat2idx = {c["id"]: names.index(c["name"]) for c in coco["categories"]}

        sizes = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
        fnames = {im["id"]: clean_name(im["file_name"]) for im in coco["images"]}

        # --- боксы, сгруппированные по изображению (class xc yc w h, нормированные) ---
        boxes = defaultdict(list)
        skipped = 0
        for a in coco["annotations"]:
            img_id = a["image_id"]
            if img_id not in sizes:
                skipped += 1
                continue
            W, H = sizes[img_id]
            x, y, w, h = a["bbox"]                 # COCO: x,y = левый верхний угол
            x, y = max(x, 0.0), max(y, 0.0)
            w, h = min(w, W - x), min(h, H - y)
            if w <= 0 or h <= 0:
                skipped += 1
                continue
            boxes[img_id].append(
                (cat2idx[a["category_id"]], (x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
            )
        if skipped:
            print(f"    пропущено вырожденных боксов: {skipped}")

        img_dir = DST / "images" / dst_split
        lbl_dir = DST / "labels" / dst_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # --- где искать картинки (по приоритету) ---
        search_dirs = [src_dir, SRC / src_split, SRC / "images" / "waste", HERE / "images" / "waste"]
        search_dirs += [d for d in (SRC / s for s in SPLIT_MAP) if d not in search_dirs]

        def locate(name: str):
            for d in search_dirs:
                p = d / name
                if p.exists():
                    return p
            return None

        n_img = n_box = n_miss = 0
        for img_id, name in fnames.items():
            dst_img = img_dir / name
            if not dst_img.exists():
                src_img = locate(name)
                if src_img is None:
                    n_miss += 1
                    continue
                shutil.copy2(src_img, dst_img)
            # метка (пустой файл = фоновый кадр)
            with open(lbl_dir / (Path(name).stem + ".txt"), "w", encoding="utf-8") as f:
                for b in boxes.get(img_id, []):
                    f.write("%d %.6f %.6f %.6f %.6f\n" % b)
                    n_box += 1
            n_img += 1

        print(f"    -> {dst_split}: copied={n_img}, boxes={n_box}, missing={n_miss}")
        grand_img += n_img
        grand_box += n_box

    (DST / "classes.txt").write_text("\n".join(names), encoding="utf-8")
    print(f"\nГотово: images={grand_img}, boxes={grand_box}, classes={names}")
    print(f"Результат: {DST}\n")

    # --- самопроверка: картинок и меток поровну ---
    for dst_split in SPLIT_MAP.values():
        ni = len(list((DST / "images" / dst_split).glob("*.jpg")))
        nl = len(list((DST / "labels" / dst_split).glob("*.txt")))
        flag = "OK" if ni == nl else "MISMATCH!"
        print(f"  {dst_split}: images={ni}, labels={nl} [{flag}]")


if __name__ == "__main__":
    main()