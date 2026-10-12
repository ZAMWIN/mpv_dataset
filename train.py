"""
Обучение YOLOv8 на датасете Waste-Dumpsites-DroneImagery.
Полная инструкция — в README.md.

Все настройки — в блоке КОНФИГ ниже: правьте под своё железо.
Ориентир для примера: i7-11700 / GTX 1660 Super 6 ГБ / 64 ГБ RAM.
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ----------------------------- КОНФИГ ---------------------------------
MODEL    = "yolov8s.pt"   # модель: yolov8n / s / m / l / x (больше = точнее и медленнее)
EPOCHS   = 100            # максимум эпох (может остановиться раньше из-за PATIENCE)
IMGSZ    = 640            # размер входа, кратно 32: 640 быстрее, 1280 точнее на мелочи
BATCH    = 4              # батч: на 6 ГБ VRAM для s=4, для n=8; OOM -> уменьшить
DEVICE   = 0              # номер GPU или "cpu"
WORKERS  = 8              # процессы загрузки данных (~число ядер CPU)
CACHE    = "ram"          # кэш датасета: "ram" (много RAM) / "disk" / False
AMP      = True           # mixed precision FP16: быстрее и экономит VRAM
PATIENCE = 20             # early stopping: стоп после N эпох без роста val-mAP
RESUME   = False          # True = продолжить прерванный прогон с weights/last.pt
SEED     = 42             # сид для воспроизводимости

# Аугментации под надирную (строго вниз) дрон-съёмку: с дрона нет "верха/низа",
# поэтому повороты и отражения безопасны; scale — много мелких объектов
DEGREES, FLIPUD, FLIPLR, SCALE = 180.0, 0.5, 0.5, 0.5
# -----------------------------------------------------------------------

BASE    = Path(__file__).resolve().parent          # папка скрипта (cwd не важен)
DATA    = BASE / "data.yaml"                       # описание датасета
RUN_DIR = BASE / "runs" / "detect" / "waste_yolov8"
LAST    = RUN_DIR / "weights" / "last.pt"          # чекпоинт для resume
BEST    = RUN_DIR / "weights" / "best.pt"          # лучшие веса по val


def load_data_cfg() -> str:
    """
    Читает data.yaml, подменяет относительный path на абсолютный
    и сохраняет результат в отдельный yaml. Ultralytics 8.4.x
    НЕ принимает dict в аргументе data — только путь к файлу.
    """
    if not DATA.exists():
        raise SystemExit(f"data.yaml не найден: {DATA} (положите рядом с train.py)")

    with open(DATA, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(BASE / "Waste-YOLO")

    resolved = BASE / "data_resolved.yaml"
    with open(resolved, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return str(resolved)

def main():
    """Основная логика обучения."""

    # ------------------------- ОБУЧЕНИЕ / RESUME --------------------------
    if RESUME:
        # Продолжаем прерванный прогон: эпоха, оптимизатор и LR восстановятся
        # из last.pt, обучение пойдёт с эпохи N+1.
        if not LAST.exists():
            raise SystemExit(f"Нет чекпоинта {LAST} — возобновлять нечего")
        print(f">>> resume: продолжаю с {LAST}")
        model = YOLO(str(LAST))
        model.train(resume=True)
    else:
        # Обучение с нуля: COCO-pretrained веса дообучаются на нашем датасете
        data_cfg = load_data_cfg()
        model = YOLO(MODEL)
        model.train(
            data=data_cfg,
            epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
            device=DEVICE, workers=WORKERS, cache=CACHE, amp=AMP,
            patience=PATIENCE, cos_lr=True,
            degrees=DEGREES, flipud=FLIPUD, fliplr=FLIPLR, scale=SCALE,
            project=str(BASE / "runs" / "detect"), name=RUN_DIR.name,
            exist_ok=True, seed=SEED,
        )

    # --------------------- ФИНАЛЬНАЯ ОЦЕНКА НА TEST -----------------------
    # test-сплит трогаем один раз в конце — это честная оценка качества
    final = YOLO(str(BEST)) if BEST.exists() else model
    m = final.val(split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE)
    print("Финальные метрики на test-сплите:")
    print(f"  mAP50-95: {m.box.map:.4f}   mAP50: {m.box.map50:.4f}   mAP75: {m.box.map75:.4f}")
    print(f"  Precision: {m.box.mp:.4f}   Recall: {m.box.mr:.4f}")


if __name__ == "__main__":
    # Защита от рекурсивного запуска процессов при multiprocessing.spawn
    main()