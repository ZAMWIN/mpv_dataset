"""Проверка GPU: видимость, VRAM, драйвер, реальная скорость на ваших кадрах."""
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent

def hr(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)

hr("СИСТЕМА")
print("OS:", platform.platform())
print("Python:", sys.version.split()[0])
print("torch:", torch.__version__, "| cuDNN:", torch.backends.cudnn.version())

hr("CUDA")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 2 ** 30
    print(f"GPU: {p.name} | VRAM: {vram:.1f} ГБ | Compute: {p.major}.{p.minor}")
    print("CUDA runtime:", torch.version.cuda)
    if vram <= 6.5:
        print("Рекомендация: imgsz=896, batch=4 (yolov8s) или batch=8 (yolov8n), cache='ram'")
    elif vram <= 10:
        print("Рекомендация: imgsz=1024, batch=8 (yolov8s), cache='ram'")
    else:
        print("Рекомендация: imgsz=1280, batch=8+ (yolov8m), cache='ram'")
else:
    print("!!! CUDA не видна — обучение пойдёт на CPU (очень медленно). Проверьте драйвер.")

smi = shutil.which("nvidia-smi")
if smi:
    hr("nvidia-smi")
    print(subprocess.run([smi], capture_output=True, text=True).stdout)

hr("Тест скорости на ваших кадрах (yolov8n @ 896)")
if torch.cuda.is_available():
    imgs = sorted((BASE / "Waste-YOLO" / "images" / "val").glob("*.jpg"))[:10]
    if imgs:
        from ultralytics import YOLO
        torch.cuda.reset_peak_memory_stats()
        model = YOLO("yolov8n.pt")
        model.predict(imgs[0], device=0, imgsz=896, verbose=False)   # прогрев
        t0 = time.time()
        for im in imgs:
            model.predict(im, device=0, imgsz=896, verbose=False)
        dt = (time.time() - t0) / len(imgs)
        print(f"Инференс: {dt*1000:.0f} мс/кадр (~{1/dt:.1f} FPS)")
        print(f"Пик VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} ГБ")
    else:
        print("Не найдены картинки Waste-YOLO/images/val")
else:
    print("Пропущено: нет CUDA")