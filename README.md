# Waste-YOLO: Детекция несанкционированных свалок с дронов

Проект для обучения YOLOv8 на датасете **Waste-Dumpsites-DroneImagery** — 2115 дрон-снимков незаконных свалок в надирной проекции (1280×1280 px).

---

## 📋 Содержание

- [Структура проекта](#структура-проекта)
- [Требования](#требования)
- [Установка](#установка)
- [Скачивание датасета](#скачивание-датасета)
- [Конвертация датасета](#конвертация-датасета)
- [Проверка GPU](#проверка-gpu)
- [Проверка разметки](#проверка-разметки)
- [Обучение](#обучение)
- [Файлы проекта](#файлы-проекта)
- [.gitignore](#gitignore)
- [Лицензия](#лицензия)

---

## Структура проекта

```
Learning/
├── .venv/                      # виртуальное окружение
├── Waste-YOLO/                 # сконвертированный YOLO-датасет
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── classes.txt
├── waste_dataset/              # исходный датасет (COCO)
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/                       # результаты обучения
│   └── detect/
│       └── waste_yolov8/
│           └── weights/
│               ├── best.pt
│               └── last.pt
├── converter.py                # конвертер COCO → YOLO
├── train.py                    # обучение модели
├── check_gpu.py                # проверка GPU и скорости инференса
├── check_Labels.py             # визуальная проверка разметки
├── data.yaml                   # конфигурация датасета
├── classes.txt                 # список классов
├── requirements.txt            # зависимости
└── yolov8n.pt                  # предобученные веса (скачивается автоматически)
```

---

## Требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| **ОС** | Windows / Linux / macOS | Windows 10/11, Ubuntu 22.04 |
| **Python** | 3.9 | 3.10+ |
| **GPU** | NVIDIA GTX 1060 (6 ГБ) | RTX 3060+ (8+ ГБ VRAM) |
| **RAM** | 16 ГБ | 32–64 ГБ (для `cache='ram'`) |
| **CUDA** | 11.8 | 12.4 |

---

## Установка

### 1. Клонирование и создание окружения

```bash
git clone <ваш-репозиторий>
cd Learning

# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Установка PyTorch с CUDA

Под вашу видеокарту (пример для CUDA 12.4):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> Актуальные команды для других версий CUDA: [pytorch.org/get-started](https://pytorch.org/get-started/locally/)

### 3. Остальные зависимости

```bash
pip install -r requirements.txt
```

**Содержимое `requirements.txt`:**

```
ultralytics>=8.3.0
huggingface_hub>=0.25.0
Pillow>=10.2.0
PyYAML>=6.0.1
tensorboard>=2.17.0
onnx>=1.16.0
onnxruntime-gpu>=1.18.0
```

---

## Скачивание датасета

**Датасет:** [INS-IntelligentNetworkSolutions/Waste-Dumpsites-DroneImagery](https://huggingface.co/datasets/INS-IntelligentNetworkSolutions/Waste-Dumpsites-DroneImagery)

Характеристики: 2115 дрон-снимков, 1280×1280 px, надирная проекция, разметка COCO, лицензия CC-BY-4.0.

### Вариант A: через `huggingface-cli` (рекомендуется)

```bash
# Установка CLI
pip install huggingface_hub

# Скачивание всего датасета
huggingface-cli download --repo-type dataset \
  INS-IntelligentNetworkSolutions/Waste-Dumpsites-DroneImagery \
  --local-dir ./waste_dataset \
  --resume-download
```

### Вариант B: через Python

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="INS-IntelligentNetworkSolutions/Waste-Dumpsites-DroneImagery",
    repo_type="dataset",
    local_dir="./waste_dataset",
    resume_download=True,
)
```

### Вариант C: через веб-интерфейс

1. Откройте [страницу датасета](https://huggingface.co/datasets/INS-IntelligentNetworkSolutions/Waste-Dumpsites-DroneImagery)
2. Перейдите на вкладку **Files and versions**
3. Скачайте папку `data/` (файлы `.parquet`) или отдельные сплиты

### Вариант D: через `datasets`

```python
from datasets import load_dataset

ds = load_dataset(
    "INS-IntelligentNetworkSolutions/Waste-Dumpsites-DroneImagery",
    cache_dir="./waste_dataset"
)
print(ds)
```

### После скачивания

Убедитесь, что структура такая:

```
waste_dataset/
├── train/
│   ├── annotations_coco.json
│   └── *.jpg
├── valid/
│   ├── annotations_coco.json
│   └── *.jpg
└── test/
    ├── annotations_coco.json
    └── *.jpg
```

> Если датасет скачался в формате `.parquet`, распакуйте его в COCO-структуру. Оригинальный датасет на Hugging Face содержит и COCO-аннотации (`annotations_coco.json`) в каждой папке сплита.

---

## Конвертация датасета

Скрипт `converter.py` преобразует COCO-разметку в формат YOLOv8.

```bash
python converter.py
```

**Что делает:**

- Ищет `annotations_coco.json` в `waste_dataset/{train,valid,test}/`
- Приводит имена Roboflow (`NAME_jpg.rf.<hash>.jpg`) к исходным (`NAME.jpg`)
- Пропускает вырожденные боксы (`w <= 0` или `h <= 0`)
- Кадры без объектов получают пустой `.txt` (фоновые примеры)
- Повторный запуск безопасен — файлы не копируются заново

**Результат:**

```
Waste-YOLO/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── classes.txt
```

**Самопроверка:** скрипт выводит количество картинок и меток в каждом сплите. Они должны совпадать.

---

## Проверка GPU

```bash
python check_gpu.py
```

**Что проверяет:**

- Видимость CUDA и версию драйвера
- Объём VRAM и рекомендации по `imgsz` / `batch`
- Реальную скорость инференса на 10 кадрах из `Waste-YOLO/images/val`
- Пиковое потребление VRAM

**Пример вывода:**

```
GPU: NVIDIA GeForce GTX 1660 SUPER | VRAM: 6.0 ГБ | Compute: 7.5
Рекомендация: imgsz=896, batch=4 (yolov8s) или batch=8 (yolov8n), cache='ram'
Инференс: 45 мс/кадр (~22.2 FPS)
Пик VRAM: 1.23 ГБ
```

---

## Проверка разметки

```bash
python check_Labels.py
```

**Что делает:**

- Выбирает 6 случайных картинок из `train`-сплита (seed=42)
- Рисует рамки из `.txt`-файлов разметки
- Сохраняет превью в `label_check/`

Откройте `label_check/*_check.jpg` и убедитесь, что рамки совпадают с объектами.

---

## Обучение

```bash
python train.py
```

### Настройки в `train.py` (блок КОНФИГ)

| Параметр | Значение | Описание |
|----------|----------|----------|
| `MODEL` | `yolov8s.pt` | Модель: n/s/m/l/x |
| `EPOCHS` | `100` | Максимум эпох |
| `IMGSZ` | `640` | Размер входа (кратно 32) |
| `BATCH` | `4` | Батч (OOM → уменьшить) |
| `DEVICE` | `0` | Номер GPU или `"cpu"` |
| `WORKERS` | `8` | Процессы загрузки (~ядра CPU) |
| `CACHE` | `"ram"` | Кэш датасета (`ram`/`disk`/`False`) |
| `AMP` | `True` | Mixed precision FP16 |
| `PATIENCE` | `20` | Early stopping |
| `RESUME` | `False` | Продолжить прерванный прогон |

### Рекомендации по VRAM

| VRAM | imgsz | batch (yolov8s) | batch (yolov8n) |
|------|-------|------------------|------------------|
| ≤ 6.5 ГБ | 896 | 4 | 8 |
| ≤ 10 ГБ | 1024 | 8 | — |
| > 10 ГБ | 1280 | 8+ | — |

### Аугментации

Под надирную съёмку с дрона (нет «верха/низа»):

- `DEGREES = 180.0` — повороты
- `FLIPUD = 0.5` — отражение по вертикали
- `FLIPLR = 0.5` — отражение по горизонтали
- `SCALE = 0.5` — масштабирование (много мелких объектов)

### Resume

Установите `RESUME = True` в `train.py`, если обучение прервалось. Скрипт продолжит с `runs/detect/waste_yolov8/weights/last.pt`.

### Финальная оценка

После обучения скрипт автоматически запускает `val(split="test")` и выводит метрики:

```
Финальные метрики на test-сплите:
  mAP50-95: 0.XXXX   mAP50: 0.XXXX   mAP75: 0.XXXX
  Precision: 0.XXXX   Recall: 0.XXXX
```

---

## Файлы проекта

| Файл | Назначение |
|------|-----------|
| `converter.py` | Конвертация COCO → YOLOv8 |
| `train.py` | Обучение YOLOv8 |
| `check_gpu.py` | Проверка GPU и скорости инференса |
| `check_Labels.py` | Визуальная проверка разметки |
| `data.yaml` | Конфигурация датасета для Ultralytics |
| `classes.txt` | Список классов (`waste`) |
| `requirements.txt` | Зависимости Python |

### `data.yaml`

```yaml
path: ./Waste-YOLO
train: images/train
val: images/val
test: images/test

names:
  0: waste
```
