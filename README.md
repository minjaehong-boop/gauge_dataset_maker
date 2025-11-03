# Gauge Dataset Maker

A workflow to generate composite YOLO datasets from individual gauge images.

## Core Scripts
* `infer_crop.py`: Runs inference on videos to crop individual gauges based on `config.json`.
* `make_dataset.py`: Combines cropped images onto a background plate (`plate.png`) to generate the final dataset.

## Quickstart Workflow

### 1. Update Config
Modify `config/config.json` with new classes and gauge definitions.

This file tells `infer_crop.py` what to look for. For example, to add Power (P) and Power Factor (PF) gauges (both model 'a') to a config that previously only handled Voltage (V) and Current (I), you would make the following changes:

**Before (`aa__` config):**
```json
{
  "classes": [
    "0", "300", "600",
    "I", "Gauge", "V"
  ],
  "gauge_definitions": [
    { "type": "전압", "model": "a", "conditions": { "unit": "V", "max_val": "600" } },
    { "type": "전류", "model": "a", "conditions": { "unit": "I", "max_val": "300" } }
  ]
}
```
**After (`aaaa` config):**

```json

{
  "classes": [
    "0", "5", "300", 
    "480", "600", "V", 
    "I", "P", "PF", "Gauge"
  ],
  "gauge_definitions": [
    { "type": "전압", "model": "a", "conditions": { "unit": "V", "max_val": "600" } },
    { "type": "전류", "model": "a", "conditions": { "unit": "I", "max_val": "300" } },
    { "type": "전력", "model": "a", "conditions": { "unit": "P", "max_val": "480" } },
    { "type": "역률", "model": "a", "conditions": { "unit": "PF", "max_val": "5" } }
  ]
}
```

### 2\. Crop Gauges

This generates cropped images (e.g., `datasets/전압/a/`).

```bash
python infer_crop.py -i ./inputs/부하변동_영상_0903.mp4
```

### 3\. Generate Dataset

This creates the final dataset (e.g., `datasets/aaaa/fix/`) with a 5:1 train/val split.

```bash
python make_dataset.py 1200 "전압/a" "전류/a" "전력/a" "역률/a"
```

  * `1200`: Total images to generate.
  * `"전압/a" ...`: Source directories for cropped images (order doesn't matter).

## Options

  * `--shuffle`: Randomizes slot positions for each image. Saves to `datasets/aaaa/shuffle/` instead of `fix/`.
  * `--plate_path`: Specify a background image (default: `plate.png`).

<!-- end list -->
