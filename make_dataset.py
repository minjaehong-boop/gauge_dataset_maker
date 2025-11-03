import cv2
import numpy as np
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_images", type=int)
parser.add_argument("gauge_paths", type=str, nargs="+")
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--plate_path", type=str, default="plate.png")
ARGS = parser.parse_args()

GAUGE_ORDER = {"전압": 0, "전류": 1, "전력": 2, "역률": 3}
NUM_SLOTS = 4


def get_sort_key(path):
    gauge_type = os.path.dirname(path)
    return GAUGE_ORDER.get(gauge_type, 99)


def generate_dynamic_slots(plate_width, plate_height):
    vertical_margin = int(plate_height * 0.1)
    slot_h = plate_height - (2 * vertical_margin)
    slot_w = slot_h
    gap_width = int(slot_w * 0.15)
    total_content_width = (NUM_SLOTS * slot_w) + ((NUM_SLOTS - 1) * gap_width)
    if total_content_width > plate_width:
        gap_width = int(plate_width * 0.02)
        slot_w = int((plate_width - (NUM_SLOTS - 1) * gap_width) / NUM_SLOTS)
        total_content_width = (NUM_SLOTS * slot_w) + ((NUM_SLOTS - 1) * gap_width)
    horizontal_margin = int((plate_width - total_content_width) / 2)
    slot_y = vertical_margin
    slots = []
    for i in range(NUM_SLOTS):
        slot_x = horizontal_margin + i * (slot_w + gap_width)
        slots.append({"x": slot_x, "y": slot_y, "width": slot_w, "height": slot_h})
    return slots


def create_composite_dataset(
    gauge_paths, num_images, plate_path, shuffle=False, base_dir="datasets"
):
    plate_img = cv2.imread(plate_path)
    COMP_H, COMP_W, _ = plate_img.shape
    SLOTS_CONFIG = generate_dynamic_slots(COMP_W, COMP_H)
    sorted_gauge_paths = sorted(gauge_paths, key=get_sort_key)
    models = [os.path.basename(path) for path in sorted_gauge_paths]
    primary_folder_name = "".join(models).ljust(4, "_")
    secondary_folder_name = "shuffle" if shuffle else "fix"
    output_dir = os.path.join(base_dir, primary_folder_name, secondary_folder_name)
    train_img_dir = os.path.join(output_dir, "train", "images")
    train_lbl_dir = os.path.join(output_dir, "train", "labels")
    val_img_dir = os.path.join(output_dir, "val", "images")
    val_lbl_dir = os.path.join(output_dir, "val", "labels")
    for path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(path, exist_ok=True)
    slot_files_map = {}
    for path_part in gauge_paths:
        path = os.path.join(base_dir, path_part)
        slot_files_map[path_part] = [
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")
        ]
    for i in range(num_images):
        composite_img = plate_img.copy()
        composite_labels = []
        input_gauges = list(gauge_paths)
        slot_indices_to_use = list(range(len(input_gauges)))
        if shuffle:
            random.shuffle(input_gauges)
            all_available_slots = list(range(len(SLOTS_CONFIG)))
            slot_indices_to_use = random.sample(
                all_available_slots, k=len(input_gauges)
            )
        for i_gauge, gauge_path in enumerate(input_gauges):
            slot_idx = slot_indices_to_use[i_gauge]
            slot_cfg = SLOTS_CONFIG[slot_idx]
            sx, sy, sw, sh = (
                slot_cfg["x"],
                slot_cfg["y"],
                slot_cfg["width"],
                slot_cfg["height"],
            )
            files = slot_files_map[gauge_path]
            crop_img_path = random.choice(files)
            crop_lbl_path = os.path.splitext(crop_img_path)[0] + ".txt"
            img_array = np.fromfile(crop_img_path, np.uint8)
            crop_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            resized_square_crop = cv2.resize(crop_img, (sw, sw))
            offset_y = (sh - sw) // 2
            paste_y_start = sy + offset_y
            paste_y_end = paste_y_start + sw
            composite_img[paste_y_start:paste_y_end, sx : sx + sw] = resized_square_crop
            if os.path.exists(crop_lbl_path):
                labels = np.loadtxt(crop_lbl_path, dtype=np.float32, ndmin=2)
                if labels.shape[0] > 0:
                    cls_ids, nx, ny, nw, nh = (
                        labels[:, 0],
                        labels[:, 1],
                        labels[:, 2],
                        labels[:, 3],
                        labels[:, 4],
                    )
                    CX = sx + nx * sw
                    CY = paste_y_start + ny * sw
                    W, H = nw * sw, nh * sw
                    NX, NY = CX / COMP_W, CY / COMP_H
                    NW, NH = W / COMP_W, H / COMP_H
                    for i_label in range(len(cls_ids)):
                        composite_labels.append(
                            f"{int(cls_ids[i_label])} {NX[i_label]:.6f} {NY[i_label]:.6f} {NW[i_label]:.6f} {NH[i_label]:.6f}"
                        )
        is_val = (i + 1) % 6 == 0
        img_out_dir = val_img_dir if is_val else train_img_dir
        lbl_out_dir = val_lbl_dir if is_val else train_lbl_dir
        plate_name = os.path.splitext(os.path.basename(ARGS.plate_path))[0]
        base_filename = f"{plate_name}_image_{i:05d}"
        cv2.imwrite(os.path.join(img_out_dir, base_filename + ".png"), composite_img)
        with open(
            os.path.join(lbl_out_dir, base_filename + ".txt"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(composite_labels))


if __name__ == "__main__":
    if len(ARGS.gauge_paths) <= NUM_SLOTS:
        create_composite_dataset(
            ARGS.gauge_paths, ARGS.num_images, ARGS.plate_path, shuffle=ARGS.shuffle
        )
