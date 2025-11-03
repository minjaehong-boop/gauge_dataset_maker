import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

parser = argparse.ArgumentParser(description="Detect, classify, and crop gauges from a video.")
parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the input video file.")
parser.add_argument("-m", "--model", type=Path, default=Path('./model/sample.onnx'), help="Path to the ONNX model file.")
parser.add_argument("-c", "--config", type=Path, default=Path('./config/config.json'), help="Path to the configuration file.")
args = parser.parse_args()

if not args.config.exists():
    raise FileNotFoundError(f"Configuration file not found at {args.config}")
if not args.model.exists():
    raise FileNotFoundError(f"Model file not found at {args.model}")

config = json.loads(args.config.read_text(encoding='utf-8'))

CLASSES = config['class']
GAUGE_DEFINITIONS = config.get('standard', [])
CONF_THRESHOLD = config.get('conf_threshold', 0.5)
NMS_THRESHOLD = config.get('nms_threshold', 0.4)

ORT_SESSION = onnxruntime.InferenceSession(str(args.model))
INPUT_INFO = ORT_SESSION.get_inputs()[0]
INPUT_NAME = INPUT_INFO.name
INPUT_SHAPE = (INPUT_INFO.shape[2], INPUT_INFO.shape[3])

def preprocess(image):
    """Resizes and pads an image to the model's input size."""
    input_height, input_width = INPUT_SHAPE
    original_h, original_w, _ = image.shape
    scale = min(input_height / original_h, input_width / original_w)
    resized_w, resized_h = int(original_w * scale), int(original_h * scale)
    
    resized_img = cv2.resize(image, (resized_w, resized_h))

    padded_img = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded_img[:resized_h, :resized_w] = resized_img

    input_tensor = np.expand_dims(padded_img.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
    return input_tensor, scale

def postprocess(output, scale):
    """Applies NMS to model output and scales boxes back to original image size."""
    predictions = np.squeeze(output).T
    
    scores = np.max(predictions[:, 4:], axis=1)
    mask = scores > CONF_THRESHOLD
    
    predictions = predictions[mask]
    scores = scores[mask]

    if predictions.shape[0] == 0:
        return [], []

    class_ids = np.argmax(predictions[:, 4:], axis=1)
    
    boxes = predictions[:, :4]
    boxes /= scale
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
    
    if not hasattr(indices, '__len__'):
        return [], []

    final_indices = indices.flatten()
    return boxes[final_indices].astype(int).tolist(), class_ids[final_indices].tolist()

def group_and_classify(boxes, class_ids):
    """Groups objects by gauge and classifies them based on content, optimized for speed."""
    if not boxes or not GAUGE_DEFINITIONS:
        return []

    gauge_class_id = CLASSES.index('Gauge')
    class_to_id = {name: i for i, name in enumerate(CLASSES)}
    rules_as_ids = [
        {
            'req_ids': {class_to_id[rule['conditions']['unit']], class_to_id[rule['conditions']['max_val']]},
            'type': rule['type'],
            'model': rule['model']
        }
        for rule in GAUGE_DEFINITIONS
    ]

    all_boxes = np.array(boxes)
    all_cids = np.array(class_ids)

    gauge_mask = all_cids == gauge_class_id
    gauge_boxes = all_boxes[gauge_mask]
    other_boxes = all_boxes[~gauge_mask]
    other_cids = all_cids[~gauge_mask]

    if gauge_boxes.shape[0] == 0 or other_boxes.shape[0] == 0:
        return []

    obj_centers_x = other_boxes[:, 0] + other_boxes[:, 2] / 2
    gauge_x1 = gauge_boxes[:, np.newaxis, 0]
    gauge_x2 = gauge_x1 + gauge_boxes[:, np.newaxis, 2]
    
    containment_matrix = (obj_centers_x >= gauge_x1) & (obj_centers_x < gauge_x2)

    classified_gauges = []
    for i, gauge_box in enumerate(gauge_boxes):
        contained_indices = np.where(containment_matrix[i])[0]
        if contained_indices.size == 0:
            continue

        contained_cids = set(other_cids[contained_indices])
        
        for rule in rules_as_ids:
            if rule['req_ids'].issubset(contained_cids):
                classified_gauges.append({
                    'box': gauge_box.tolist(),
                    'type': rule['type'],
                    'model': rule['model'],
                    'objects': [{'box': other_boxes[j].tolist(), 'cid': other_cids[j].item()} for j in contained_indices]
                })
                break
    return classified_gauges

def save_results(frame, gauge_info, frame_idx, video_basename):
    """Saves a cropped gauge image and its YOLO label file."""
    output_dir = Path('datasets') / gauge_info['type'] / gauge_info['model']
    output_dir.mkdir(parents=True, exist_ok=True)

    gx, gy, gw, gh = gauge_info['box']
    if gw <= 0 or gh <= 0: return

    gauge_crop = frame[gy:gy+gh, gx:gx+gw]
    
    yolo_labels = []
    for obj in gauge_info['objects']:
        x, y, w, h = obj['box']
        cx = (x - gx + w / 2) / gw
        cy = (y - gy + h / 2) / gh
        nw, nh = w / gw, h / gh
        yolo_labels.append(f"{obj['cid']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    yolo_labels.append(f"{CLASSES.index('Gauge')} 0.5 0.5 1.0 1.0")

    base_filename = f"{video_basename}_frame_{frame_idx:05d}"
    cv2.imwrite(str(output_dir / f"{base_filename}.png"), gauge_crop)
    (output_dir / f"{base_filename}.txt").write_text("\n".join(yolo_labels), encoding='utf-8')

def main():
    """Main video processing loop."""
    video_path = args.input
    if not video_path.exists():
        print(f"ERROR: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video -> {video_path}")
        return
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        
        input_tensor, scale = preprocess(frame)
        output = ORT_SESSION.run(None, {INPUT_NAME: input_tensor})
        boxes, class_ids = postprocess(output[0], scale)
        
        classified_gauges = group_and_classify(boxes, class_ids)

        if classified_gauges:
            for gauge_info in classified_gauges:
                save_results(frame, gauge_info, frame_idx, video_path.stem)
        
        frame_idx += 1

    cap.release()

if __name__ == '__main__':
    main()
