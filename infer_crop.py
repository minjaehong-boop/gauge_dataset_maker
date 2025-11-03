import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=Path, required=True)
parser.add_argument("-m", "--model", type=Path, default=Path("./model/sample.onnx"))
parser.add_argument("-c", "--config", type=Path, default=Path("./config/config.json"))
args = parser.parse_args()

config = json.loads(args.config.read_text(encoding="utf-8"))

CLASSES = config["class"]
GAUGE_DEFINITIONS = config.get("standard", [])
CONF_THRESHOLD = config.get("conf_threshold", 0.5)
NMS_THRESHOLD = config.get("nms_threshold", 0.4)

ORT_SESSION = onnxruntime.InferenceSession(str(args.model))
INPUT_INFO = ORT_SESSION.get_inputs()[0]
INPUT_NAME = INPUT_INFO.name
INPUT_SHAPE = (INPUT_INFO.shape[2], INPUT_INFO.shape[3])

def preprocess(image):
    ih, iw = INPUT_SHAPE
    oh, ow, _ = image.shape
    scale = min(ih / oh, iw / ow)
    rw, rh = int(ow * scale), int(oh * scale)
    resized = cv2.resize(image, (rw, rh))
    padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
    padded[:rh, :rw] = resized
    x = np.expand_dims(padded.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
    return x, scale

def postprocess(output, scale):
    preds = np.squeeze(output).T
    scores = np.max(preds[:, 4:], axis=1)
    mask = scores > CONF_THRESHOLD
    preds = preds[mask]
    scores = scores[mask]
    if preds.shape[0] == 0:
        return [], []
    class_ids = np.argmax(preds[:, 4:], axis=1)
    boxes = preds[:, :4]
    boxes /= scale
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
    if not hasattr(idx, "__len__"):
        return [], []
    idx = idx.flatten()
    return boxes[idx].astype(int).tolist(), class_ids[idx].tolist()

def group_and_classify(boxes, class_ids):
    if not boxes or not GAUGE_DEFINITIONS:
        return []
    gauge_id = CLASSES.index("Gauge")
    class_to_id = {n: i for i, n in enumerate(CLASSES)}
    rules = [
        {
            "req_ids": {class_to_id[r["conditions"]["unit"]], class_to_id[r["conditions"]["max_val"]]},
            "type": r["type"],
            "model": r["model"],
        }
        for r in GAUGE_DEFINITIONS
    ]
    b = np.array(boxes)
    c = np.array(class_ids)
    gmask = c == gauge_id
    gboxes = b[gmask]
    oboxes = b[~gmask]
    ocids = c[~gmask]
    if gboxes.shape[0] == 0 or oboxes.shape[0] == 0:
        return []
    obj_cx = oboxes[:, 0] + oboxes[:, 2] / 2
    gx1 = gboxes[:, np.newaxis, 0]
    gx2 = gx1 + gboxes[:, np.newaxis, 2]
    contain = (obj_cx >= gx1) & (obj_cx < gx2)
    out = []
    for i, g in enumerate(gboxes):
        idx = np.where(contain[i])[0]
        if idx.size == 0:
            continue
        cidset = set(ocids[idx])
        for r in rules:
            if r["req_ids"].issubset(cidset):
                out.append(
                    {
                        "box": g.tolist(),
                        "type": r["type"],
                        "model": r["model"],
                        "objects": [{"box": oboxes[j].tolist(), "cid": ocids[j].item()} for j in idx],
                    }
                )
                break
    return out

def save_results(frame, gauge_info, frame_idx, video_basename):
    out_dir = Path("datasets") / gauge_info["type"] / gauge_info["model"]
    out_dir.mkdir(parents=True, exist_ok=True)
    gx, gy, gw, gh = gauge_info["box"]
    if gw <= 0 or gh <= 0:
        return
    crop = frame[gy : gy + gh, gx : gx + gw]
    labels = []
    for obj in gauge_info["objects"]:
        x, y, w, h = obj["box"]
        cx = (x - gx + w / 2) / gw
        cy = (y - gy + h / 2) / gh
        nw, nh = w / gw, h / gh
        labels.append(f"{obj['cid']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    labels.append(f"{CLASSES.index('Gauge')} 0.5 0.5 1.0 1.0")
    base = f"{video_basename}_frame_{frame_idx:05d}"
    cv2.imwrite(str(out_dir / f"{base}.png"), crop)
    (out_dir / f"{base}.txt").write_text("\n".join(labels), encoding="utf-8")

def main():
    cap = cv2.VideoCapture(str(args.input))
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        x, scale = preprocess(frame)
        y = ORT_SESSION.run(None, {INPUT_NAME: x})[0]
        boxes, cids = postprocess(y, scale)
        for g in group_and_classify(boxes, cids):
            save_results(frame, g, frame_idx, args.input.stem)
        frame_idx += 1
    cap.release()

if __name__ == "__main__":
    main()
