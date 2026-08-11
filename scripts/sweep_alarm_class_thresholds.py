from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds for one alarm class")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "valid", "test"])
    parser.add_argument("--alarm-class", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-conf", type=float, default=0.01)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--negative-dir", default=None)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-fpr", type=float, default=0.03)
    parser.add_argument("--output", default="runs/alarm_class_threshold_sweep.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root = Path(data["path"])
    split_key = "val" if args.split == "valid" else args.split
    images_dir = root / data[split_key]
    labels_dir = Path(str(images_dir).replace("/images", "/labels").replace("\\images", "\\labels"))
    names = {int(k): v for k, v in data["names"].items()} if isinstance(data["names"], dict) else dict(enumerate(data["names"]))
    alarm_id = next(idx for idx, name in names.items() if name == args.alarm_class)

    model = YOLO(args.weights)
    samples = []
    for image_path in iter_images(images_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        samples.append(
            {
                "gt": read_yolo_boxes(labels_dir / f"{image_path.stem}.txt", alarm_id, width, height),
                "pred": predict_boxes(model, image_path, alarm_id, args.imgsz, args.min_conf, args.device),
            }
        )

    negative_predictions = []
    if args.negative_dir:
        for image_path in iter_images(Path(args.negative_dir)):
            negative_predictions.append(predict_boxes(model, image_path, alarm_id, args.imgsz, args.min_conf, args.device))

    thresholds = sorted(set(args.thresholds or default_thresholds()))
    rows = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for sample in samples:
            gt_boxes = sample["gt"]
            pred_boxes = [box for box in sample["pred"] if box["conf"] >= threshold]
            matched = set()
            for pred in sorted(pred_boxes, key=lambda item: item["conf"], reverse=True):
                best_iou = 0.0
                best_idx = None
                for idx, gt in enumerate(gt_boxes):
                    if idx in matched:
                        continue
                    iou = box_iou(pred["xyxy"], gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx is not None and best_iou >= args.iou_match:
                    tp += 1
                    matched.add(best_idx)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched)

        negative_images = len(negative_predictions)
        false_positive_frames = sum(
            1 for predictions in negative_predictions if any(box["conf"] >= threshold for box in predictions)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        fpr = false_positive_frames / negative_images if negative_images else None
        accepted = (
            precision >= args.target_precision
            and recall >= args.target_recall
            and (fpr is None or fpr <= args.max_fpr)
        )
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "negative_images": negative_images,
                "false_positive_frames": false_positive_frames if negative_images else None,
                "false_positive_rate": fpr,
                "f1": f1_score(precision, recall),
                "accepted": accepted,
            }
        )

    best = choose_best_row(rows, max_fpr=args.max_fpr)
    output = {
        "alarm_class": args.alarm_class,
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "iou_match": args.iou_match,
        "targets": {
            "precision": args.target_precision,
            "recall": args.target_recall,
            "max_false_positive_rate": args.max_fpr,
        },
        "best": best,
        "accepted": bool(best and best["accepted"]),
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def iter_images(path: Path):
    for suffix in IMAGE_SUFFIXES:
        yield from path.rglob(f"*{suffix}")


def default_thresholds() -> list[float]:
    return [
        0.05,
        0.08,
        0.10,
        0.12,
        0.15,
        0.18,
        0.20,
        0.22,
        0.25,
        0.28,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]


def choose_best_row(rows: list[dict], *, max_fpr: float) -> dict | None:
    if not rows:
        return None
    accepted = [row for row in rows if row["accepted"]]
    if accepted:
        return max(accepted, key=lambda row: (row["recall"], row["precision"], -row_fpr(row), row["f1"]))
    fpr_ok = [row for row in rows if row["false_positive_rate"] is None or row["false_positive_rate"] <= max_fpr]
    candidates = fpr_ok or rows
    return max(candidates, key=lambda row: (row["f1"], row["recall"], row["precision"], -row_fpr(row)))


def row_fpr(row: dict) -> float:
    fpr = row["false_positive_rate"]
    return float(fpr) if fpr is not None else 0.0


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def read_yolo_boxes(label_path: Path, class_id: int, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != class_id:
            continue
        cx, cy, bw, bh = map(float, parts[1:])
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def predict_boxes(model, image_path: Path, class_id: int, imgsz: int, conf: float, device: str):
    results = model.predict(str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)
    boxes = []
    if not results or results[0].boxes is None:
        return boxes
    for box in results[0].boxes:
        if int(box.cls[0].detach().cpu().item()) != class_id:
            continue
        xyxy = tuple(map(float, box.xyxy[0].detach().cpu().tolist()))
        boxes.append({"xyxy": xyxy, "conf": float(box.conf[0].detach().cpu().item())})
    return boxes


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


if __name__ == "__main__":
    main()
