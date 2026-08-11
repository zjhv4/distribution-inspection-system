from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def evaluate_frame_false_positive_rate(
    *,
    model,
    negative_dir: Path | None,
    alarm_classes: set[str] | None,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
) -> dict:
    if negative_dir is None:
        return {
            "mode": "precision_proxy",
            "negative_images": 0,
            "false_positive_frames": None,
            "false_positive_rate": None,
            "explanation": "No negative image directory was supplied.",
        }

    image_paths = sorted(path for path in negative_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise RuntimeError(f"No negative images found under {negative_dir}")

    names = getattr(model.model, "names", {}) or {}
    false_positive_frames = 0
    examples: list[str] = []

    for image_path in image_paths:
        results = model.predict(str(image_path), imgsz=imgsz, device=device, conf=conf, iou=iou, verbose=False)
        has_alarm = False
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0].detach().cpu().item())
                class_name = str(names.get(class_id, class_id))
                if alarm_classes is None or class_name in alarm_classes:
                    has_alarm = True
                    break
        if has_alarm:
            false_positive_frames += 1
            if len(examples) < 20:
                examples.append(str(image_path))

    return {
        "mode": "negative_frame_set",
        "negative_images": len(image_paths),
        "false_positive_frames": false_positive_frames,
        "false_positive_rate": false_positive_frames / len(image_paths),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model against acceptance targets")
    parser.add_argument("--task", choices=["intrusion", "breaker"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--evaluation-role",
        default="development",
        choices=["development", "calibration", "independent_test", "site_holdout"],
        help="Evidence role. independent_test requires a threshold source selected without this split.",
    )
    parser.add_argument(
        "--threshold-source",
        default=None,
        help="Versioned calibration report/config that fixed --conf before an independent test.",
    )
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-fpr", type=float, default=0.03)
    parser.add_argument("--negative-dir", default=None, help="Directory of images that should not trigger alarms")
    parser.add_argument(
        "--alarm-classes",
        nargs="*",
        default=None,
        help="Class names counted as alarm detections for false-positive-rate evaluation",
    )
    parser.add_argument("--output", default="runs/eval_report.json")
    args = parser.parse_args()
    if args.evaluation_role == "independent_test" and not args.threshold_source:
        parser.error("--threshold-source is required when --evaluation-role=independent_test")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        verbose=False,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    fpr_report = evaluate_frame_false_positive_rate(
        model=model,
        negative_dir=Path(args.negative_dir) if args.negative_dir else None,
        alarm_classes=set(args.alarm_classes) if args.alarm_classes else None,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )
    false_positive_rate = fpr_report["false_positive_rate"]
    if false_positive_rate is None:
        false_positive_rate = max(0.0, 1.0 - precision)
        fpr_report["false_positive_rate"] = false_positive_rate
        fpr_report["explanation"] = "Using 1 - precision as a conservative proxy because --negative-dir was not supplied."

    accepted = precision >= args.target_accuracy and recall >= args.target_recall and false_positive_rate <= args.max_fpr

    report = {
        "task": args.task,
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "evaluation_role": args.evaluation_role,
        "threshold": {
            "conf": args.conf,
            "source": args.threshold_source,
        },
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "false_positive_rate": false_positive_rate,
        "false_positive_evidence": fpr_report,
        "targets": {
            "accuracy_precision": args.target_accuracy,
            "recall": args.target_recall,
            "max_false_positive_rate": args.max_fpr,
        },
        "accepted": accepted,
        "note": "如未提供 --negative-dir，误报率使用 1 - precision 的保守代理值；正式验收建议提供独立负样本集。",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
