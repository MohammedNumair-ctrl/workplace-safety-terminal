"""
Workplace Safety Sentinel — starter prototype.

This prototype runs a YOLO detector on an image, video, or webcam.
Important: the default COCO model detects general objects (including people),
not PPE compliance. To flag missing helmets/vests, supply a PPE-trained YOLO
model whose class names explicitly represent those conditions.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
INCIDENTS_PATH = ROOT / "incidents.json"

# Only explicit model classes indicating a potential violation are treated as
# safety incidents. Generic "person" detections are not safety violations.
RISK_LABELS = {
    "no_helmet", "no-helmet", "without_helmet", "no_hardhat", "no-hardhat",
    "no_vest", "no-vest", "without_vest", "no_safety_vest",
    "unsafe_behavior", "fall", "fire", "smoke"
}


def load_incidents():
    if not INCIDENTS_PATH.exists():
        return []
    try:
        data = json.loads(INCIDENTS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        raise SystemExit("incidents.json is not valid JSON. Fix it before running.")


def save_incidents(items):
    INCIDENTS_PATH.write_text(
        json.dumps(items, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Workplace Safety Sentinel YOLO prototype")
    parser.add_argument("--source", required=True,
                        help="Path to an image/video, or webcam index such as 0")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO model path/name; use a PPE-trained model for PPE checks")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Detection confidence threshold (0 to 1)")
    parser.add_argument("--device", default="cpu",
                        help="Inference device: cpu, mps (supported Apple Silicon), or cuda:0")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(args.model)
    source = int(args.source) if args.source.isdigit() else args.source
    is_camera = isinstance(source, int)
    path = Path(source) if isinstance(source, str) else None
    is_image = bool(path and path.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    })

    incidents = load_incidents()
    run_stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    new_events = []
    event_keys = set()

    def process_frame(frame, frame_index=None):
        result = model.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
        annotated = result.plot()
        names = result.names
        boxes = result.boxes

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = str(names.get(cls_id, cls_id)).strip().lower().replace(" ", "_")
                confidence = float(box.conf[0].item())
                if label in RISK_LABELS:
                    key = (frame_index, label)
                    # Avoid logging the same class multiple times in one frame.
                    if key not in event_keys:
                        event_keys.add(key)
                        new_events.append({
                            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "run_started_at": run_stamp,
                            "event_type": "potential_safety_risk",
                            "label": label,
                            "confidence": round(confidence, 4),
                            "source": str(args.source),
                            "frame_index": frame_index,
                            "review_status": "requires_human_review",
                            "note": "Model flag only; verify visually. Not a definitive safety determination."
                        })
        return annotated

    if is_image:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"Could not read image: {path}")
        annotated = process_frame(frame, 0)
        output_path = OUTPUT_DIR / f"{path.stem}_annotated.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            raise SystemExit("Could not write annotated image.")
        print(f"Annotated image saved to: {output_path}")
    else:
        cap_source = source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video/webcam source: {args.source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        output_path = OUTPUT_DIR / ("webcam_annotated.mp4" if is_camera else f"{path.stem}_annotated.mp4")
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise SystemExit("Could not create output video.")
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated = process_frame(frame, frame_index)
                writer.write(annotated)
                cv2.imshow("Workplace Safety Sentinel — press Q to stop", annotated)
                frame_index += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
        print(f"Annotated video saved to: {output_path}")

    incidents.extend(new_events)
    save_incidents(incidents)
    print(f"Potential safety-risk events logged this run: {len(new_events)}")
    print(f"Incident log saved to: {INCIDENTS_PATH}")
    if args.model == "yolo11n.pt":
        print("NOTE: The default COCO model is not a PPE-compliance model.")
        print("Use a verified PPE-trained model to detect missing PPE; do not infer it from person detections.")


if __name__ == "__main__":
    main()
