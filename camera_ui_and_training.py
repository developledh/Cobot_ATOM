#!/usr/bin/env python3
"""myCobot AI workflow helper.

Commands:
- collect: Open camera UI and save snapshots by class hotkeys.
- make-yaml: Generate YOLO data.yaml for vial/glass_substrate classes.
- train: Train YOLO model via ultralytics.
- infer: Real-time inference with optional myCobot actions.
- robot-init: Connect to pymycobot and move to a safe initial pose.
- make-qr: Generate a printable QR marker for camera position checks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import yaml


DEFAULT_QR_TEXT = "MYCOBOT_CAMERA_POSE_V1"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def detect_qr_text(frame: Any) -> tuple[str, Any]:
    detector = cv2.QRCodeDetector()
    text, points, _ = detector.detectAndDecode(frame)
    return text, points


def collect_images(
    output: Path,
    camera_index: int,
    require_qr_text: str | None,
) -> None:
    ensure_dir(output)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: index={camera_index}")

    print("[INFO] Camera UI started")
    print("[INFO] Keys: 1=vial, 2=glass_substrate, 0=background, q=quit")
    if require_qr_text:
        print(f"[INFO] Save requires QR text: {require_qr_text}")

    key_to_class = {
        ord("1"): "vial",
        ord("2"): "glass_substrate",
        ord("0"): "background",
    }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame")
                continue

            qr_text, qr_points = detect_qr_text(frame)
            qr_ok = True
            if require_qr_text:
                qr_ok = qr_text == require_qr_text

            guide = "1:vial  2:glass_substrate  0:background  q:quit"
            cv2.putText(
                frame,
                guide,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if require_qr_text:
                qr_state = f"QR: {'OK' if qr_ok else 'MISSING/MISMATCH'}"
                qr_color = (0, 255, 0) if qr_ok else (0, 0, 255)
                cv2.putText(
                    frame,
                    qr_state,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    qr_color,
                    2,
                    cv2.LINE_AA,
                )

            if qr_points is not None and len(qr_points) > 0:
                pts = qr_points.astype(int).reshape(-1, 2)
                for i in range(len(pts)):
                    p1 = tuple(pts[i])
                    p2 = tuple(pts[(i + 1) % len(pts)])
                    cv2.line(frame, p1, p2, (255, 0, 0), 2)

            cv2.imshow("myCobot Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key in key_to_class:
                if require_qr_text and not qr_ok:
                    print("[SKIP] QR mismatch. Save blocked for stable camera pose collection.")
                    continue

                cls = key_to_class[key]
                filename = f"{cls}_{now_stamp()}.jpg"
                path = output / filename
                cv2.imwrite(str(path), frame)
                print(f"[SAVE] {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def make_yaml(dataset_root: Path, classes: list[str]) -> Path:
    data_yaml = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)},
    }

    out_path = dataset_root / "data.yaml"
    ensure_dir(dataset_root)
    out_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"[INFO] Generated: {out_path}")
    return out_path


def train_yolo(data: Path, model: str, imgsz: int, epochs: int, batch: int) -> None:
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(data=str(data), imgsz=imgsz, epochs=epochs, batch=batch)


def connect_mycobot(port: str, baud: int):
    from pymycobot.mycobot import MyCobot

    mc = MyCobot(port, baud)
    time.sleep(1.0)
    return mc


def robot_init_pose(port: str, baud: int, speed: int) -> None:
    mc = connect_mycobot(port=port, baud=baud)
    # Safe neutral angle example (adjust for your setup).
    safe_angles = [0, -20, -20, 0, 90, 0]
    mc.send_angles(safe_angles, speed)
    print(f"[INFO] Sent safe init pose: {safe_angles} @ speed={speed}")


def maybe_trigger_robot_action(mc, class_name: str, confidence: float) -> None:
    """Basic pymycobot hook.

    Replace this function with calibrated camera-to-robot coordinate conversion.
    """
    if confidence < 0.60:
        return

    if class_name == "vial":
        mc.set_gripper_value(80, 30)
        print("[ROBOT] vial detected -> demo gripper close")
    elif class_name == "glass_substrate":
        mc.set_gripper_value(20, 30)
        print("[ROBOT] glass_substrate detected -> demo gripper open")


def infer_realtime(
    weights: Path,
    camera_index: int,
    conf: float,
    use_mycobot: bool,
    mycobot_port: str,
    mycobot_baud: int,
) -> None:
    from ultralytics import YOLO

    mc = None
    if use_mycobot:
        mc = connect_mycobot(port=mycobot_port, baud=mycobot_baud)
        print(f"[INFO] myCobot connected: {mycobot_port} @ {mycobot_baud}")

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: index={camera_index}")

    print("[INFO] Realtime inference started (press q to quit)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            results = model.predict(source=frame, conf=conf, verbose=False)
            annotated = results[0].plot()

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_idx = int(boxes.conf.argmax().item())
                best_cls = int(boxes.cls[best_idx].item())
                best_conf = float(boxes.conf[best_idx].item())
                class_name = model.names.get(best_cls, str(best_cls))

                if mc is not None:
                    maybe_trigger_robot_action(mc=mc, class_name=class_name, confidence=best_conf)

            cv2.imshow("myCobot Inference", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def make_camera_qr(output: Path, text: str, box_size: int, border: int) -> Path:
    import qrcode

    ensure_dir(output.parent)
    img = qrcode.make(text, box_size=box_size, border=border)
    img.save(output)
    print(f"[INFO] QR generated: {output}")
    print(f"[INFO] QR text: {text}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="myCobot AI helper")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="camera UI for image collection")
    p_collect.add_argument("--output", type=Path, required=True)
    p_collect.add_argument("--camera-index", type=int, default=0)
    p_collect.add_argument(
        "--require-qr-text",
        default=None,
        help="only save frames when this QR text is visible (camera pose consistency)",
    )

    p_yaml = sub.add_parser("make-yaml", help="generate YOLO data.yaml")
    p_yaml.add_argument("--dataset-root", type=Path, required=True)
    p_yaml.add_argument(
        "--classes",
        nargs="+",
        default=["vial", "glass_substrate"],
        help="class names in order",
    )

    p_train = sub.add_parser("train", help="train YOLO with ultralytics")
    p_train.add_argument("--data", type=Path, required=True)
    p_train.add_argument("--model", default="yolo11n.pt")
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch", type=int, default=16)

    p_infer = sub.add_parser("infer", help="realtime inference")
    p_infer.add_argument("--weights", type=Path, required=True)
    p_infer.add_argument("--camera-index", type=int, default=0)
    p_infer.add_argument("--conf", type=float, default=0.4)
    p_infer.add_argument("--use-mycobot", action="store_true")
    p_infer.add_argument("--mycobot-port", default="/dev/ttyUSB0")
    p_infer.add_argument("--mycobot-baud", type=int, default=115200)

    p_robot = sub.add_parser("robot-init", help="init myCobot safe pose")
    p_robot.add_argument("--mycobot-port", default="/dev/ttyUSB0")
    p_robot.add_argument("--mycobot-baud", type=int, default=115200)
    p_robot.add_argument("--speed", type=int, default=30)

    p_qr = sub.add_parser("make-qr", help="generate camera position QR marker")
    p_qr.add_argument("--output", type=Path, default=Path("assets/camera_pose_qr.png"))
    p_qr.add_argument("--text", default=DEFAULT_QR_TEXT)
    p_qr.add_argument("--box-size", type=int, default=12)
    p_qr.add_argument("--border", type=int, default=4)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        collect_images(
            output=args.output,
            camera_index=args.camera_index,
            require_qr_text=args.require_qr_text,
        )
        return 0

    if args.command == "make-yaml":
        make_yaml(dataset_root=args.dataset_root, classes=args.classes)
        return 0

    if args.command == "train":
        train_yolo(
            data=args.data,
            model=args.model,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
        )
        return 0

    if args.command == "infer":
        infer_realtime(
            weights=args.weights,
            camera_index=args.camera_index,
            conf=args.conf,
            use_mycobot=args.use_mycobot,
            mycobot_port=args.mycobot_port,
            mycobot_baud=args.mycobot_baud,
        )
        return 0

    if args.command == "robot-init":
        robot_init_pose(
            port=args.mycobot_port,
            baud=args.mycobot_baud,
            speed=args.speed,
        )
        return 0

    if args.command == "make-qr":
        make_camera_qr(
            output=args.output,
            text=args.text,
            box_size=args.box_size,
            border=args.border,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
