"""Transparent top-layer HUD overlay for YOLO detections."""

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit("opencv-python is required. Install it with `pip install opencv-python`.") from exc

try:
    import mss
except ImportError as exc:
    raise SystemExit("mss is required. Install it with `pip install mss`.") from exc

try:
    from PyQt5.QtCore import Qt, QTimer, QRect
    from PyQt5.QtGui import QColor, QFont, QPainter, QPen
    from PyQt5.QtWidgets import QApplication, QWidget
except ImportError as exc:
    raise SystemExit("PyQt5 is required for the overlay. Install it with `pip install pyqt5`.") from exc

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("ultralytics is required. Install it with `pip install ultralytics`.") from exc

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


MonitorDict = Dict[str, int]
MetaDict = Dict[str, float]


def _load_class_names(model: YOLO, data_path: Optional[Path]) -> Dict[int, str]:
    candidates = [getattr(model, "names", None), getattr(getattr(model, "model", None), "names", None)]
    for names in candidates:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {int(i): str(v) for i, v in enumerate(names)}
    if data_path:
        if yaml is None:
            print("Warning: PyYAML is not installed; cannot read class names from data file.", file=sys.stderr)
        else:
            try:
                with data_path.open("r", encoding="utf-8") as handle:
                    data_cfg = yaml.safe_load(handle)
                data_names = data_cfg.get("names") if isinstance(data_cfg, dict) else None
                if isinstance(data_names, dict):
                    return {int(k): str(v) for k, v in data_names.items()}
                if isinstance(data_names, (list, tuple)):
                    return {int(i): str(v) for i, v in enumerate(data_names)}
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to load class names from {data_path}: {exc}", file=sys.stderr)
    return {}


class DetectionWorker(threading.Thread):
    def __init__(
        self,
        weights: Path,
        monitor: MonitorDict,
        update_cb: Callable[[List[Detection], Optional[MetaDict]], None],
        stop_event: threading.Event,
        conf: float,
        frame_interval: float,
        data_path: Optional[Path] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._weights = weights
        self._monitor = dict(monitor)
        self._update_cb = update_cb
        self._stop_event = stop_event
        self._conf = conf
        self._frame_interval = max(0.0, frame_interval)
        self._data_path = data_path
        self._device = device
        self._imgsz = imgsz
        self._class_names: Dict[int, str] = {}
        self._width = int(self._monitor["width"])
        self._height = int(self._monitor["height"])

    def run(self) -> None:
        try:
            model = YOLO(str(self._weights))
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load YOLO weights {self._weights}: {exc}", file=sys.stderr)
            self._stop_event.set()
            return

        self._class_names = _load_class_names(model, self._data_path)
        if not self._class_names:
            print("Warning: could not resolve class names; falling back to class ids.", file=sys.stderr)

        predict_kwargs = {"conf": self._conf, "device": self._device, "verbose": False}
        if self._imgsz:
            predict_kwargs["imgsz"] = self._imgsz

        with mss.mss() as sct:
            monitor = dict(self._monitor)
            while not self._stop_event.is_set():
                start_time = time.perf_counter()
                try:
                    raw = np.array(sct.grab(monitor))
                except Exception as exc:  # noqa: BLE001
                    print(f"Screen capture failed: {exc}", file=sys.stderr)
                    self._stop_event.set()
                    break

                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                try:
                    results = model.predict(frame, **predict_kwargs)
                except Exception as exc:  # noqa: BLE001
                    print(f"Inference failed: {exc}", file=sys.stderr)
                    time.sleep(self._frame_interval or 0.1)
                    continue

                detections: List[Detection] = []
                for batch_result in results or []:
                    boxes = getattr(batch_result, "boxes", None)
                    if boxes is None or boxes.xyxy is None:
                        continue
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    conf = boxes.conf.cpu().numpy()
                    for (x1, y1, x2, y2), cid, score in zip(xyxy, cls, conf):
                        x1_i = int(np.clip(x1, 0, self._width - 1))
                        y1_i = int(np.clip(y1, 0, self._height - 1))
                        x2_i = int(np.clip(x2, 0, self._width - 1))
                        y2_i = int(np.clip(y2, 0, self._height - 1))
                        if x2_i <= x1_i or y2_i <= y1_i:
                            continue
                        label = self._class_names.get(int(cid), f"class_{int(cid)}")
                        detections.append(
                            Detection(
                                label=label,
                                confidence=float(score),
                                bbox=(x1_i, y1_i, x2_i, y2_i),
                            )
                        )

                elapsed = time.perf_counter() - start_time
                meta: MetaDict = {"inference_time": elapsed, "timestamp": time.time()}
                self._update_cb(detections, meta)

                sleep_time = self._frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self._update_cb([], None)


class DetectionOverlay(QWidget):
    def __init__(
        self,
        monitor: MonitorDict,
        weights: Path,
        conf: float,
        frame_interval: float,
        data_path: Optional[Path],
        device: Optional[str],
        imgsz: Optional[int],
        show_fps: bool,
        thickness: int,
        font_size: int,
    ) -> None:
        super().__init__()

        self._monitor = dict(monitor)
        self._detections: List[Detection] = []
        self._meta: MetaDict = {}
        self._lock = threading.Lock()
        self._color_cache: Dict[str, QColor] = {}
        self._show_fps = show_fps
        self._pen_width = max(1, thickness)
        self._stop_event = threading.Event()

        flags = Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        if hasattr(Qt, "WindowTransparentForInput"):
            flags |= Qt.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)

        font = QFont(self.font())
        font.setPointSize(max(6, font_size))
        self.setFont(font)

        self.setGeometry(
            int(self._monitor["left"]),
            int(self._monitor["top"]),
            int(self._monitor["width"]),
            int(self._monitor["height"]),
        )

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.update)
        self._refresh_timer.start(33)

        self._worker = DetectionWorker(
            weights=weights,
            monitor=self._monitor,
            update_cb=self._set_detections,
            stop_event=self._stop_event,
            conf=conf,
            frame_interval=frame_interval,
            data_path=data_path,
            device=device,
            imgsz=imgsz,
        )
        self._worker.start()

    def _set_detections(self, detections: List[Detection], meta: Optional[MetaDict]) -> None:
        with self._lock:
            self._detections = detections
            self._meta = meta or {}

    def _color_for_label(self, label: str) -> QColor:
        color = self._color_cache.get(label)
        if color is None:
            seed = abs(hash(label))
            r = 64 + ((seed & 0xFF) // 2)
            g = 64 + (((seed >> 8) & 0xFF) // 2)
            b = 64 + (((seed >> 16) & 0xFF) // 2)
            color = QColor(r % 256, g % 256, b % 256)
            self._color_cache[label] = color
        return color

    def paintEvent(self, event) -> None:  # noqa: D401
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        with self._lock:
            detections = list(self._detections)
            meta = dict(self._meta)

        for detection in detections:
            color = self._color_for_label(detection.label)
            pen = QPen(color)
            pen.setWidth(self._pen_width)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            x1, y1, x2, y2 = detection.bbox
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.save()
            self._draw_label(painter, detection, color)
            painter.restore()

        if self._show_fps and meta:
            inference_time = meta.get("inference_time", 0.0)
            fps = (1.0 / inference_time) if inference_time else 0.0
            text = f"{len(detections)} objects | {fps:.1f} FPS"
            self._draw_status(painter, text)

    def _draw_label(self, painter: QPainter, detection: Detection, color: QColor) -> None:
        text = f"{detection.label} {detection.confidence * 100:.1f}%"
        metrics = painter.fontMetrics()
        padding = 4
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()
        x1, y1, _, _ = detection.bbox
        rect_y = max(0, y1 - text_height - (padding * 2))
        rect = QRect(x1, rect_y, text_width + (padding * 2), text_height + (padding * 2))
        painter.fillRect(rect, QColor(color.red(), color.green(), color.blue(), 200))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect.adjusted(padding, padding, -padding, -padding), Qt.AlignLeft | Qt.AlignVCenter, text)

    def _draw_status(self, painter: QPainter, text: str) -> None:
        metrics = painter.fontMetrics()
        padding = 6
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()
        rect = QRect(
            padding,
            self.height() - text_height - (padding * 2),
            text_width + (padding * 2),
            text_height + (padding * 2),
        )
        painter.fillRect(rect, QColor(0, 0, 0, 180))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect.adjusted(padding, padding, -padding, -padding), Qt.AlignLeft | Qt.AlignVCenter, text)

    def closeEvent(self, event) -> None:  # noqa: N802,D401
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        if hasattr(self, "_refresh_timer") and self._refresh_timer.isActive():
            self._refresh_timer.stop()
        self._stop_event.set()
        if hasattr(self, "_worker") and self._worker.is_alive():
            self._worker.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detections as a transparent HUD overlay.")
    parser.add_argument("--weights", type=Path, default=Path("runs/detect/train/weights/best.pt"), help="Path to the YOLO weights file.")
    parser.add_argument("--data", type=Path, default=None, help="Optional dataset YAML to load class names.")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index to capture (use --list-monitors to inspect).")
    parser.add_argument("--list-monitors", action="store_true", help="List available monitors and exit.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--frame-interval", type=float, default=0.05, help="Minimum delay between inference cycles (seconds).")
    parser.add_argument("--device", type=str, default=None, help="Torch device identifier such as '0' or 'cpu'.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference image size override.")
    parser.add_argument("--show-fps", action="store_true", help="Display detection count and FPS banner.")
    parser.add_argument("--thickness", type=int, default=3, help="Bounding box line thickness in pixels.")
    parser.add_argument("--font-size", type=int, default=14, help="Overlay font size (points).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_monitors:
        try:
            with mss.mss() as sct:
                for idx, monitor in enumerate(sct.monitors):
                    description = f"{monitor['width']}x{monitor['height']} @ ({monitor['left']},{monitor['top']})"
                    label = "Virtual desktop" if idx == 0 else f"Display {idx}"
                    print(f"[{idx}] {label}: {description}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to enumerate monitors: {exc}", file=sys.stderr)
            return 1
        return 0

    weights = Path(args.weights).expanduser()
    if not weights.exists():
        print(f"Weights file not found: {weights}", file=sys.stderr)
        return 1

    data_path = Path(args.data).expanduser() if args.data else None

    try:
        with mss.mss() as sct:
            monitors = sct.monitors
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to access monitors: {exc}", file=sys.stderr)
        return 1

    if not monitors:
        print("No monitors detected.", file=sys.stderr)
        return 1

    if args.monitor < 0 or args.monitor >= len(monitors):
        print(
            f"Monitor index {args.monitor} is out of range (found {len(monitors)} monitors).",
            file=sys.stderr,
        )
        return 1

    monitor = dict(monitors[args.monitor])

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    overlay = DetectionOverlay(
        monitor=monitor,
        weights=weights,
        conf=args.conf,
        frame_interval=max(0.0, args.frame_interval),
        data_path=data_path,
        device=args.device,
        imgsz=args.imgsz,
        show_fps=args.show_fps,
        thickness=max(1, args.thickness),
        font_size=max(6, args.font_size),
    )
    overlay.show()
    overlay.raise_()

    app.aboutToQuit.connect(overlay.shutdown)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
