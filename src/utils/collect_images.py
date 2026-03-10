import argparse
import json
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config.json"


def ensure_opencv_highgui() -> None:
    build_info = cv2.getBuildInformation()
    if "GUI:                           NONE" in build_info:
        raise RuntimeError(
            "OpenCV was installed without GUI support (likely opencv-python-headless), "
            "so cv2.imshow cannot run.\n"
            "Fix:\n"
            "  uv pip uninstall opencv-python-headless\n"
            "  uv pip install opencv-python"
        )


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    slug = slug.strip("_")
    return slug or "class"


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_classes(config: dict) -> list[str]:
    classes = config.get("classes", [])
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("No classes found in config. Update src/config.json with a non-empty `classes` list.")
    return [str(name).strip() for name in classes]


def ensure_dataset_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "images": output_root / "images",
        "labels": output_root / "labels",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def clamp_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(0, min(width - 1, x2)))
    y2 = int(max(0, min(height - 1, y2)))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def merge_bboxes(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return x1, y1, x2, y2


@dataclass
class CaptureSettings:
    capture_interval_sec: float = 1.2
    stable_frames: int = 6
    min_dwell_sec: float = 0.45
    min_sharpness: float = 80.0
    palm_padding_px: int = 20
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    min_skin_area_ratio: float = 0.015


class MediaPipeHandsDetector:

    def __init__(self, settings: CaptureSettings):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        self.settings = settings

    def detect_hand_bboxes(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return []

        bboxes = []
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            landmarks = hand_landmarks.landmark
            # Use all landmarks so the box includes finger patterns (not palm-only).
            xs = [lm.x * width for lm in landmarks]
            ys = [lm.y * height for lm in landmarks]
            pad = self.settings.palm_padding_px
            x1, y1 = min(xs) - pad, min(ys) - pad
            x2, y2 = max(xs) + pad, max(ys) + pad
            bboxes.append(clamp_bbox(x1, y1, x2, y2, width, height))
        return bboxes

    def close(self):
        self._hands.close()


class SkinColorHandsDetector:
    """Fallback hand detector using skin-color segmentation and up to two largest contours."""

    def __init__(self, settings: CaptureSettings):
        self.settings = settings

    def detect_hand_bboxes(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width, _ = frame_bgr.shape
        frame_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

        ycrcb = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        min_area = float(width * height) * self.settings.min_skin_area_ratio
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        bboxes = []
        for contour in sorted_contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            pad = self.settings.palm_padding_px
            bboxes.append(clamp_bbox(x - pad, y - pad, x + w + pad, y + h + pad, width, height))
            if len(bboxes) == 2:
                break
        return bboxes

    def close(self):
        return None


def create_hand_detector(settings: CaptureSettings):
    if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        print("Detector backend: MediaPipe Hands")
        return MediaPipeHandsDetector(settings)

    print(
        "Detector backend: OpenCV skin-color fallback (mediapipe.solutions unavailable). "
        "For best quality install a full mediapipe build such as 0.10.21."
    )
    return SkinColorHandsDetector(settings)


class AutoCaptureSession:
    def __init__(
        self,
        classes: list[str],
        output_root: Path,
        class_targets: list[int],
        camera_id: int,
        settings: CaptureSettings,
        auto_advance: bool = True,
    ):
        self.classes = classes
        self.output_root = output_root
        self.class_targets = class_targets
        self.settings = settings
        self.auto_advance = auto_advance
        self.paths = ensure_dataset_dirs(output_root)
        self.class_counts = [0 for _ in classes]
        self.current_class_idx = 0
        self.bbox_history = deque(maxlen=max(2, settings.stable_frames))
        self.stable_since = None
        self.last_capture_ts = 0.0
        self.detector = create_hand_detector(settings)
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

    def _advance_if_target_complete(self):
        if not self.auto_advance:
            return
        while (
            self.current_class_idx < len(self.classes) - 1
            and self.class_counts[self.current_class_idx] >= self.class_targets[self.current_class_idx]
        ):
            self.current_class_idx += 1
            self._reset_stability()

    def _all_targets_completed(self) -> bool:
        return all(
            self.class_counts[idx] >= self.class_targets[idx]
            for idx in range(len(self.class_targets))
        )

    def _reset_stability(self):
        self.bbox_history.clear()
        self.stable_since = None

    def _is_stable(self, bbox: tuple[int, int, int, int]) -> bool:
        self.bbox_history.append(bbox)
        if len(self.bbox_history) < self.settings.stable_frames:
            return False

        boxes = np.array(self.bbox_history, dtype=np.float32)
        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0))
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        avg_diag = float(np.mean(np.sqrt(widths * widths + heights * heights)))
        if avg_diag <= 1.0:
            return False

        center_jitter = float(np.mean(np.std(centers, axis=0)))
        size_jitter = float(np.mean(np.std(np.column_stack((widths, heights)), axis=0)))
        return center_jitter < (0.08 * avg_diag) and size_jitter < (0.12 * avg_diag)

    def _is_sharp(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[bool, float]:
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return sharpness >= self.settings.min_sharpness, sharpness

    def _save_sample(self, frame: np.ndarray, bboxes: list[tuple[int, int, int, int]], class_idx: int) -> Path:
        class_name = self.classes[class_idx]
        class_slug = slugify(class_name)
        stem = f"{uuid.uuid4().hex[:10]}-{class_slug}-{int(time.time() * 1000)}"

        image_dir = self.paths["images"]
        label_dir = self.paths["labels"]
        image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}.txt"
        ok = cv2.imwrite(str(image_path), frame)
        if not ok:
            raise RuntimeError(f"Failed to save image: {image_path}")

        h, w, _ = frame.shape
        lines = []
        for x1, y1, x2, y2 in bboxes:
            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        label_path.write_text("".join(lines), encoding="utf-8")

        self.class_counts[class_idx] += 1
        self.last_capture_ts = time.time()
        return image_path

    def _draw_overlay(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
        stable: bool,
        sharp: bool,
        sharpness: float,
        recent_save_text: str,
    ) -> np.ndarray:
        view = frame.copy()
        class_name = self.classes[self.current_class_idx]
        captured = self.class_counts[self.current_class_idx]
        target = self.class_targets[self.current_class_idx]

        status = []
        status.append(f"hands:{len(bboxes)}")
        status.append("stable:yes" if stable else "stable:no")
        status.append("sharp:yes" if sharp else "sharp:no")
        status_text = " | ".join(status)

        cv2.putText(view, f"Class {self.current_class_idx + 1}/{len(self.classes)}: {class_name}", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, f"Captured: {captured}/{target}", (16, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, f"{status_text} | sharpness:{sharpness:.1f}", (16, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(view, "Keys: [n]ext [p]rev [space]force save [q]quit", (16, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (190, 190, 190), 2, cv2.LINE_AA)
        if recent_save_text:
            cv2.putText(view, recent_save_text, (16, 164),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2, cv2.LINE_AA)

        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            color = (0, 200, 255)
            if stable and sharp:
                color = (0, 255, 0)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                view,
                f"hand{idx + 1}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        return view

    def run(self):
        recent_save_text = ""
        recent_save_ts = 0.0

        while True:
            self._advance_if_target_complete()
            if self._all_targets_completed():
                print("All target images captured. Exiting capture session.")
                break

            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from camera")

            now = time.time()
            bboxes = self.detector.detect_hand_bboxes(frame)
            stable = False
            sharp = False
            sharpness = 0.0

            if len(bboxes) == 0:
                self._reset_stability()
            else:
                combined_bbox = merge_bboxes(bboxes)
                stable = self._is_stable(combined_bbox)
                sharp, sharpness = self._is_sharp(frame, combined_bbox)
                if stable and sharp:
                    if self.stable_since is None:
                        self.stable_since = now
                else:
                    self.stable_since = None

                dwell_ready = self.stable_since is not None and (now - self.stable_since) >= self.settings.min_dwell_sec
                interval_ready = (now - self.last_capture_ts) >= self.settings.capture_interval_sec
                target_ready = self.class_counts[self.current_class_idx] < self.class_targets[self.current_class_idx]

                if dwell_ready and interval_ready and target_ready:
                    image_path = self._save_sample(frame, bboxes, self.current_class_idx)
                    recent_save_text = f"saved -> {image_path.name}"
                    recent_save_ts = now

                    self._advance_if_target_complete()

            if recent_save_text and (now - recent_save_ts) > 2.0:
                recent_save_text = ""

            overlay = self._draw_overlay(frame, bboxes, stable, sharp, sharpness, recent_save_text)
            cv2.imshow("Auto Hand Capture", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                self.current_class_idx = min(self.current_class_idx + 1, len(self.classes) - 1)
                self._reset_stability()
            if key == ord("p"):
                self.current_class_idx = max(self.current_class_idx - 1, 0)
                self._reset_stability()
            if key == ord(" "):
                if len(bboxes) > 0 and self.class_counts[self.current_class_idx] < self.class_targets[self.current_class_idx]:
                    image_path = self._save_sample(frame, bboxes, self.current_class_idx)
                    recent_save_text = f"saved(manual) -> {image_path.name}"
                    recent_save_ts = now

        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()

        print("\nCapture summary:")
        for idx, name in enumerate(self.classes):
            print(f"{idx:02d} {name}: {self.class_counts[idx]}/{self.class_targets[idx]} images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture sign-language samples with automatic hand detection (1 or 2 hands) and YOLO label generation."
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--images-per-class", type=int, default=120, help="Target images per class.")
    parser.add_argument("--total-images", type=int, default=None, help="Target total images across all classes.")
    parser.add_argument("--output-root", type=Path, default=None, help="Dataset root path. Defaults to config dataset_root.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.json.")
    parser.add_argument("--capture-interval", type=float, default=1.2, help="Seconds between auto captures.")
    parser.add_argument("--stable-frames", type=int, default=6, help="Consecutive frames needed for stability.")
    parser.add_argument("--min-dwell-sec", type=float, default=0.45, help="Stable duration before capture.")
    parser.add_argument("--min-sharpness", type=float, default=80.0, help="Min Laplacian variance for sharpness.")
    parser.add_argument("--palm-padding", type=int, default=20, help="Padding around detected hand bbox in pixels.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.6, help="MediaPipe min detection confidence.")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.6, help="MediaPipe min tracking confidence.")
    parser.add_argument("--no-auto-advance", action="store_true", help="Disable moving to next class automatically.")
    return parser.parse_args()


def main():
    ensure_opencv_highgui()
    args = parse_args()
    if args.images_per_class <= 0:
        raise ValueError("--images-per-class must be > 0.")
    if args.total_images is not None and args.total_images <= 0:
        raise ValueError("--total-images must be > 0.")

    config = load_config(args.config)
    classes = load_classes(config)
    if args.total_images is None:
        class_targets = [args.images_per_class for _ in classes]
    else:
        class_count = len(classes)
        base = args.total_images // class_count
        remainder = args.total_images % class_count
        class_targets = [base + 1 if idx < remainder else base for idx in range(class_count)]

    dataset_root = args.output_root
    if dataset_root is None:
        dataset_root_name = config.get("dataset_root", "data")
        dataset_root = PROJECT_ROOT / str(dataset_root_name)

    settings = CaptureSettings(
        capture_interval_sec=args.capture_interval,
        stable_frames=args.stable_frames,
        min_dwell_sec=args.min_dwell_sec,
        min_sharpness=args.min_sharpness,
        palm_padding_px=args.palm_padding,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    print(f"Loaded {len(classes)} classes from {args.config}")
    print(f"Dataset root: {dataset_root}")
    print("Classes:")
    for idx, name in enumerate(classes):
        print(f"  {idx:02d}: {name} (target: {class_targets[idx]})")
    print(f"Total target images: {sum(class_targets)}")

    session = AutoCaptureSession(
        classes=classes,
        output_root=dataset_root,
        class_targets=class_targets,
        camera_id=args.camera_id,
        settings=settings,
        auto_advance=not args.no_auto_advance,
    )
    session.run()


if __name__ == "__main__":
    main()