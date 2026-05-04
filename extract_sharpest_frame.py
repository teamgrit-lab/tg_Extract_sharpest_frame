import argparse
import csv
import json
import multiprocessing
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from math import ceil
from pathlib import Path
from time import monotonic
from typing import Callable, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


Logger = Optional[Callable[[str], None]]
CancelChecker = Optional[Callable[[], bool]]
PROGRESS_PREFIX = "[progress] "
SPINNER_FRAMES = "|/-\\"
MULTIPROCESS_PROGRESS_INTERVAL = 0.2
SUPPORTED_OUTPUT_FORMATS = ("png", "jpg")
DEFAULT_YOLO_CLASSES = "person,bicycle,car,motorcycle,bus,truck"
COCO_CLASS_IDS = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


class SharpestFrameError(Exception):
    pass


class SharpestFrameCancelled(SharpestFrameError):
    pass


class GuiTqdm(tqdm):
    def __init__(self, *args, logger: Logger = None, **kwargs) -> None:
        self.logger = logger
        self._last_displayed_message = ""
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        displayed = super().update(n)
        if self.logger is not None:
            self.display()
        return displayed

    def display(self, msg=None, pos=None) -> None:
        if self.logger is None:
            super().display(msg=msg, pos=pos)
            return

        progress_message = msg if msg is not None else str(self)
        cleaned = progress_message.strip()
        if cleaned and cleaned != self._last_displayed_message:
            self._last_displayed_message = cleaned
            emit_progress(cleaned, self.logger)


def emit(message: str, logger: Logger = None) -> None:
    if logger is None:
        print(message)
        return

    logger(message)


def emit_progress(message: str, logger: Logger = None) -> None:
    if logger is None:
        return

    logger(f"{PROGRESS_PREFIX}{message}")


def create_progress(total: Optional[int], desc: str, logger: Logger = None):
    progress_kwargs = {
        "total": total,
        "desc": desc,
        "unit": "frame",
        "dynamic_ncols": True,
        "leave": False,
        "bar_format": "{l_bar}{bar:20}{r_bar}",
    }

    if logger is None:
        return tqdm(**progress_kwargs)

    progress_kwargs["mininterval"] = 0
    progress_kwargs["miniters"] = 1
    progress_kwargs["ascii"] = False
    return GuiTqdm(logger=logger, **progress_kwargs)


def ensure_opencv_available() -> None:
    if cv2 is None:
        raise SharpestFrameError("opencv-python is required. Install it with: pip install -r requirements.txt")


def ensure_tqdm_available() -> None:
    if tqdm is None:
        raise SharpestFrameError("tqdm is required. Install it with: pip install -r requirements.txt")


def raise_if_cancelled(should_cancel: CancelChecker = None) -> None:
    if should_cancel is not None and should_cancel():
        raise SharpestFrameCancelled("Processing was cancelled.")


def get_frame_count(capture) -> Optional[int]:
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        return total_frames
    return None


def load_custom_mask(mask_path: Optional[Path]):
    if mask_path is None:
        return None

    ensure_opencv_available()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SharpestFrameError(f"Failed to read custom mask: {mask_path}")
    return mask


def resize_mask_for_frame(mask, frame_shape):
    if mask is None:
        return None

    height, width = frame_shape[:2]
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def compute_sharpness(frame, scale_width: int, mask=None) -> float:
    height, width = frame.shape[:2]

    if scale_width > 0 and width > scale_width:
        scale_ratio = scale_width / float(width)
        resized_height = max(1, int(height * scale_ratio))
        frame = cv2.resize(frame, (scale_width, resized_height), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (scale_width, resized_height), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if mask is not None:
        active_pixels = mask > 0
        if active_pixels.any():
            return float(laplacian[active_pixels].var())
    return float(laplacian.var())


def parse_jpeg_quality_value(value) -> int:
    if isinstance(value, int):
        quality = value
    else:
        normalized = str(value).strip()
        if normalized.endswith("%"):
            normalized = normalized[:-1].strip()

        try:
            quality = int(normalized)
        except ValueError as exc:
            raise SharpestFrameError("JPEG quality must be an integer percentage from 1 to 100.") from exc

    if quality < 1 or quality > 100:
        raise SharpestFrameError("JPEG quality must be between 1 and 100.")

    return quality


def normalize_output_format(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "jpeg":
        normalized = "jpg"

    if normalized not in SUPPORTED_OUTPUT_FORMATS:
        raise SharpestFrameError(
            f"--output-format must be one of: {', '.join(SUPPORTED_OUTPUT_FORMATS)}"
        )

    return normalized


def build_analysis_ranges(total_frames: int, workers: int, first_frame: int = 0) -> List[Tuple[int, int]]:
    task_count = max(workers * 8, workers)
    range_size = max(1, ceil(total_frames / task_count))
    ranges: List[Tuple[int, int]] = []
    start_frame = first_frame
    final_frame = first_frame + total_frames

    while start_frame < final_frame:
        end_frame = min(final_frame, start_frame + range_size)
        ranges.append((start_frame, end_frame))
        start_frame = end_frame

    return ranges


def analyze_video_range(
    video_path: str,
    start_frame: int,
    end_frame: int,
    scale_width: int,
    custom_mask_path: Optional[str] = None,
) -> List[Tuple[int, float]]:
    ensure_opencv_available()

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SharpestFrameError(f"Failed to open video: {video_path}")

    custom_mask = load_custom_mask(Path(custom_mask_path)) if custom_mask_path else None
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    results: List[Tuple[int, float]] = []
    frame_number = start_frame

    while frame_number < end_frame:
        ok, frame = capture.read()
        if not ok:
            break

        frame_mask = resize_mask_for_frame(custom_mask, frame.shape) if custom_mask is not None else None
        results.append((frame_number, compute_sharpness(frame, scale_width, frame_mask)))
        frame_number += 1

    capture.release()
    return results


def analyze_video_single_process(
    video_file: Path,
    metadata_path: Path,
    scale_width: int,
    total_frames: Optional[int],
    first_frame: int = 0,
    final_frame: Optional[int] = None,
    custom_mask_path: Optional[Path] = None,
    logger: Logger = None,
    should_cancel: CancelChecker = None,
) -> int:
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise SharpestFrameError(f"Failed to open video: {video_file}")

    custom_mask = load_custom_mask(custom_mask_path) if custom_mask_path else None
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    frame_number = first_frame
    written_count = 0
    try:
        with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
            writer = csv.writer(metadata_file)
            writer.writerow(["frame", "sharpness"])

            with create_progress(total=total_frames, desc="Analyze", logger=logger) as progress_bar:
                while True:
                    raise_if_cancelled(should_cancel)
                    if final_frame is not None and frame_number >= final_frame:
                        break

                    ok, frame = capture.read()
                    if not ok:
                        break

                    frame_mask = resize_mask_for_frame(custom_mask, frame.shape) if custom_mask is not None else None
                    sharpness = compute_sharpness(frame, scale_width, frame_mask)
                    writer.writerow([frame_number, f"{sharpness:.10f}"])
                    frame_number += 1
                    written_count += 1
                    progress_bar.update(1)
    except SharpestFrameCancelled:
        capture.release()
        if metadata_path.exists():
            metadata_path.unlink()
        raise

    capture.release()
    return written_count


def analyze_video_multi_process(
    video_file: Path,
    metadata_path: Path,
    scale_width: int,
    total_frames: int,
    workers: int,
    first_frame: int = 0,
    custom_mask_path: Optional[Path] = None,
    logger: Logger = None,
    should_cancel: CancelChecker = None,
) -> int:
    ranges = build_analysis_ranges(total_frames, workers, first_frame=first_frame)
    completed_ranges: Dict[int, List[Tuple[int, float]]] = {}
    next_range_index = 0
    written_frames = 0
    spinner_index = 0
    last_progress_refresh = 0.0

    try:
        with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
            writer = csv.writer(metadata_file)
            writer.writerow(["frame", "sharpness"])

            with create_progress(total=total_frames, desc="Analyze", logger=logger) as progress_bar:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    future_map = {
                        executor.submit(
                            analyze_video_range,
                            str(video_file),
                            start_frame,
                            end_frame,
                            scale_width,
                            str(custom_mask_path) if custom_mask_path else None,
                        ): index
                        for index, (start_frame, end_frame) in enumerate(ranges)
                    }

                    progress_bar.set_postfix_str(f"{SPINNER_FRAMES[spinner_index]} {len(future_map)} ranges")
                    progress_bar.display()
                    last_progress_refresh = monotonic()

                    while future_map:
                        raise_if_cancelled(should_cancel)

                        done_futures, _ = wait(future_map, timeout=0.1, return_when=FIRST_COMPLETED)
                        if not done_futures:
                            now = monotonic()
                            if now - last_progress_refresh >= MULTIPROCESS_PROGRESS_INTERVAL:
                                spinner_index = (spinner_index + 1) % len(SPINNER_FRAMES)
                                progress_bar.set_postfix_str(f"{SPINNER_FRAMES[spinner_index]} {len(future_map)} ranges")
                                progress_bar.display()
                                last_progress_refresh = now
                            continue

                        for future in done_futures:
                            range_index = future_map.pop(future)
                            segment_results = future.result()
                            completed_ranges[range_index] = segment_results
                            progress_bar.update(len(segment_results))

                        spinner_index = (spinner_index + 1) % len(SPINNER_FRAMES)
                        progress_bar.set_postfix_str(f"{SPINNER_FRAMES[spinner_index]} {len(future_map)} ranges")
                        progress_bar.display()
                        last_progress_refresh = monotonic()

                        while next_range_index in completed_ranges:
                            for frame_number, sharpness in completed_ranges.pop(next_range_index):
                                writer.writerow([frame_number, f"{sharpness:.10f}"])
                                written_frames += 1
                            next_range_index += 1

                    progress_bar.set_postfix_str("done")
                    progress_bar.display()
    except SharpestFrameCancelled:
        if metadata_path.exists():
            metadata_path.unlink()
        raise

    return written_frames


def analyze_video(
    video_file: Path,
    metadata_path: Path,
    scale_width: int,
    workers: int = 4,
    first_frame: int = 0,
    final_frame: Optional[int] = None,
    custom_mask_path: Optional[Path] = None,
    logger: Logger = None,
    should_cancel: CancelChecker = None,
) -> None:
    ensure_opencv_available()
    ensure_tqdm_available()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    emit("[1/3] Analyzing sharpness...", logger)
    probe_capture = cv2.VideoCapture(str(video_file))
    if not probe_capture.isOpened():
        raise SharpestFrameError(f"Failed to open video: {video_file}")

    video_total_frames = get_frame_count(probe_capture)
    probe_capture.release()
    if final_frame is None:
        final_frame = video_total_frames
    if final_frame is not None:
        total_frames = max(0, final_frame - first_frame)
    else:
        total_frames = None

    if workers <= 1 or total_frames is None or total_frames <= 1:
        frame_number = analyze_video_single_process(
            video_file=video_file,
            metadata_path=metadata_path,
            scale_width=scale_width,
            total_frames=total_frames,
            first_frame=first_frame,
            final_frame=final_frame,
            custom_mask_path=custom_mask_path,
            logger=logger,
            should_cancel=should_cancel,
        )
    else:
        frame_number = analyze_video_multi_process(
            video_file=video_file,
            metadata_path=metadata_path,
            scale_width=scale_width,
            total_frames=total_frames,
            workers=workers,
            first_frame=first_frame,
            custom_mask_path=custom_mask_path,
            logger=logger,
            should_cancel=should_cancel,
        )

    if frame_number == 0:
        raise SharpestFrameError("No frames could be read from the video.")


def parse_best_frames(
    metadata_path: Path,
    chunk_size: int,
    logger: Logger = None,
    should_cancel: CancelChecker = None,
) -> List[int]:
    ensure_tqdm_available()
    frame_data: List[Tuple[int, float]] = []

    try:
        with metadata_path.open("r", encoding="utf-8") as line_count_file:
            total_lines = max(0, sum(1 for _ in line_count_file) - 1)

        with metadata_path.open("r", newline="", encoding="utf-8") as metadata_file:
            reader = csv.DictReader(metadata_file)
            with create_progress(total=total_lines if total_lines > 0 else None, desc="Parse", logger=logger) as progress_bar:
                for row in reader:
                    raise_if_cancelled(should_cancel)

                    try:
                        frame_data.append((int(row["frame"]), float(row["sharpness"])))
                    except (KeyError, TypeError, ValueError):
                        pass
                    finally:
                        progress_bar.update(1)
    except FileNotFoundError as exc:
        raise SharpestFrameError(f"Metadata file not found: {metadata_path}") from exc

    if not frame_data:
        raise SharpestFrameError("No valid frame data was found in metadata.")

    best_frame_numbers: List[int] = []
    current_chunk: List[Tuple[int, float]] = []

    for data in frame_data:
        current_chunk.append(data)
        if len(current_chunk) == chunk_size:
            best = max(current_chunk, key=lambda item: item[1])
            best_frame_numbers.append(best[0])
            current_chunk = []

    if current_chunk:
        best = max(current_chunk, key=lambda item: item[1])
        best_frame_numbers.append(best[0])

    return best_frame_numbers


def resolve_video_range(
    video_file: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Tuple[int, Optional[int]]:
    ensure_opencv_available()
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise SharpestFrameError(f"Failed to open video: {video_file}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    total_frames = get_frame_count(capture)
    capture.release()

    first_frame = start_frame
    final_frame = end_frame
    if start_time is not None:
        if fps <= 0:
            raise SharpestFrameError("--start-time requires a video with a readable FPS value.")
        first_frame = int(start_time * fps)
    if end_time is not None:
        if fps <= 0:
            raise SharpestFrameError("--end-time requires a video with a readable FPS value.")
        final_frame = int(end_time * fps)

    if first_frame < 0:
        raise SharpestFrameError("Video range start must be 0 or greater.")
    if total_frames is not None:
        first_frame = min(first_frame, total_frames)
        if final_frame is not None:
            final_frame = min(final_frame, total_frames)
    if final_frame is not None and final_frame <= first_frame:
        raise SharpestFrameError("Video range end must be greater than the start.")
    return first_frame, final_frame


def frame_signature(frame, width: int):
    height, original_width = frame.shape[:2]
    if width > 0 and original_width > width:
        ratio = width / float(original_width)
        frame = cv2.resize(frame, (width, max(1, int(height * ratio))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def signature_similarity(previous, current) -> float:
    if previous.shape != current.shape:
        current = cv2.resize(current, (previous.shape[1], previous.shape[0]), interpolation=cv2.INTER_AREA)
    difference = cv2.absdiff(previous, current)
    return 1.0 - (float(difference.mean()) / 255.0)


def review_similar_frames(
    video_file: Path,
    frame_numbers: List[int],
    threshold: float,
    scale_width: int,
    review_path: Optional[Path] = None,
    logger: Logger = None,
    should_cancel: CancelChecker = None,
) -> List[int]:
    ensure_opencv_available()
    ensure_tqdm_available()
    if threshold <= 0 or len(frame_numbers) <= 1:
        return frame_numbers
    if threshold > 1:
        raise SharpestFrameError("--similarity-threshold must be between 0 and 1.")

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise SharpestFrameError(f"Failed to open video: {video_file}")

    kept_frames: List[int] = []
    previous_signature = None
    rows: List[Tuple[int, str, str]] = []

    emit(f"Reviewing similar frames with threshold {threshold:.3f}...", logger)
    with create_progress(total=len(frame_numbers), desc="Similar", logger=logger) as progress_bar:
        for frame_number in sorted(frame_numbers):
            raise_if_cancelled(should_cancel)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                rows.append((frame_number, "", "missing"))
                progress_bar.update(1)
                continue

            current_signature = frame_signature(frame, scale_width)
            if previous_signature is None:
                kept_frames.append(frame_number)
                previous_signature = current_signature
                rows.append((frame_number, "", "keep"))
            else:
                similarity = signature_similarity(previous_signature, current_signature)
                if similarity >= threshold:
                    rows.append((frame_number, f"{similarity:.6f}", "drop"))
                else:
                    kept_frames.append(frame_number)
                    previous_signature = current_signature
                    rows.append((frame_number, f"{similarity:.6f}", "keep"))
            progress_bar.update(1)

    capture.release()

    if review_path is not None:
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open("w", newline="", encoding="utf-8") as review_file:
            writer = csv.writer(review_file)
            writer.writerow(["frame", "similarity_to_previous_kept", "action"])
            writer.writerows(rows)
        emit(f"Similar-frame review written: {review_path}", logger)

    dropped_count = len(frame_numbers) - len(kept_frames)
    if dropped_count:
        emit(f"Excluded {dropped_count} similar frames.", logger)
    return kept_frames


def format_output_filename(output_pattern: str, output_index: int, output_format: str) -> str:
    try:
        output_name = output_pattern % output_index
    except (TypeError, ValueError):
        if "%d" not in output_pattern:
            output_name = output_pattern
        else:
            raise SharpestFrameError(
                "--output-pattern must be a valid printf-style pattern such as output_frame_%05d.png"
            )

    output_path = Path(output_name)
    current_suffix = output_path.suffix.lower()
    target_suffix = f".{output_format}"

    if current_suffix in {".jpg", ".jpeg", ".png"}:
        return str(output_path.with_suffix(target_suffix))

    if current_suffix:
        raise SharpestFrameError(
            "--output-pattern must end with .png or .jpg, or omit the extension entirely."
        )

    return f"{output_name}{target_suffix}"


def save_frame(frame, output_file: Path, jpeg_quality: int, output_format: str) -> None:
    ensure_opencv_available()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "jpg":
        success = cv2.imwrite(str(output_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    else:
        success = cv2.imwrite(str(output_file), frame)

    if not success:
        raise SharpestFrameError(f"Failed to write image: {output_file}")


def format_mask_filename(output_pattern: str, output_index: int) -> str:
    output_name = format_output_filename(output_pattern, output_index, "png")
    output_path = Path(output_name)
    return f"{output_path.stem}_mask.png"


def parse_yolo_class_filter(class_names: str) -> Optional[set]:
    names = [name.strip().lower() for name in str(class_names).split(",") if name.strip()]
    if not names:
        return None
    class_ids = set()
    for name in names:
        if name.isdigit():
            class_ids.add(int(name))
        elif name in COCO_CLASS_IDS:
            class_ids.add(COCO_CLASS_IDS[name])
        else:
            raise SharpestFrameError(f"Unknown YOLO class name: {name}")
    return class_ids


def import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SharpestFrameError(
            "YOLO mask generation requires the optional ultralytics package. "
            "Install it separately with: pip install ultralytics"
        ) from exc
    return YOLO


def yolo_mask_for_image(image, model, class_filter: Optional[set]):
    mask = None
    results = model.predict(image, verbose=False)
    if not results:
        return None
    result = results[0]
    height, width = image.shape[:2]

    if getattr(result, "masks", None) is not None and result.masks is not None:
        mask = cv2.resize(
            result.masks.data.cpu().numpy().max(axis=0).astype("uint8") * 255,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        if class_filter is not None and getattr(result, "boxes", None) is not None:
            filtered = None
            classes = result.boxes.cls.cpu().numpy().astype("int")
            masks = result.masks.data.cpu().numpy()
            for index, class_id in enumerate(classes):
                if class_id not in class_filter:
                    continue
                instance_mask = cv2.resize(
                    masks[index].astype("uint8") * 255,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                filtered = instance_mask if filtered is None else cv2.bitwise_or(filtered, instance_mask)
            mask = filtered
    elif getattr(result, "boxes", None) is not None:
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask[:, :] = 0
        for box in result.boxes:
            class_id = int(box.cls.cpu().numpy()[0])
            if class_filter is not None and class_id not in class_filter:
                continue
            x1, y1, x2, y2 = [int(value) for value in box.xyxy.cpu().numpy()[0]]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    return mask


def write_mask(mask, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if mask is None:
        raise SharpestFrameError(f"No mask was generated for: {output_file}")
    if not cv2.imwrite(str(output_file), mask):
        raise SharpestFrameError(f"Failed to write mask: {output_file}")


def generate_masks_for_video_frames(
    video_file: Path,
    frame_numbers: List[int],
    output_dir: Path,
    output_pattern: str,
    custom_mask_path: Optional[Path] = None,
    yolo_model: Optional[str] = None,
    yolo_classes: str = DEFAULT_YOLO_CLASSES,
    logger: Logger = None,
    should_cancel: CancelChecker = None,
) -> None:
    if not frame_numbers or (custom_mask_path is None and yolo_model is None):
        return

    ensure_opencv_available()
    ensure_tqdm_available()
    custom_mask = load_custom_mask(custom_mask_path) if custom_mask_path else None
    model = import_yolo()(yolo_model) if yolo_model else None
    class_filter = parse_yolo_class_filter(yolo_classes) if model is not None else None
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise SharpestFrameError(f"Failed to open video: {video_file}")

    emit(f"Generating masks... ({len(frame_numbers)} frames)", logger)
    with create_progress(total=len(frame_numbers), desc="Mask", logger=logger) as progress_bar:
        for output_index, frame_number in enumerate(sorted(set(frame_numbers)), start=1):
            raise_if_cancelled(should_cancel)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                progress_bar.update(1)
                continue

            if model is not None:
                mask = yolo_mask_for_image(frame, model, class_filter)
            else:
                mask = resize_mask_for_frame(custom_mask, frame.shape)
            write_mask(mask, output_dir / format_mask_filename(output_pattern, output_index))
            progress_bar.update(1)

    capture.release()


def generate_masks_for_still_images(
    image_paths: List[str],
    output_dir: str,
    custom_mask: Optional[str] = None,
    yolo_model: Optional[str] = None,
    yolo_classes: str = DEFAULT_YOLO_CLASSES,
    logger: Logger = None,
) -> None:
    if not custom_mask and not yolo_model:
        raise SharpestFrameError("Mask-only mode requires --custom-mask or --yolo-model.")
    ensure_opencv_available()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    static_mask = load_custom_mask(Path(custom_mask)) if custom_mask else None
    model = import_yolo()(yolo_model) if yolo_model else None
    class_filter = parse_yolo_class_filter(yolo_classes) if model is not None else None

    for image_path in image_paths:
        image_file = Path(image_path)
        image = cv2.imread(str(image_file))
        if image is None:
            raise SharpestFrameError(f"Failed to read image: {image_file}")
        mask = yolo_mask_for_image(image, model, class_filter) if model is not None else resize_mask_for_frame(static_mask, image.shape)
        write_mask(mask, output_dir_path / f"{image_file.stem}_mask.png")
    emit(f"Done: Masks were written to {output_dir_path.resolve()}", logger)


def extract_frames(
    video_file: Path,
    frame_numbers: List[int],
    output_dir: Path,
    output_pattern: str,
    jpeg_quality: int,
    output_format: str,
    logger: Logger = None,
    should_cancel: CancelChecker = None,
) -> None:
    ensure_opencv_available()
    ensure_tqdm_available()

    if not frame_numbers:
        emit("No frames selected for extraction.", logger)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    target_frame_numbers = sorted(set(frame_numbers))
    output_index = 1

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise SharpestFrameError(f"Failed to open video: {video_file}")

    emit(f"[3/3] Extracting frames... ({len(frame_numbers)} frames)", logger)

    saved_count = 0
    with create_progress(total=len(target_frame_numbers), desc="Extract", logger=logger) as progress_bar:
        for frame_number in target_frame_numbers:
            raise_if_cancelled(should_cancel)

            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                progress_bar.update(1)
                continue

            output_name = format_output_filename(output_pattern, output_index, output_format)
            save_frame(frame, output_dir / output_name, jpeg_quality, output_format)
            output_index += 1
            saved_count += 1
            progress_bar.update(1)

    capture.release()

    if saved_count != len(target_frame_numbers):
        raise SharpestFrameError(
            "Some selected frames could not be extracted. "
            f"Expected {len(target_frame_numbers)}, saved {saved_count}."
        )


def run_extraction(
    video: str,
    chunk_size: int = 30,
    scale_width: int = 1920,
    workers: int = 4,
    output_dir: str = "sharp_frames",
    output_pattern: str = "output_frame_%05d.png",
    output_format: str = "png",
    jpeg_quality: int = 95,
    analysis_only: bool = False,
    reuse_metadata: bool = True,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    custom_mask: Optional[str] = None,
    similarity_threshold: float = 0.0,
    similarity_scale_width: int = 320,
    review_similarity_only: bool = False,
    save_masks: bool = False,
    yolo_model: Optional[str] = None,
    yolo_classes: str = DEFAULT_YOLO_CLASSES,
    logger: Logger = None,
    should_cancel: CancelChecker = None,
) -> Tuple[Path, List[int]]:
    ensure_opencv_available()
    ensure_tqdm_available()

    video_file = Path(video)
    if not video_file.exists():
        raise SharpestFrameError(f"Input video was not found: {video_file}")

    if chunk_size <= 0:
        raise SharpestFrameError("--chunk-size must be 1 or greater.")

    if scale_width < 0:
        raise SharpestFrameError("--scale-width must be 0 or greater.")

    if workers <= 0:
        raise SharpestFrameError("--workers must be 1 or greater.")

    output_format = normalize_output_format(output_format)
    jpeg_quality = parse_jpeg_quality_value(jpeg_quality)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir_path / "_sharpness_metadata.csv"
    custom_mask_path = Path(custom_mask) if custom_mask else None
    if custom_mask_path is not None and not custom_mask_path.exists():
        raise SharpestFrameError(f"Custom mask was not found: {custom_mask_path}")
    first_frame, final_frame = resolve_video_range(
        video_file,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=start_time,
        end_time=end_time,
    )
    if first_frame or final_frame is not None:
        end_label = final_frame if final_frame is not None else "end"
        emit(f"Using video range: frames {first_frame} to {end_label}", logger)

    raise_if_cancelled(should_cancel)

    if metadata_path.exists() and reuse_metadata:
        emit(f"Using existing sharpness metadata: {metadata_path}", logger)
    else:
        if metadata_path.exists() and not reuse_metadata:
            emit(f"Regenerating sharpness metadata: {metadata_path}", logger)
        else:
            emit(f"Generating sharpness metadata: {metadata_path}", logger)
        analyze_video(
            video_file=video_file,
            metadata_path=metadata_path,
            scale_width=scale_width,
            workers=workers,
            first_frame=first_frame,
            final_frame=final_frame,
            custom_mask_path=custom_mask_path,
            logger=logger,
            should_cancel=should_cancel,
        )

    if analysis_only:
        emit("Done: Sharpness metadata is available.", logger)
        emit(f"Metadata path: {metadata_path.resolve()}", logger)
        return metadata_path, []

    raise_if_cancelled(should_cancel)
    emit("[2/3] Parsing metadata...", logger)
    best_frames = parse_best_frames(metadata_path, chunk_size, logger=logger, should_cancel=should_cancel)
    if similarity_threshold > 0:
        review_path = output_dir_path / "_similar_frame_review.csv"
        best_frames = review_similar_frames(
            video_file,
            best_frames,
            threshold=similarity_threshold,
            scale_width=similarity_scale_width,
            review_path=review_path,
            logger=logger,
            should_cancel=should_cancel,
        )
        if review_similarity_only:
            emit("Done: Similar-frame review is available; extraction was skipped.", logger)
            return metadata_path, best_frames

    extract_frames(
        video_file=video_file,
        frame_numbers=best_frames,
        output_dir=output_dir_path,
        output_pattern=output_pattern,
        jpeg_quality=jpeg_quality,
        output_format=output_format,
        logger=logger,
        should_cancel=should_cancel,
    )
    if save_masks or yolo_model:
        generate_masks_for_video_frames(
            video_file=video_file,
            frame_numbers=best_frames,
            output_dir=output_dir_path,
            output_pattern=output_pattern,
            custom_mask_path=custom_mask_path if save_masks else None,
            yolo_model=yolo_model,
            yolo_classes=yolo_classes,
            logger=logger,
            should_cancel=should_cancel,
        )

    emit("Done: Sharp frames were extracted successfully.", logger)
    emit(f"Output directory: {output_dir_path.resolve()}", logger)
    return metadata_path, best_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract sharp frames from a video without the ffmpeg executable."
    )
    parser.add_argument("--video", help="Input video file path")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Select one frame per N frames (e.g., 30 at 30fps ~= once per second)",
    )
    parser.add_argument(
        "--scale-width",
        type=int,
        default=1920,
        help="Width used for sharpness analysis; frames wider than this are resized",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker count for metadata extraction multiprocessing",
    )
    parser.add_argument("--output-dir", default="sharp_frames", help="Output directory")
    parser.add_argument(
        "--output-pattern",
        default="output_frame_%05d.png",
        help="Output filename pattern in printf format; extension is normalized to --output-format",
    )
    parser.add_argument(
        "--output-format",
        default="png",
        choices=SUPPORTED_OUTPUT_FORMATS,
        help="Output image format",
    )
    parser.add_argument(
        "--jpeg-quality",
        default="95",
        help="JPEG quality percentage from 1 to 100; used only when --output-format=jpg",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Analyze frames and write metadata only without extracting images",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First video frame to analyze")
    parser.add_argument("--end-frame", type=int, default=None, help="Stop before this video frame")
    parser.add_argument("--start-time", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end-time", type=float, default=None, help="End time in seconds")
    parser.add_argument("--custom-mask", default=None, help="Static mask image used for sharpness and/or mask export")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.0,
        help="Drop selected frames with similarity >= this value; 0 disables similar-frame exclusion",
    )
    parser.add_argument(
        "--similarity-scale-width",
        type=int,
        default=320,
        help="Resize width used while comparing selected frames for similarity",
    )
    parser.add_argument(
        "--review-similarity-only",
        action="store_true",
        help="Write _similar_frame_review.csv and skip extraction",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Write *_mask.png files for extracted frames using --custom-mask",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Optional Ultralytics YOLO model path/name for automatic object mask generation",
    )
    parser.add_argument(
        "--yolo-classes",
        default=DEFAULT_YOLO_CLASSES,
        help="Comma-separated COCO class names/ids to include in YOLO masks",
    )
    parser.add_argument(
        "--mask-only-images",
        nargs="+",
        default=None,
        help="Still-image mask-only mode; provide one or more image paths",
    )
    parser.add_argument(
        "--mask-output-dir",
        default="masks",
        help="Output folder for --mask-only-images",
    )
    parser.add_argument("--config", default=None, help="Load options from a JSON config file")
    parser.add_argument("--save-config", default=None, help="Save resolved options to a JSON config file and continue")
    parser.add_argument(
        "--regenerate-metadata",
        dest="reuse_metadata",
        action="store_false",
        help="Ignore existing metadata and regenerate _sharpness_metadata.csv",
    )
    parser.add_argument(
        "--gui-log-output",
        action="store_true",
        help="Emit GUI-friendly progress lines instead of terminal-formatted tqdm output",
    )
    parser.set_defaults(reuse_metadata=True)
    return parser


def option_was_provided(option_name: str, argv: List[str]) -> bool:
    return any(argument == option_name or argument.startswith(f"{option_name}=") for argument in argv)


def apply_config(args, parser: argparse.ArgumentParser, argv: List[str]) -> None:
    if not args.config:
        return
    config_path = Path(args.config)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except OSError as exc:
        raise SharpestFrameError(f"Failed to read config: {config_path}") from exc
    if not isinstance(config, dict):
        raise SharpestFrameError("Config file must contain a JSON object.")

    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    for key, value in config.items():
        if key not in actions:
            continue
        option_strings = actions[key].option_strings
        if any(option_was_provided(option, argv) for option in option_strings):
            continue
        setattr(args, key, value)


def save_config(args, config_path: str) -> None:
    excluded = {"config", "gui_log_output"}
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded and value is not None
    }
    output_path = Path(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, ensure_ascii=False)
        config_file.write("\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.gui_log_output and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logger = (lambda message: print(message, flush=True)) if args.gui_log_output else None

    try:
        apply_config(args, parser, sys.argv[1:])
        if args.save_config:
            save_config(args, args.save_config)
            emit(f"Config saved: {Path(args.save_config).resolve()}", logger)

        if args.mask_only_images:
            generate_masks_for_still_images(
                image_paths=args.mask_only_images,
                output_dir=args.mask_output_dir,
                custom_mask=args.custom_mask,
                yolo_model=args.yolo_model,
                yolo_classes=args.yolo_classes,
                logger=logger,
            )
            return

        if not args.video:
            raise SharpestFrameError("--video is required unless --mask-only-images is used.")

        run_extraction(
            video=args.video,
            chunk_size=args.chunk_size,
            scale_width=args.scale_width,
            workers=args.workers,
            output_dir=args.output_dir,
            output_pattern=args.output_pattern,
            output_format=args.output_format,
            jpeg_quality=args.jpeg_quality,
            analysis_only=args.analysis_only,
            reuse_metadata=args.reuse_metadata,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            start_time=args.start_time,
            end_time=args.end_time,
            custom_mask=args.custom_mask,
            similarity_threshold=args.similarity_threshold,
            similarity_scale_width=args.similarity_scale_width,
            review_similarity_only=args.review_similarity_only,
            save_masks=args.save_masks,
            yolo_model=args.yolo_model,
            yolo_classes=args.yolo_classes,
            logger=logger,
        )
    except SharpestFrameError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
