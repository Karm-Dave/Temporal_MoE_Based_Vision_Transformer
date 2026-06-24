"""Extract Anti-UAV videos into frame folders for faster training.

Example:

python -m train.extract_frames \
  --data-root /home/newuser1/dataset \
  --output-root /home/newuser1/dataset_frames \
  --splits train val test \
  --modalities visible \
  --workers 4
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass(frozen=True)
class ExtractionTask:
    split: str
    sequence_name: str
    video_path: Path
    annotation_path: Path
    output_frame_dir: Path
    output_annotation_path: Path
    image_ext: str
    jpeg_quality: int
    overwrite: bool


def annotation_candidates(sequence_dir: Path, modality: str) -> list[Path]:
    return [
        sequence_dir / f"{modality}.json",
        sequence_dir / f"{sequence_dir.name}_{modality}.json",
        sequence_dir / f"{modality}_{sequence_dir.name}.json",
        sequence_dir / f"{sequence_dir.name}.json",
    ]


def find_annotation(sequence_dir: Path, modality: str) -> Path | None:
    for candidate in annotation_candidates(sequence_dir, modality):
        if candidate.exists():
            return candidate
    return None


def discover_tasks(
    data_root: Path,
    output_root: Path,
    splits: list[str],
    modalities: list[str],
    image_ext: str,
    jpeg_quality: int,
    overwrite: bool,
) -> list[ExtractionTask]:
    tasks: list[ExtractionTask] = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for modality in modalities:
                video_path = sequence_dir / f"{modality}.mp4"
                annotation_path = find_annotation(sequence_dir, modality)
                if not video_path.exists() or annotation_path is None:
                    continue
                output_sequence_dir = output_root / split / sequence_dir.name
                tasks.append(
                    ExtractionTask(
                        split=split,
                        sequence_name=sequence_dir.name,
                        video_path=video_path,
                        annotation_path=annotation_path,
                        output_frame_dir=output_sequence_dir / modality,
                        output_annotation_path=output_sequence_dir / f"{modality}.json",
                        image_ext=image_ext,
                        jpeg_quality=jpeg_quality,
                        overwrite=overwrite,
                    )
                )
    return tasks


def sequence_done(task: ExtractionTask) -> bool:
    if task.overwrite:
        return False
    if not task.output_annotation_path.exists() or not task.output_frame_dir.exists():
        return False
    return any(task.output_frame_dir.glob(f"*{task.image_ext}"))


def extract_task(task: ExtractionTask) -> tuple[str, int]:
    if sequence_done(task):
        return f"{task.split}/{task.sequence_name}", 0

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("Install opencv-python-headless before extracting frames.") from exc

    if task.overwrite and task.output_frame_dir.exists():
        shutil.rmtree(task.output_frame_dir)
    task.output_frame_dir.mkdir(parents=True, exist_ok=True)
    task.output_annotation_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task.annotation_path, task.output_annotation_path)

    capture = cv2.VideoCapture(str(task.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {task.video_path}")

    params = []
    if task.image_ext.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), task.jpeg_quality]

    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_path = task.output_frame_dir / f"{frame_count:06d}{task.image_ext}"
            if task.overwrite or not frame_path.exists():
                if not cv2.imwrite(str(frame_path), frame, params):
                    raise RuntimeError(f"Failed to write frame: {frame_path}")
            frame_count += 1
    finally:
        capture.release()

    return f"{task.split}/{task.sequence_name}", frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--modalities", nargs="+", choices=["visible", "infrared"], default=["visible"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-ext", choices=[".jpg", ".png"], default=".jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = discover_tasks(
        data_root=args.data_root,
        output_root=args.output_root,
        splits=args.splits,
        modalities=args.modalities,
        image_ext=args.image_ext,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    if not tasks:
        raise SystemExit(
            f"No videos found under {args.data_root}; expected split/sequence/<modality>.mp4"
        )

    print(f"Found {len(tasks)} sequence/modality videos.")
    print(f"Writing extracted frames to: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.workers <= 1:
        for task in tqdm(tasks, desc="extract"):
            extract_task(task)
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_task, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="extract"):
            future.result()


if __name__ == "__main__":
    main()
