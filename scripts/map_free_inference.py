"""Run native, intrinsic-conditioned Pi3X inference on Map-Free scenes.

The saved ``.npz`` files intentionally follow the MapAnything Map-Free output
contract so that existing pseudo-pose consumers can read Pi3X predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm


MODEL_NAME = "pi3x"
DEFAULT_MODEL_ID = "yyfz233/Pi3X"
# DEFAULT_DATASET_ROOT = Path("/mnt/ssd2/map_free")
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/dataset/map_free")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "pi3x_inference_outputs"
DEFAULT_SPLIT = "train"
DEFAULT_WINDOW_SIZE = 0
BASE_RESOLUTION = 518
PATCH_SIZE = 14
MAP_FREE_CAMERA_NAME = "all"
SUPPORTED_SPLITS = ("train", "val", "test")
SUPPORTED_SEQUENCES = ("seq0", "seq1")
IMAGE_LIST_REQUIRED_COLUMNS = frozenset(
    {"scene_id", "seq", "seq_idx", "frame_idx", "rel_path", "image_path"}
)

# Matches the fixed 518-pixel aspect-ratio mapping used by MapAnything.
RESOLUTION_MAPPING_518: tuple[tuple[float, tuple[int, int]], ...] = (
    (1.000, (518, 518)),
    (1.321, (518, 392)),
    (1.542, (518, 336)),
    (1.762, (518, 294)),
    (2.056, (518, 252)),
    (3.083, (518, 168)),
    (0.757, (392, 518)),
    (0.649, (336, 518)),
    (0.567, (294, 518)),
    (0.486, (252, 518)),
)


@dataclass(frozen=True)
class CameraIntrinsics:
    matrix: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class MapFreeInferenceConfig:
    model: str = MODEL_NAME
    dataset_root: Path = DEFAULT_DATASET_ROOT
    split: str = DEFAULT_SPLIT
    scenes: list[str] | None = None
    image_list_csv: Path | None = None
    window_size: int = DEFAULT_WINDOW_SIZE
    long_side_resolution: int | None = None
    device: str = "auto"
    model_id: str = DEFAULT_MODEL_ID
    ckpt: Path | None = None
    output_root: Path = DEFAULT_OUTPUT_ROOT
    overwrite: bool = False
    resume: bool = False
    skip_failures: bool = False
    use_amp: bool = True
    amp_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.model != MODEL_NAME:
            raise ValueError(f"Only model={MODEL_NAME!r} is supported, got {self.model!r}.")
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported split {self.split!r}. Expected one of {SUPPORTED_SPLITS}."
            )
        if self.window_size < 0:
            raise ValueError(
                "window_size must be non-negative; use 0 for full-scene inference."
            )
        if self.long_side_resolution is not None:
            if self.long_side_resolution <= 0:
                raise ValueError("long_side_resolution must be positive when provided.")
            if self.long_side_resolution % PATCH_SIZE != 0:
                raise ValueError(
                    f"long_side_resolution must be divisible by Pi3X patch size "
                    f"{PATCH_SIZE}, got {self.long_side_resolution}."
                )
        if self.amp_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(
                f"amp_dtype must be one of ('bf16', 'fp16', 'fp32'), got {self.amp_dtype!r}."
            )
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite cannot both be enabled.")
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty.")

        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.image_list_csv is not None:
            object.__setattr__(self, "image_list_csv", Path(self.image_list_csv))
        if self.ckpt is not None:
            object.__setattr__(self, "ckpt", Path(self.ckpt))
        if self.scenes is not None:
            object.__setattr__(self, "scenes", list(self.scenes))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native, calibrated Pi3X inference on the Map-Free dataset."
    )
    parser.add_argument("--model", default=MODEL_NAME, choices=(MODEL_NAME,))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Map-Free root containing train/val/test split directories.",
    )
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional scene IDs. Defaults to every scene in the selected split.",
    )
    parser.add_argument(
        "--image-list-csv",
        type=Path,
        default=None,
        help="Optional CSV selecting and ordering seq0/seq1 frames per scene.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Images per inference window; 0 processes the complete scene at once.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Alias for --window-size; takes precedence when supplied.",
    )
    parser.add_argument(
        "--long-side-resolution",
        type=int,
        default=None,
        help="Optional patch-aligned long-side resolution instead of the fixed 518 mapping.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (auto, cuda, cuda:0, or cpu).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model identifier used when --ckpt is not supplied.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Optional local .safetensors or PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for MapAnything-compatible Pi3X inference outputs.",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing batch outputs."
    )
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenes whose complete expected batch set already exists.",
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Continue with other scenes after a scene-level failure.",
    )
    parser.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        default=True,
        help="Enable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> MapFreeInferenceConfig:
    effective_window_size = (
        args.num_images if args.num_images is not None else args.window_size
    )
    return MapFreeInferenceConfig(
        model=args.model,
        dataset_root=args.dataset_root,
        split=args.split,
        scenes=args.scenes,
        image_list_csv=args.image_list_csv,
        window_size=effective_window_size,
        long_side_resolution=args.long_side_resolution,
        device=args.device,
        model_id=args.model_id,
        ckpt=args.ckpt,
        output_root=args.output_root,
        overwrite=args.overwrite,
        resume=args.resume,
        skip_failures=args.skip_failures,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
    )


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(
        "cuda" if device_arg == "auto" and torch.cuda.is_available() else
        "cpu" if device_arg == "auto" else
        device_arg
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
    return device


def discover_scene_dirs(
    dataset_root: Path,
    split: str,
    selected_scenes: list[str] | None,
) -> list[Path]:
    split_root = dataset_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing Map-Free split directory: {split_root}")

    available = {
        path.name: path for path in sorted(split_root.iterdir()) if path.is_dir()
    }
    if not available:
        raise ValueError(f"No scene directories found under {split_root}")
    if not selected_scenes:
        return [available[name] for name in sorted(available)]

    missing = [name for name in selected_scenes if name not in available]
    if missing:
        raise ValueError(
            f"Requested scenes not found under {split_root}: {', '.join(missing)}"
        )
    return [available[name] for name in selected_scenes]


def scene_relative_path(scene_dir: Path, path: Path) -> str:
    return path.relative_to(scene_dir).as_posix()


def load_scene_image_paths(scene_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for sequence in SUPPORTED_SEQUENCES:
        sequence_dir = scene_dir / sequence
        if sequence_dir.is_dir():
            image_paths.extend(sequence_dir.glob("*.jpg"))
    return sorted(image_paths, key=lambda path: scene_relative_path(scene_dir, path))


def _required_csv_value(
    row: dict[str, str | None], column: str, line_number: int
) -> str:
    value = row.get(column)
    if value is None or value.strip() == "":
        raise ValueError(
            f"Missing value for {column!r} on image-list CSV line {line_number}."
        )
    return value.strip()


def _validate_frame_rel_path(
    rel_path: str,
    *,
    source_path: Path,
    line_number: int,
) -> None:
    pure_path = PurePosixPath(rel_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(
            f"Invalid Map-Free relative path at {source_path}:{line_number}: {rel_path!r}"
        )
    if len(pure_path.parts) != 2 or pure_path.parent.name not in SUPPORTED_SEQUENCES:
        raise ValueError(
            "Expected a path like 'seq0/frame_00000.jpg' at "
            f"{source_path}:{line_number}, got {rel_path!r}."
        )
    if pure_path.suffix.lower() != ".jpg":
        raise ValueError(
            f"Expected a .jpg frame at {source_path}:{line_number}, got {rel_path!r}."
        )


def load_image_list_csv(
    dataset_root: Path,
    split: str,
    image_list_csv: Path,
) -> dict[str, list[Path]]:
    if not image_list_csv.is_file():
        raise FileNotFoundError(f"Missing Map-Free image-list CSV: {image_list_csv}")

    image_paths_by_scene: dict[str, list[Path]] = {}
    seen_frames: set[tuple[str, str]] = set()
    with image_list_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(
            IMAGE_LIST_REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        )
        if missing_columns:
            raise ValueError(
                f"Map-Free image-list CSV is missing columns {missing_columns}: "
                f"{image_list_csv}"
            )

        for row in reader:
            line_number = reader.line_num
            scene_id = _required_csv_value(row, "scene_id", line_number)
            sequence = _required_csv_value(row, "seq", line_number)
            rel_path = _required_csv_value(row, "rel_path", line_number)
            csv_image_path = _required_csv_value(row, "image_path", line_number)
            _validate_frame_rel_path(
                rel_path, source_path=image_list_csv, line_number=line_number
            )
            if sequence not in SUPPORTED_SEQUENCES:
                raise ValueError(
                    f"Unsupported sequence at {image_list_csv}:{line_number}: {sequence!r}."
                )
            try:
                sequence_index = int(_required_csv_value(row, "seq_idx", line_number))
                frame_index = int(_required_csv_value(row, "frame_idx", line_number))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer index at {image_list_csv}:{line_number}."
                ) from exc

            expected_sequence_index = SUPPORTED_SEQUENCES.index(sequence)
            if sequence_index != expected_sequence_index:
                raise ValueError(
                    f"seq_idx mismatch at {image_list_csv}:{line_number}: "
                    f"{sequence!r} expects {expected_sequence_index}, got {sequence_index}."
                )
            expected_rel_path = f"{sequence}/frame_{frame_index:05d}.jpg"
            if rel_path != expected_rel_path:
                raise ValueError(
                    f"rel_path mismatch at {image_list_csv}:{line_number}: "
                    f"expected {expected_rel_path!r}, got {rel_path!r}."
                )

            frame_key = (scene_id, rel_path)
            if frame_key in seen_frames:
                raise ValueError(
                    f"Duplicate frame at {image_list_csv}:{line_number}: "
                    f"{scene_id}/{rel_path}"
                )
            seen_frames.add(frame_key)

            image_path = dataset_root / split / scene_id / rel_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Listed Map-Free image does not exist: {image_path} "
                    f"(CSV image_path={csv_image_path})"
                )
            image_paths_by_scene.setdefault(scene_id, []).append(image_path)

    if not image_paths_by_scene:
        raise ValueError(f"No frame rows found in image-list CSV: {image_list_csv}")
    return image_paths_by_scene


def select_image_list_for_scene_dirs(
    scene_dirs: list[Path],
    image_paths_by_scene: dict[str, list[Path]],
    image_list_csv: Path,
) -> dict[str, list[Path]]:
    missing = [path.name for path in scene_dirs if path.name not in image_paths_by_scene]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise ValueError(
            f"Requested scene(s) are missing from {image_list_csv}: {preview}{suffix}"
        )
    return {path.name: image_paths_by_scene[path.name] for path in scene_dirs}


def parse_map_free_intrinsics(scene_dir: Path) -> dict[str, CameraIntrinsics]:
    intrinsics_path = scene_dir / "intrinsics.txt"
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"Missing Map-Free intrinsics file: {intrinsics_path}")

    intrinsics_by_frame: dict[str, CameraIntrinsics] = {}
    with intrinsics_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 7:
                raise ValueError(
                    f"Malformed intrinsics at {intrinsics_path}:{line_number}; expected "
                    "'frame_path fx fy cx cy frame_width frame_height'."
                )
            frame_path = parts[0]
            _validate_frame_rel_path(
                frame_path, source_path=intrinsics_path, line_number=line_number
            )
            try:
                values = np.asarray(parts[1:], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed numeric intrinsics at {intrinsics_path}:{line_number}."
                ) from exc
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite intrinsics at {intrinsics_path}:{line_number}."
                )
            fx, fy, cx, cy, width_value, height_value = values.tolist()
            if fx <= 0 or fy <= 0:
                raise ValueError(
                    f"Focal lengths must be positive at {intrinsics_path}:{line_number}."
                )
            if width_value <= 0 or height_value <= 0:
                raise ValueError(
                    f"Frame dimensions must be positive at {intrinsics_path}:{line_number}."
                )
            if not width_value.is_integer() or not height_value.is_integer():
                raise ValueError(
                    f"Frame dimensions must be integers at {intrinsics_path}:{line_number}."
                )
            if frame_path in intrinsics_by_frame:
                raise ValueError(
                    f"Duplicate intrinsics for {frame_path!r} in {intrinsics_path}."
                )
            matrix = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            intrinsics_by_frame[frame_path] = CameraIntrinsics(
                matrix=matrix,
                width=int(width_value),
                height=int(height_value),
            )

    if not intrinsics_by_frame:
        raise ValueError(f"No intrinsics entries found in {intrinsics_path}")
    return intrinsics_by_frame


def validate_intrinsics_for_images(
    scene_dir: Path,
    image_paths: list[Path],
    intrinsics_by_frame: Mapping[str, CameraIntrinsics],
) -> None:
    missing = [
        scene_relative_path(scene_dir, path)
        for path in image_paths
        if scene_relative_path(scene_dir, path) not in intrinsics_by_frame
    ]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise KeyError(
            f"Missing intrinsics for frame(s) in {scene_dir}: {preview}{suffix}"
        )


def build_batch_ranges(num_images: int, window_size: int) -> list[tuple[int, int]]:
    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, got {window_size}.")
    if num_images <= 0:
        return []
    if window_size == 0 or num_images <= window_size:
        return [(0, num_images)]

    ranges: list[tuple[int, int]] = []
    start = 0
    while start + window_size <= num_images:
        ranges.append((start, start + window_size))
        start += window_size
    if num_images % window_size:
        ranges.append((num_images - window_size, num_images))
    return ranges


def expected_output_paths_for_job(
    save_dir: Path, num_images: int, window_size: int
) -> list[Path]:
    return [
        save_dir / f"{batch_index}.npz"
        for batch_index, _ in enumerate(build_batch_ranges(num_images, window_size))
    ]


def should_resume_skip_job(
    save_dir: Path, num_images: int, window_size: int
) -> bool:
    expected = expected_output_paths_for_job(save_dir, num_images, window_size)
    return bool(expected) and all(path.exists() for path in expected)


def get_resolution_label(config: MapFreeInferenceConfig) -> str:
    if config.long_side_resolution is not None:
        return f"long_side_{config.long_side_resolution}"
    return f"res_{BASE_RESOLUTION}"


def get_window_label(window_size: int) -> str:
    return "window_all" if window_size == 0 else f"window_{window_size}"


def output_dir_for_scene(
    output_root: Path,
    window_size: int,
    resolution_label: str,
    split: str,
    scene_name: str,
) -> Path:
    return (
        output_root
        / MODEL_NAME
        / f"{get_window_label(window_size)}_{resolution_label}"
        / split
        / scene_name
    )


def determine_target_size(
    calibrations: list[CameraIntrinsics],
    long_side_resolution: int | None,
) -> tuple[int, int]:
    if not calibrations:
        raise ValueError("Cannot determine a target size without camera calibrations.")
    average_aspect_ratio = sum(
        calibration.width / calibration.height for calibration in calibrations
    ) / len(calibrations)

    if long_side_resolution is None:
        return min(
            RESOLUTION_MAPPING_518,
            key=lambda item: abs(item[0] - average_aspect_ratio),
        )[1]

    patch_steps = long_side_resolution // PATCH_SIZE
    if average_aspect_ratio >= 1.0:
        target_width = long_side_resolution
        target_height = round(patch_steps / average_aspect_ratio) * PATCH_SIZE
    else:
        target_width = round(patch_steps * average_aspect_ratio) * PATCH_SIZE
        target_height = long_side_resolution
    if target_width < PATCH_SIZE or target_height < PATCH_SIZE:
        raise ValueError(
            f"Aspect ratio {average_aspect_ratio:g} produces invalid target size "
            f"{target_width}x{target_height}."
        )
    return target_width, target_height


def resize_and_center_crop_with_intrinsics(
    image: Image.Image,
    calibration: CameraIntrinsics,
    target_size: tuple[int, int],
) -> tuple[Image.Image, np.ndarray]:
    source_width, source_height = image.size
    if (source_width, source_height) != (calibration.width, calibration.height):
        raise ValueError(
            "Image dimensions do not match intrinsics metadata: "
            f"image={source_width}x{source_height}, "
            f"intrinsics={calibration.width}x{calibration.height}."
        )

    target_width, target_height = target_size
    scale = max(target_width / source_width, target_height / source_height) + 1e-8
    resized_width = max(target_width, int(np.floor(source_width * scale)))
    resized_height = max(target_height, int(np.floor(source_height * scale)))
    resample = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
    resized = image.resize((resized_width, resized_height), resample=resample)

    # Account for the fractional margin introduced when the isotropic target
    # scale is rounded to integer image dimensions. The +0.5/-0.5 terms retain
    # the OpenCV convention in which the top-left pixel center is (0, 0).
    fractional_margin_x = source_width * scale - resized_width
    fractional_margin_y = source_height * scale - resized_height
    intrinsics = calibration.matrix.astype(np.float64, copy=True)
    intrinsics[0, 0] *= scale
    intrinsics[1, 1] *= scale
    intrinsics[0, 2] = (
        (intrinsics[0, 2] + 0.5) * scale - 0.5 * fractional_margin_x - 0.5
    )
    intrinsics[1, 2] = (
        (intrinsics[1, 2] + 0.5) * scale - 0.5 * fractional_margin_y - 0.5
    )

    crop_left = int(np.round((resized_width - target_width) * 0.5))
    crop_top = int(np.round((resized_height - target_height) * 0.5))
    cropped = resized.crop(
        (
            crop_left,
            crop_top,
            crop_left + target_width,
            crop_top + target_height,
        )
    )
    intrinsics[0, 2] -= crop_left
    intrinsics[1, 2] -= crop_top
    if cropped.size != target_size:
        raise RuntimeError(
            f"Unexpected processed image size {cropped.size}; expected {target_size}."
        )
    if not np.isfinite(intrinsics).all() or np.linalg.det(intrinsics) == 0:
        raise ValueError("Preprocessing produced invalid camera intrinsics.")
    return cropped, intrinsics.astype(np.float32)


def preprocess_batch(
    image_paths: list[Path],
    scene_dir: Path,
    intrinsics_by_frame: Mapping[str, CameraIntrinsics],
    target_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    image_tensors: list[torch.Tensor] = []
    intrinsic_tensors: list[torch.Tensor] = []
    frame_ids: list[str] = []

    for image_path in image_paths:
        frame_id = scene_relative_path(scene_dir, image_path)
        calibration = intrinsics_by_frame[frame_id]
        with Image.open(image_path) as loaded_image:
            image = ImageOps.exif_transpose(loaded_image).convert("RGB")
            processed_image, processed_intrinsics = (
                resize_and_center_crop_with_intrinsics(
                    image,
                    calibration=calibration,
                    target_size=target_size,
                )
            )
        image_array = np.asarray(processed_image, dtype=np.uint8).copy()
        image_tensor = (
            torch.from_numpy(image_array).permute(2, 0, 1).contiguous().float() / 255.0
        )
        image_tensors.append(image_tensor)
        intrinsic_tensors.append(torch.from_numpy(processed_intrinsics))
        frame_ids.append(frame_id)

    images = torch.stack(image_tensors, dim=0).unsqueeze(0)
    intrinsics = torch.stack(intrinsic_tensors, dim=0).unsqueeze(0).float()
    return images, intrinsics, frame_ids


def load_pi3x_model(
    config: MapFreeInferenceConfig, device: torch.device
) -> torch.nn.Module:
    from pi3.models.pi3x import Pi3X

    if config.ckpt is None:
        model = Pi3X.from_pretrained(config.model_id)
    else:
        if not config.ckpt.is_file():
            raise FileNotFoundError(f"Missing Pi3X checkpoint: {config.ckpt}")
        model = Pi3X(use_multimodal=True)
        if config.ckpt.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            state_dict: Any = load_file(str(config.ckpt), device="cpu")
        else:
            state_dict = torch.load(
                config.ckpt, map_location="cpu", weights_only=False
            )
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise ValueError(
                f"Checkpoint does not contain a state dictionary: {config.ckpt}"
            )
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(
                f"Warning: checkpoint is missing {len(incompatible.missing_keys)} model keys."
            )
        if incompatible.unexpected_keys:
            print(
                "Warning: checkpoint contains "
                f"{len(incompatible.unexpected_keys)} unexpected keys."
            )

    if not getattr(model, "use_multimodal", False):
        raise RuntimeError("Pi3X must retain its multimodal branches for intrinsics input.")
    return model.to(device).eval()


def _autocast_context(
    device: torch.device, config: MapFreeInferenceConfig
) -> Any:
    enabled = (
        config.use_amp and device.type == "cuda" and config.amp_dtype != "fp32"
    )
    if not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_pi3x_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    device: torch.device,
    config: MapFreeInferenceConfig,
) -> np.ndarray:
    images = images.to(device, non_blocking=device.type == "cuda")
    intrinsics = intrinsics.to(device, non_blocking=device.type == "cuda")
    if images.ndim != 5 or images.shape[0] != 1:
        raise ValueError(f"Expected images shape (1,N,3,H,W), got {tuple(images.shape)}.")
    if intrinsics.shape != (1, images.shape[1], 3, 3):
        raise ValueError(
            f"Expected intrinsics shape (1,N,3,3), got {tuple(intrinsics.shape)}."
        )

    num_views = images.shape[1]
    mask_add_ray = torch.ones((1, num_views), dtype=torch.bool, device=device)
    mask_add_depth = torch.zeros((1, num_views), dtype=torch.bool, device=device)
    mask_add_pose = torch.zeros((1, num_views), dtype=torch.bool, device=device)
    with torch.inference_mode(), _autocast_context(device, config):
        outputs = model(
            imgs=images,
            intrinsics=intrinsics,
            with_prior=True,
            mask_add_ray=mask_add_ray,
            mask_add_depth=mask_add_depth,
            mask_add_pose=mask_add_pose,
        )

    if not isinstance(outputs, Mapping) or "camera_poses" not in outputs:
        raise KeyError("Pi3X output is missing required key 'camera_poses'.")
    camera_poses = outputs["camera_poses"]
    expected_shape = (1, num_views, 4, 4)
    if not isinstance(camera_poses, torch.Tensor) or tuple(camera_poses.shape) != expected_shape:
        shape = tuple(camera_poses.shape) if hasattr(camera_poses, "shape") else None
        raise ValueError(
            f"Expected Pi3X camera_poses shape {expected_shape}, got {shape}."
        )
    camera_poses_numpy = camera_poses[0].detach().float().cpu().numpy()
    if not np.isfinite(camera_poses_numpy).all():
        raise ValueError("Pi3X returned non-finite camera poses.")
    return camera_poses_numpy.astype(np.float32, copy=False)


def save_batch_output(
    save_path: Path,
    split_name: str,
    scene_name: str,
    frame_ids: list[str],
    window_start: int,
    window_end: int,
    camera_poses: np.ndarray,
) -> None:
    np.savez(
        save_path,
        model=np.array(MODEL_NAME),
        date=np.array(split_name),
        drive=np.array(scene_name),
        camera=np.array(MAP_FREE_CAMERA_NAME),
        frame_ids=np.asarray(frame_ids),
        window_start=np.int64(window_start),
        window_end=np.int64(window_end),
        poses=camera_poses.astype(np.float32, copy=False),
    )


def _metadata_summary(scene_dir: Path) -> dict[str, dict[str, bool]]:
    return {
        name: {"exists": (scene_dir / name).exists()}
        for name in ("intrinsics.txt", "poses.txt", "poses_device.txt", "overlaps.npz")
    }


def _json_config(config: MapFreeInferenceConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def write_run_manifest(
    output_dir: Path,
    scene_dir: Path,
    image_paths: list[Path],
    target_size: tuple[int, int],
    config: MapFreeInferenceConfig,
    resolved_device: torch.device,
) -> None:
    frame_ids = [scene_relative_path(scene_dir, path) for path in image_paths]
    manifest = {
        "model": MODEL_NAME,
        "date": config.split,
        "drive": scene_dir.name,
        "camera": MAP_FREE_CAMERA_NAME,
        "resolved_device": str(resolved_device),
        "image_list_csv": (
            str(config.image_list_csv) if config.image_list_csv is not None else None
        ),
        "image_list_mode": config.image_list_csv is not None,
        "num_listed_frames": len(frame_ids) if config.image_list_csv is not None else None,
        "first_frame_id": frame_ids[0] if frame_ids else None,
        "last_frame_id": frame_ids[-1] if frame_ids else None,
        "config": _json_config(config),
        "model_source": {
            "model_id": config.model_id if config.ckpt is None else None,
            "checkpoint": str(config.ckpt) if config.ckpt is not None else None,
        },
        "model_spec": {
            "resolution": BASE_RESOLUTION,
            "patch_size": PATCH_SIZE,
            "input_range": [0.0, 1.0],
            "target_width": target_size[0],
            "target_height": target_size[1],
        },
        "conditioning": {
            "intrinsics": True,
            "poses": False,
            "depth": False,
        },
        "map_free_metadata": _metadata_summary(scene_dir),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def run_scene_job(
    model: torch.nn.Module,
    scene_dir: Path,
    config: MapFreeInferenceConfig,
    device: torch.device,
    image_paths: list[Path] | None = None,
) -> None:
    selected_paths = load_scene_image_paths(scene_dir) if image_paths is None else list(image_paths)
    if not selected_paths:
        print(f"Warning: no images found in Map-Free scene {scene_dir}; skipping.")
        return
    print(f"Map-Free scene {config.split}/{scene_dir.name}: {len(selected_paths)} images")

    intrinsics_by_frame = parse_map_free_intrinsics(scene_dir)
    validate_intrinsics_for_images(scene_dir, selected_paths, intrinsics_by_frame)
    calibrations = [
        intrinsics_by_frame[scene_relative_path(scene_dir, path)]
        for path in selected_paths
    ]
    target_size = determine_target_size(calibrations, config.long_side_resolution)
    save_dir = output_dir_for_scene(
        output_root=config.output_root,
        window_size=config.window_size,
        resolution_label=get_resolution_label(config),
        split=config.split,
        scene_name=scene_dir.name,
    )
    if config.resume and should_resume_skip_job(
        save_dir, len(selected_paths), config.window_size
    ):
        print(f"Skipping completed Map-Free scene: {config.split}/{scene_dir.name}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        output_dir=save_dir,
        scene_dir=scene_dir,
        image_paths=selected_paths,
        target_size=target_size,
        config=config,
        resolved_device=device,
    )

    batch_ranges = build_batch_ranges(len(selected_paths), config.window_size)
    batch_iterator = tqdm(
        enumerate(batch_ranges),
        total=len(batch_ranges),
        desc=f"{config.split}/{scene_dir.name}",
        leave=False,
    )
    for batch_index, (start, end) in batch_iterator:
        save_path = save_dir / f"{batch_index}.npz"
        if save_path.exists() and not config.overwrite:
            continue
        images, intrinsics, frame_ids = preprocess_batch(
            selected_paths[start:end],
            scene_dir=scene_dir,
            intrinsics_by_frame=intrinsics_by_frame,
            target_size=target_size,
        )
        camera_poses = run_pi3x_inference(
            model=model,
            images=images,
            intrinsics=intrinsics,
            device=device,
            config=config,
        )
        save_batch_output(
            save_path=save_path,
            split_name=config.split,
            scene_name=scene_dir.name,
            frame_ids=frame_ids,
            window_start=start,
            window_end=end,
            camera_poses=camera_poses,
        )


def run_map_free_inference(
    config: MapFreeInferenceConfig,
    *,
    model: torch.nn.Module | None = None,
) -> None:
    device = resolve_device(config.device)
    scene_dirs = discover_scene_dirs(config.dataset_root, config.split, config.scenes)
    image_paths_by_scene: dict[str, list[Path]] | None = None
    if config.image_list_csv is not None:
        loaded_paths = load_image_list_csv(
            config.dataset_root, config.split, config.image_list_csv
        )
        image_paths_by_scene = select_image_list_for_scene_dirs(
            scene_dirs, loaded_paths, config.image_list_csv
        )

    if config.resume:
        pending_scene_dirs: list[Path] = []
        for scene_dir in scene_dirs:
            image_paths = (
                image_paths_by_scene[scene_dir.name]
                if image_paths_by_scene is not None
                else load_scene_image_paths(scene_dir)
            )
            save_dir = output_dir_for_scene(
                output_root=config.output_root,
                window_size=config.window_size,
                resolution_label=get_resolution_label(config),
                split=config.split,
                scene_name=scene_dir.name,
            )
            if image_paths and should_resume_skip_job(
                save_dir, len(image_paths), config.window_size
            ):
                print(f"Skipping completed Map-Free scene: {config.split}/{scene_dir.name}")
                continue
            pending_scene_dirs.append(scene_dir)
        scene_dirs = pending_scene_dirs
        if not scene_dirs:
            print("All requested Map-Free scenes are already processed; nothing to do.")
            return

    if model is None:
        model = load_pi3x_model(config, device)

    failed_scenes: list[str] = []
    for scene_dir in tqdm(scene_dirs, desc="Map-Free scene jobs"):
        try:
            run_scene_job(
                model=model,
                scene_dir=scene_dir,
                config=config,
                device=device,
                image_paths=(
                    image_paths_by_scene[scene_dir.name]
                    if image_paths_by_scene is not None
                    else None
                ),
            )
        except Exception as exc:
            if not config.skip_failures:
                raise
            failed_scenes.append(scene_dir.name)
            print(
                f"Warning: skipping failed scene {config.split}/{scene_dir.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    if failed_scenes:
        print(
            f"Completed with {len(failed_scenes)} skipped failed scene(s): "
            f"{', '.join(failed_scenes)}"
        )


def main(argv: list[str] | None = None) -> None:
    run_map_free_inference(config_from_args(parse_args(argv)))


if __name__ == "__main__":
    main()
