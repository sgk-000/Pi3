from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "map_free_inference.py"
SPEC = importlib.util.spec_from_file_location("pi3_map_free_inference", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
mf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mf
SPEC.loader.exec_module(mf)


def _write_image(
    path: Path,
    *,
    width: int = 200,
    height: int = 100,
    value: int = 32,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.full((height, width, 3), value, dtype=np.uint8)
    ).save(path)


def _write_scene(
    dataset_root: Path,
    scene_name: str,
    frames_by_sequence: dict[str, list[int]],
) -> Path:
    scene_dir = dataset_root / "train" / scene_name
    intrinsics_lines: list[str] = []
    pose_lines: list[str] = []
    for sequence, frame_indices in frames_by_sequence.items():
        sequence_offset = 100 if sequence == "seq1" else 0
        for frame_index in frame_indices:
            frame_id = f"{sequence}/frame_{frame_index:05d}.jpg"
            _write_image(
                scene_dir / frame_id,
                value=32 + sequence_offset + frame_index,
            )
            fx = 500.0 + sequence_offset + frame_index
            fy = 501.0 + sequence_offset + frame_index
            intrinsics_lines.append(
                f"{frame_id} {fx} {fy} 100.0 50.0 200 100"
            )
            pose_lines.append(
                f"{frame_id} 1.0 0.0 0.0 0.0 {frame_index}.0 0.0 0.0"
            )
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "intrinsics.txt").write_text(
        "\n".join(intrinsics_lines), encoding="utf-8"
    )
    (scene_dir / "poses.txt").write_text("\n".join(pose_lines), encoding="utf-8")
    (scene_dir / "poses_device.txt").write_text(
        "\n".join(pose_lines), encoding="utf-8"
    )
    np.savez(
        scene_dir / "overlaps.npz",
        idxs=np.zeros((0, 4), dtype=np.uint32),
        overlaps=np.zeros((0,), dtype=np.float32),
    )
    return scene_dir


def _write_image_list(
    path: Path,
    dataset_root: Path,
    rows: list[tuple[str, str]],
) -> Path:
    lines = ["scene_id,seq,seq_idx,frame_idx,rel_path,image_path"]
    for scene_id, frame_id in rows:
        sequence, filename = frame_id.split("/")
        sequence_index = 0 if sequence == "seq0" else 1
        frame_index = int(filename.removeprefix("frame_").removesuffix(".jpg"))
        image_path = dataset_root / "train" / scene_id / frame_id
        lines.append(
            f"{scene_id},{sequence},{sequence_index},{frame_index},"
            f"{frame_id},{image_path}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class StubPi3X(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(self, **kwargs):
        images = kwargs["imgs"]
        intrinsics = kwargs["intrinsics"]
        self.calls.append(
            {
                "keys": set(kwargs),
                "images": images.detach().cpu().clone(),
                "intrinsics": intrinsics.detach().cpu().clone(),
                "mask_add_ray": kwargs["mask_add_ray"].detach().cpu().clone(),
                "mask_add_depth": kwargs["mask_add_depth"].detach().cpu().clone(),
                "mask_add_pose": kwargs["mask_add_pose"].detach().cpu().clone(),
                "with_prior": kwargs["with_prior"],
            }
        )
        batch_size, num_views = images.shape[:2]
        camera_poses = torch.eye(4, device=images.device).reshape(1, 1, 4, 4)
        camera_poses = camera_poses.repeat(batch_size, num_views, 1, 1)
        camera_poses[:, :, 0, 3] = torch.arange(
            num_views, dtype=torch.float32, device=images.device
        )
        return {"camera_poses": camera_poses}


class MapFreeInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temporary_directory.name)
        self.dataset_root = self.temp_path / "dataset"
        self.scene0 = _write_scene(
            self.dataset_root,
            "s00000",
            {"seq0": [0, 2], "seq1": [0, 3]},
        )
        self.scene1 = _write_scene(
            self.dataset_root,
            "s00001",
            {"seq0": [0, 2], "seq1": [0]},
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _config(self, output_name: str, **kwargs) -> mf.MapFreeInferenceConfig:
        return mf.MapFreeInferenceConfig(
            dataset_root=self.dataset_root,
            scenes=["s00000"],
            device="cpu",
            use_amp=False,
            output_root=self.temp_path / output_name,
            **kwargs,
        )

    def test_scene_discovery_intrinsics_and_metadata_validation(self) -> None:
        self.assertEqual(
            [path.name for path in mf.discover_scene_dirs(self.dataset_root, "train", None)],
            ["s00000", "s00001"],
        )
        self.assertEqual(
            [
                mf.scene_relative_path(self.scene0, path)
                for path in mf.load_scene_image_paths(self.scene0)
            ],
            [
                "seq0/frame_00000.jpg",
                "seq0/frame_00002.jpg",
                "seq1/frame_00000.jpg",
                "seq1/frame_00003.jpg",
            ],
        )

        calibrations = mf.parse_map_free_intrinsics(self.scene0)
        calibration = calibrations["seq0/frame_00002.jpg"]
        self.assertEqual((calibration.width, calibration.height), (200, 100))
        np.testing.assert_allclose(
            calibration.matrix,
            np.array(
                [[502.0, 0.0, 100.0], [0.0, 503.0, 50.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        )

        bad_scene = self.dataset_root / "train" / "bad"
        bad_scene.mkdir(parents=True)
        (bad_scene / "intrinsics.txt").write_text(
            "seq0/frame_00000.jpg nan 500 100 50 200 100\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            mf.parse_map_free_intrinsics(bad_scene)

        with self.assertRaisesRegex(KeyError, "Missing intrinsics"):
            mf.validate_intrinsics_for_images(
                self.scene0,
                [self.scene0 / "seq0" / "frame_00000.jpg"],
                {},
            )

    def test_image_dimension_mismatch_is_rejected(self) -> None:
        frame_path = self.scene0 / "seq0" / "frame_00000.jpg"
        _write_image(frame_path, width=201, height=100)
        calibrations = mf.parse_map_free_intrinsics(self.scene0)
        with self.assertRaisesRegex(ValueError, "do not match intrinsics"):
            mf.preprocess_batch(
                [frame_path],
                self.scene0,
                calibrations,
                target_size=(518, 252),
            )

    def test_image_list_preserves_csv_order_and_rejects_duplicates(self) -> None:
        csv_path = _write_image_list(
            self.temp_path / "frames.csv",
            self.dataset_root,
            [
                ("s00000", "seq1/frame_00003.jpg"),
                ("s00000", "seq0/frame_00000.jpg"),
                ("s00001", "seq1/frame_00000.jpg"),
            ],
        )
        paths_by_scene = mf.load_image_list_csv(
            self.dataset_root, "train", csv_path
        )
        self.assertEqual(
            [mf.scene_relative_path(self.scene0, path) for path in paths_by_scene["s00000"]],
            ["seq1/frame_00003.jpg", "seq0/frame_00000.jpg"],
        )
        selected = mf.select_image_list_for_scene_dirs(
            [self.scene1], paths_by_scene, csv_path
        )
        self.assertEqual(len(selected["s00001"]), 1)

        duplicate_csv = _write_image_list(
            self.temp_path / "duplicate.csv",
            self.dataset_root,
            [
                ("s00000", "seq0/frame_00000.jpg"),
                ("s00000", "seq0/frame_00000.jpg"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "Duplicate frame"):
            mf.load_image_list_csv(self.dataset_root, "train", duplicate_csv)

    def test_preprocessing_resizes_images_and_intrinsics(self) -> None:
        image_paths = mf.load_scene_image_paths(self.scene0)[:2]
        calibrations = mf.parse_map_free_intrinsics(self.scene0)
        target_size = mf.determine_target_size(
            [calibrations[mf.scene_relative_path(self.scene0, path)] for path in image_paths],
            None,
        )
        self.assertEqual(target_size, (518, 252))

        images, intrinsics, frame_ids = mf.preprocess_batch(
            image_paths,
            self.scene0,
            calibrations,
            target_size,
        )
        self.assertEqual(tuple(images.shape), (1, 2, 3, 252, 518))
        self.assertEqual(tuple(intrinsics.shape), (1, 2, 3, 3))
        self.assertEqual(images.dtype, torch.float32)
        self.assertGreaterEqual(float(images.min()), 0.0)
        self.assertLessEqual(float(images.max()), 1.0)
        self.assertEqual(
            frame_ids,
            ["seq0/frame_00000.jpg", "seq0/frame_00002.jpg"],
        )
        np.testing.assert_allclose(
            intrinsics[0, 0].numpy(),
            np.array(
                [[1295.0, 0.0, 259.795], [0.0, 1297.59, 126.295], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            rtol=1e-5,
            atol=2e-4,
        )
        self.assertEqual(
            mf.determine_target_size(
                [calibrations["seq0/frame_00000.jpg"]], 714
            ),
            (714, 364),
        )

    def test_inference_always_injects_intrinsics_and_no_other_geometry(self) -> None:
        model = StubPi3X()
        images = torch.zeros((1, 2, 3, 14, 28), dtype=torch.float32)
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
        poses = mf.run_pi3x_inference(
            model,
            images,
            intrinsics,
            torch.device("cpu"),
            self._config("conditioning"),
        )
        self.assertEqual(poses.shape, (2, 4, 4))
        call = model.calls[0]
        self.assertEqual(
            call["keys"],
            {
                "imgs",
                "intrinsics",
                "with_prior",
                "mask_add_ray",
                "mask_add_depth",
                "mask_add_pose",
            },
        )
        self.assertTrue(call["with_prior"])
        self.assertTrue(torch.all(call["mask_add_ray"]))
        self.assertFalse(torch.any(call["mask_add_depth"]))
        self.assertFalse(torch.any(call["mask_add_pose"]))
        torch.testing.assert_close(call["intrinsics"], intrinsics)

    def test_cli_window_alias_validation_and_batch_ranges(self) -> None:
        args = mf.parse_args(
            [
                "--window-size",
                "10",
                "--num-images",
                "2",
                "--long-side-resolution",
                "714",
                "--no-amp",
            ]
        )
        config = mf.config_from_args(args)
        self.assertEqual(config.window_size, 2)
        self.assertEqual(config.long_side_resolution, 714)
        self.assertFalse(config.use_amp)
        self.assertEqual(mf.build_batch_ranges(3, 2), [(0, 2), (1, 3)])
        self.assertEqual(mf.build_batch_ranges(4, 0), [(0, 4)])
        with self.assertRaisesRegex(ValueError, "divisible"):
            mf.MapFreeInferenceConfig(long_side_resolution=701)
        with self.assertRaisesRegex(ValueError, "cannot both"):
            mf.MapFreeInferenceConfig(resume=True, overwrite=True)

    def test_full_scene_output_and_manifest_are_compatible(self) -> None:
        model = StubPi3X()
        config = self._config("full_scene")
        mf.run_map_free_inference(config, model=model)

        output_dir = (
            config.output_root
            / "pi3x"
            / "window_all_res_518"
            / "train"
            / "s00000"
        )
        with np.load(output_dir / "0.npz") as batch:
            self.assertEqual(
                set(batch.files),
                {
                    "model",
                    "date",
                    "drive",
                    "camera",
                    "frame_ids",
                    "window_start",
                    "window_end",
                    "poses",
                },
            )
            self.assertEqual(batch["model"].item(), "pi3x")
            self.assertEqual(batch["date"].item(), "train")
            self.assertEqual(batch["drive"].item(), "s00000")
            self.assertEqual(batch["camera"].item(), "all")
            self.assertEqual(batch["poses"].dtype, np.float32)
            self.assertEqual(batch["poses"].shape, (4, 4, 4))
            self.assertEqual(
                batch["frame_ids"].tolist(),
                [
                    "seq0/frame_00000.jpg",
                    "seq0/frame_00002.jpg",
                    "seq1/frame_00000.jpg",
                    "seq1/frame_00003.jpg",
                ],
            )

        manifest = json.loads(
            (output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["conditioning"],
            {"intrinsics": True, "poses": False, "depth": False},
        )
        self.assertEqual(manifest["model_spec"]["target_width"], 518)
        self.assertEqual(manifest["model_spec"]["target_height"], 252)
        self.assertEqual(manifest["model_source"]["model_id"], mf.DEFAULT_MODEL_ID)
        self.assertEqual(len(model.calls), 1)

    def test_csv_windowing_resume_and_overwrite(self) -> None:
        csv_path = _write_image_list(
            self.temp_path / "selected.csv",
            self.dataset_root,
            [
                ("s00000", "seq1/frame_00003.jpg"),
                ("s00000", "seq0/frame_00000.jpg"),
                ("s00000", "seq0/frame_00002.jpg"),
            ],
        )
        model = StubPi3X()
        config = self._config(
            "windowed", image_list_csv=csv_path, window_size=2
        )
        mf.run_map_free_inference(config, model=model)
        output_dir = (
            config.output_root
            / "pi3x"
            / "window_2_res_518"
            / "train"
            / "s00000"
        )
        with np.load(output_dir / "0.npz") as first, np.load(
            output_dir / "1.npz"
        ) as second:
            self.assertEqual(
                first["frame_ids"].tolist(),
                ["seq1/frame_00003.jpg", "seq0/frame_00000.jpg"],
            )
            self.assertEqual(
                second["frame_ids"].tolist(),
                ["seq0/frame_00000.jpg", "seq0/frame_00002.jpg"],
            )
            self.assertEqual((first["window_start"], first["window_end"]), (0, 2))
            self.assertEqual((second["window_start"], second["window_end"]), (1, 3))
        self.assertEqual(len(model.calls), 2)

        resumed_model = StubPi3X()
        mf.run_map_free_inference(
            replace(config, resume=True), model=resumed_model
        )
        self.assertEqual(resumed_model.calls, [])

        overwrite_model = StubPi3X()
        mf.run_map_free_inference(
            replace(config, overwrite=True), model=overwrite_model
        )
        self.assertEqual(len(overwrite_model.calls), 2)

    def test_skip_failures_continues_to_next_scene(self) -> None:
        (self.scene0 / "intrinsics.txt").unlink()
        model = StubPi3X()
        config = mf.MapFreeInferenceConfig(
            dataset_root=self.dataset_root,
            scenes=["s00000", "s00001"],
            device="cpu",
            use_amp=False,
            output_root=self.temp_path / "skip_failures",
            skip_failures=True,
        )
        mf.run_map_free_inference(config, model=model)
        completed_output = (
            config.output_root
            / "pi3x"
            / "window_all_res_518"
            / "train"
            / "s00001"
            / "0.npz"
        )
        self.assertTrue(completed_output.is_file())
        self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main()
