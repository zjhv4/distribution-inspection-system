from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def normalize_torch_device(device: str) -> str:
    """Translate Ultralytics' numeric CUDA syntax to native PyTorch syntax."""
    value = str(device).strip()
    return f"cuda:{value}" if value.isdigit() else value


class ReconstructionAnomalyScorer:
    """Normal-only reconstruction scorer for a configured breaker ROI.

    The raw score is calibrated by a normal-only threshold file. A normalized
    score of 0.5 is the anomaly boundary consumed by the temporal event logic.
    """

    def __init__(
        self,
        model_path: str | Path,
        calibration_path: str | Path,
        *,
        device: str = "cpu",
        imgsz: int = 256,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for reconstruction anomaly scoring.") from exc

        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        threshold = calibration.get("calibration", {}).get("normal_quantile_threshold")
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError("Anomaly calibration file has no positive normal_quantile_threshold.")

        self.torch = torch
        self.device = torch.device(normalize_torch_device(device))
        self.imgsz = int(imgsz)
        self.threshold = float(threshold)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    def score(
        self,
        crop: Any,
        *,
        asset_id: str | None = None,
        allow_calibration: bool = False,
    ) -> dict[str, Any]:
        torch = self.torch
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = cv2.resize(gray, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        array = gray.astype(np.float32) / 255.0
        tensor = torch.from_numpy(array)[None, None].to(self.device)

        with torch.inference_mode():
            reconstruction = self.model(tensor).clamp(0, 1)
            residual = reconstruction_residual(tensor, reconstruction)
            raw_score = float(top_fraction_mean(residual, fraction=0.01).item())

        ratio = raw_score / self.threshold
        if ratio <= 1:
            normalized_score = 0.5 * ratio
        else:
            normalized_score = 0.5 + 0.5 * (1 - np.exp(-(ratio - 1)))
        return {
            "anomaly_score": float(np.clip(normalized_score, 0, 1)),
            "anomaly_raw_score": raw_score,
            "anomaly_threshold": self.threshold,
            "anomaly_model": "normal_only_reconstruction",
        }


class DinoReferenceAnomalyScorer:
    """Self-bootstrapping normal feature bank for each configured asset ROI."""

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "cpu",
        imgsz: int = 224,
        bootstrap_frames: int = 100,
        bank_size: int = 300,
        sample_stride: int = 15,
        neighbors: int = 5,
        normal_quantile: float = 0.995,
        min_raw_threshold: float = 0.02,
    ) -> None:
        try:
            import timm
            import torch
            from safetensors.torch import load_file
            from timm.models.vision_transformer import resize_pos_embed
        except ImportError as exc:
            raise RuntimeError("timm, safetensors, and PyTorch are required for DINOv2 scoring.") from exc

        self.torch = torch
        self.device = torch.device(normalize_torch_device(device))
        self.imgsz = int(imgsz)
        self.bootstrap_frames = int(bootstrap_frames)
        self.bank_size = max(int(bank_size), self.bootstrap_frames)
        self.sample_stride = max(1, int(sample_stride))
        self.neighbors = max(1, int(neighbors))
        self.normal_quantile = float(normal_quantile)
        self.min_raw_threshold = float(min_raw_threshold)
        self._banks: dict[str, list[np.ndarray]] = {}
        self._thresholds: dict[str, float] = {}
        self._normal_observations: dict[str, int] = {}

        model = timm.create_model(
            "vit_small_patch14_dinov2",
            pretrained=False,
            num_classes=0,
            img_size=self.imgsz,
        )
        state_dict = load_file(str(weights))
        if state_dict["pos_embed"].shape != model.pos_embed.shape:
            state_dict["pos_embed"] = resize_pos_embed(
                state_dict["pos_embed"],
                model.pos_embed,
                num_prefix_tokens=model.num_prefix_tokens,
                gs_new=model.patch_embed.grid_size,
            )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        material_missing = [key for key in missing if not key.startswith("head")]
        material_unexpected = [key for key in unexpected if not key.startswith("head")]
        if material_missing or material_unexpected:
            raise RuntimeError(
                f"DINOv2 checkpoint mismatch: missing={material_missing}, "
                f"unexpected={material_unexpected}"
            )
        self.model = model.to(self.device).eval()

    def score(
        self,
        crop: Any,
        *,
        asset_id: str | None = None,
        allow_calibration: bool = False,
    ) -> dict[str, Any]:
        if not asset_id:
            raise ValueError("DINOv2 reference scoring requires a stable asset_id.")
        embedding = self._embedding(crop)
        bank = self._banks.setdefault(asset_id, [])
        observation_count = self._normal_observations.get(asset_id, 0)
        if allow_calibration:
            observation_count += 1
            self._normal_observations[asset_id] = observation_count

        if asset_id not in self._thresholds:
            if (
                allow_calibration
                and observation_count % self.sample_stride == 0
                and len(bank) < self.bootstrap_frames
            ):
                bank.append(embedding)
            if len(bank) >= self.bootstrap_frames:
                self._thresholds[asset_id] = self._leave_one_out_threshold(np.stack(bank))
            else:
                return {
                    "anomaly_model": "dinov2_online_normal_bank",
                    "anomaly_calibration_ready": False,
                    "anomaly_calibration_samples": len(bank),
                    "anomaly_calibration_required": self.bootstrap_frames,
                }

        threshold = self._thresholds[asset_id]
        raw_score = self._nearest_score(embedding, np.stack(bank))
        normalized_score = normalize_anomaly_score(raw_score, threshold)
        if (
            allow_calibration
            and raw_score < threshold * 0.75
            and observation_count % self.sample_stride == 0
            and len(bank) < self.bank_size
        ):
            bank.append(embedding)
        return {
            "anomaly_score": normalized_score,
            "anomaly_raw_score": raw_score,
            "anomaly_threshold": threshold,
            "anomaly_model": "dinov2_online_normal_bank",
            "anomaly_calibration_ready": True,
            "anomaly_calibration_samples": len(bank),
        }

    def _embedding(self, crop: Any) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        rgb = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        array = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        tensor = torch.from_numpy((array - mean) / std)[None].to(self.device)
        with torch.inference_mode():
            embedding = self.model(tensor).float()
            embedding = torch.nn.functional.normalize(embedding, dim=1)
        return embedding[0].cpu().numpy()

    def _leave_one_out_threshold(self, bank: np.ndarray) -> float:
        similarities = bank @ bank.T
        np.fill_diagonal(similarities, -np.inf)
        count = min(self.neighbors, len(bank) - 1)
        nearest = np.partition(similarities, len(bank) - count, axis=1)[:, -count:]
        scores = 1.0 - nearest.mean(axis=1)
        return max(float(np.quantile(scores, self.normal_quantile)), self.min_raw_threshold)

    def _nearest_score(self, embedding: np.ndarray, bank: np.ndarray) -> float:
        similarities = bank @ embedding
        count = min(self.neighbors, len(similarities))
        nearest = np.partition(similarities, len(similarities) - count)[-count:]
        return float(1.0 - nearest.mean())


def normalize_anomaly_score(raw_score: float, threshold: float) -> float:
    ratio = raw_score / threshold
    if ratio <= 1:
        normalized = 0.5 * ratio
    else:
        normalized = 0.5 + 0.5 * (1 - np.exp(-(ratio - 1)))
    return float(np.clip(normalized, 0, 1))


def reconstruction_residual(image, reconstruction):
    """Blend pixel and edge reconstruction errors into an anomaly map."""

    torch = __import__("torch")
    pixel = (image - reconstruction).abs()
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    image_gx = torch.nn.functional.conv2d(image, sobel_x, padding=1)
    image_gy = torch.nn.functional.conv2d(image, sobel_y, padding=1)
    recon_gx = torch.nn.functional.conv2d(reconstruction, sobel_x, padding=1)
    recon_gy = torch.nn.functional.conv2d(reconstruction, sobel_y, padding=1)
    edge = ((image_gx - recon_gx).square() + (image_gy - recon_gy).square() + 1e-8).sqrt()
    return 0.7 * pixel + 0.3 * edge.clamp(0, 1)


def top_fraction_mean(values, *, fraction: float):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    flat = values.flatten(1)
    count = max(1, int(round(flat.shape[1] * fraction)))
    return flat.topk(count, dim=1).values.mean(dim=1)
