"""Published crack-seg U-Net runtime used by the desktop application.

The architecture and ImageNet normalization follow the MIT-licensed
``ishaan1402/crack-seg`` project.  It is deliberately kept independent from
its FastAPI service so the delivered product remains a self-contained desktop
application.  A U-Net segmentation mask also supplies a *crack region
location* bounding box when a separate crack-object detector is unavailable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


_RUNTIMES: dict[tuple[str, str], "CrackUNetRuntime"] = {}


def get_runtime(weights: str, device: str = "auto") -> "CrackUNetRuntime":
    """Return one shared runtime so locating and segmenting reuse inference."""
    key = (str(Path(weights).resolve()), device)
    if key not in _RUNTIMES:
        _RUNTIMES[key] = CrackUNetRuntime(weights, device)
    return _RUNTIMES[key]


class _SELayer:
    """Lazy wrapper: torch is imported only when a real model is enabled."""

    @staticmethod
    def build(torch, nn, channels: int):
        class SELayer(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = max(channels // 8, 8)
                self.fc = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1, bias=False),
                    nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid(),
                )

            def forward(self, x):
                return x * self.fc(x)
        return SELayer()


class CrackUNetRuntime:
    """CPU/GPU inference for the public crack-seg checkpoints (.pth)."""

    patch_size = 448
    overlap = 0.5
    threshold = 0.5

    def __init__(self, weights: str, device: str = "auto"):
        import torch
        import torch.nn as nn

        self.weights = str(Path(weights))
        self.device = torch.device(
            device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        state = torch.load(self.weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if not isinstance(state, dict) or "encoder.0.conv.0.weight" not in state:
            raise ValueError("不是 crack-seg U-Net 的有效 state_dict")

        features: list[int] = []
        index = 0
        while f"encoder.{index}.conv.0.weight" in state:
            features.append(int(state[f"encoder.{index}.conv.0.weight"].shape[0]))
            index += 1
        se = any(key.endswith(".se.fc.1.weight") for key in state)
        deep = "deep_heads.0.weight" in state
        self.model = self._build_unet(torch, nn, features, se, deep).to(self.device)
        remapped = {}
        for key, value in state.items():
            if key.startswith("up."):
                key = "up_transposes." + key[3:]
            elif key.startswith("final."):
                key = "final_conv." + key[6:]
            remapped[key] = value
        self.model.load_state_dict(remapped, strict=True)
        self.model.eval()
        self.torch = torch
        self._last_sig = ""
        self._last_mask: np.ndarray | None = None
        self._last_prob: np.ndarray | None = None

    @staticmethod
    def _build_unet(torch, nn, features: list[int], se: bool, deep: bool):
        class DoubleConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                )
                self.se = _SELayer.build(torch, nn, out_channels) if se else None

            def forward(self, x):
                x = self.conv(x)
                return self.se(x) if self.se is not None else x

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.ModuleList()
                self.pool = nn.MaxPool2d(2, 2)
                in_channels = 3
                for f in features:
                    self.encoder.append(DoubleConv(in_channels, f))
                    in_channels = f
                self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
                self.bottleneck_dropout = nn.Identity()
                self.up_transposes = nn.ModuleList()
                self.decoder = nn.ModuleList()
                for f in reversed(features):
                    self.up_transposes.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
                    self.decoder.append(DoubleConv(f * 2, f))
                self.final_conv = nn.Conv2d(features[0], 1, 1)
                self.deep_heads = nn.ModuleList(
                    [nn.Conv2d(f, 1, 1) for f in list(reversed(features))[:-1]] if deep else []
                )

            def forward(self, x):
                skips = []
                for down in self.encoder:
                    x = down(x); skips.append(x); x = self.pool(x)
                x = self.bottleneck_dropout(self.bottleneck(x))
                skips.reverse()
                for i, up in enumerate(self.up_transposes):
                    x = up(x)
                    if x.shape[2:] != skips[i].shape[2:]:
                        x = nn.functional.interpolate(x, size=skips[i].shape[2:])
                    x = self.decoder[i](torch.cat((skips[i], x), dim=1))
                return self.final_conv(x)
        return UNet()

    @staticmethod
    def _signature(bgr: np.ndarray) -> str:
        return hashlib.blake2b(np.ascontiguousarray(bgr).tobytes(), digest_size=12).hexdigest()

    def _forward(self, patches_rgb: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(patches_rgb).permute(0, 3, 1, 2).to(self.device)
        mean = self.torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = self.torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        tensor = (tensor.float() / 255.0 - mean) / std
        with self.torch.inference_mode():
            return self.torch.sigmoid(self.model(tensor)).squeeze(1).cpu().numpy()

    @staticmethod
    def _kernel(size: int) -> np.ndarray:
        c, sigma = (size - 1) / 2.0, size * 0.125
        a = np.arange(size) - c
        xx, yy = np.meshgrid(a, a)
        return np.maximum(np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma)), 1e-4)

    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a 0/255 mask and probability map in original image coordinates."""
        if bgr is None or bgr.size == 0:
            raise ValueError("空影像无法分割")
        sig = self._signature(bgr)
        if sig == self._last_sig and self._last_mask is not None and self._last_prob is not None:
            return self._last_mask.copy(), self._last_prob.copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) <= self.patch_size:
            prob = self._forward(rgb[None])[0]
        else:
            pad_h, pad_w = max(0, self.patch_size - h), max(0, self.patch_size - w)
            image = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            ih, iw = image.shape[:2]
            stride = int(self.patch_size * (1.0 - self.overlap))
            ys = list(range(0, ih - self.patch_size + 1, stride))
            xs = list(range(0, iw - self.patch_size + 1, stride))
            if ys[-1] + self.patch_size < ih: ys.append(ih - self.patch_size)
            if xs[-1] + self.patch_size < iw: xs.append(iw - self.patch_size)
            kernel = self._kernel(self.patch_size)
            accum, weight = np.zeros((ih, iw), np.float32), np.zeros((ih, iw), np.float32)
            for y in ys:
                for x in xs:
                    p = self._forward(image[y:y + self.patch_size, x:x + self.patch_size][None])[0]
                    accum[y:y + self.patch_size, x:x + self.patch_size] += p * kernel
                    weight[y:y + self.patch_size, x:x + self.patch_size] += kernel
            prob = (accum / np.maximum(weight, 1e-5))[:h, :w]
        mask = (prob > self.threshold).astype(np.uint8) * 255
        self._last_sig, self._last_mask, self._last_prob = sig, mask.copy(), prob.copy()
        return mask, prob

    def locate(self, bgr: np.ndarray, min_conf: float = 0.35) -> list[tuple[int, int, int, int, float]]:
        """Turn connected crack masks into labelled crack-region boxes."""
        mask, prob = self.predict(bgr)
        h, w = mask.shape
        joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        # Region boxes are for operator review.  Suppress isolated edge
        # speckles while retaining the full-resolution mask for quantification.
        minimum = max(48, int(h * w * 0.00010))
        found = []
        for i in range(1, count):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < minimum or bw < 3 or bh < 3:
                continue
            region = labels[y:y + bh, x:x + bw] == i
            score = float(prob[y:y + bh, x:x + bw][region].mean())
            if score >= min_conf:
                found.append((x, y, x + bw, y + bh, round(min(score, 0.99), 2)))
        return sorted(found, key=lambda item: -((item[2] - item[0]) * (item[3] - item[1])))[:8]
