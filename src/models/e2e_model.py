"""
End-to-End Weather Detection Model
===================================

The key architectural contribution: Histoformer's restored image flows
*directly* as a GPU Tensor into YOLOv8 — no disk read/write, no detach.

Data-flow:
    weather_img  (B, 3, H, W)
         │
    ┌────▼────────────────────┐
    │  HistoformerRestoration  │   ← image restoration branch
    └────┬────────────────────┘
         │ restored  (B, 3, H, W)   ← still on GPU, gradient attached
         │
    ┌────▼────────────────────┐
    │     YOLOv8Detector       │   ← object detection branch
    └────┬────────────────────┘
         │ detections
         ▼
    {restored, detections}  → CompositeLoss (single backward pass)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .restoration_net import HistoformerRestoration, RestorationConfig, build_restoration_model
from .yolov8_detector  import YOLOv8Detector, YOLOv8Config


class E2EWeatherModel(nn.Module):
    """
    Chains restoration → detection with a *single* forward pass.

    The restored tensor is never written to disk; it flows from
    ``HistoformerRestoration.forward()`` straight into
    ``YOLOv8Detector.forward()`` as a GPU Tensor, preserving the
    computation graph for joint back-propagation.
    """

    def __init__(self, restoration_cfg: dict, detection_cfg: dict) -> None:
        super().__init__()
        self.restoration: HistoformerRestoration = build_restoration_model(restoration_cfg)
        self.detection:   YOLOv8Detector         = YOLOv8Detector(
            YOLOv8Config(
                weights   = detection_cfg["yolov8_weights"],
                conf_thres= detection_cfg.get("conf_thres", 0.25),
                iou_thres = detection_cfg.get("iou_thres",  0.45),
                freeze    = detection_cfg.get("freeze",     False),
            )
        )

    def forward(
        self,
        weather_img:    torch.Tensor,           # degraded input (B, 3, H, W), [0, 1]
        targets:        Optional[Any] = None,
        return_restored: bool = True,
    ) -> Dict[str, Any]:
        # ── Step 1 ──────────────────────────────────────────────────────────
        # Histoformer restores the degraded image entirely on-GPU.
        # `restored` is a regular torch.Tensor; gradients flow back through it.
        restored = self.restoration(weather_img)          # (B, 3, H, W) ∈ [0,1]

        # ── Step 2 ──────────────────────────────────────────────────────────
        # The restored tensor is fed *directly* into YOLOv8 — no .cpu(),
        # no PIL conversion, no disk I/O.  This is the "End-to-End" property.
        detections = self.detection(restored)             # multi-scale feature maps

        out: Dict[str, Any] = {"detections": detections}
        if return_restored:
            out["restored"] = restored
        if targets is not None:
            out["targets"] = targets
        return out

    # ── Convenience freeze/unfreeze helpers ─────────────────────────────────

    def freeze_restoration(self)   -> None:
        for p in self.restoration.parameters(): p.requires_grad_(False)

    def unfreeze_restoration(self) -> None:
        for p in self.restoration.parameters(): p.requires_grad_(True)

    def freeze_detection(self)     -> None:
        for p in self.detection.parameters():   p.requires_grad_(False)

    def unfreeze_detection(self)   -> None:
        for p in self.detection.parameters():   p.requires_grad_(True)


def build_e2e_model(cfg: dict) -> E2EWeatherModel:
    return E2EWeatherModel(
        restoration_cfg=cfg.get("restoration", {}),
        detection_cfg  =cfg.get("detection",   {}),
    )
