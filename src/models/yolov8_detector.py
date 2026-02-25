"""YOLOv8 detector wrapper (Ultralytics backend)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class YOLOv8Config:
    weights:    str
    conf_thres: float = 0.25
    iou_thres:  float = 0.45
    freeze:     bool  = False


class YOLOv8Detector(nn.Module):
    """
    Thin wrapper around Ultralytics YOLOv8.

    - Training  : delegates to Ultralytics internal loss.
    - Inference : runs NMS prediction and returns normalized [cx,cy,w,h,conf,cls] boxes.
    """

    def __init__(self, cfg: YOLOv8Config, device: Optional[torch.device] = None) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("pip install 'ultralytics>=8.3.228,<9'") from e

        # Allow PyTorch ≥2.6 safe unpickler to load Ultralytics checkpoints
        try:
            from torch.serialization import add_safe_globals
            from ultralytics.nn.tasks import DetectionModel
            add_safe_globals([DetectionModel])
        except Exception:
            pass

        self.cfg = cfg
        object.__setattr__(self, "_yolo", YOLO(cfg.weights))
        base  = getattr(self._yolo, "model", None)
        inner = getattr(base, "model", base)
        assert isinstance(inner, nn.Module)
        self.net            = inner
        self._engine_model  = base
        if device is not None:
            self.net.to(device)
        if cfg.freeze:
            for p in self.net.parameters():
                p.requires_grad_(False)
        self._loss   = self._build_loss()
        self._device = next(self.net.parameters()).device

    def _build_loss(self) -> Any:
        em = self._engine_model
        if em is not None and hasattr(em, "init_criterion"):
            return em.init_criterion()
        return None

    def forward(self, x: torch.Tensor) -> Any:
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, conf_thres: Optional[float] = None,
                iou_thres: Optional[float] = None, **kwargs):
        """Run inference and return Ultralytics Results objects.

        Extra keyword arguments (e.g. conf, iou, imgsz, verbose) are forwarded
        directly to the underlying Ultralytics YOLO.predict() call so callers
        can use either the training-style API or the Ultralytics API.
        """
        self._ensure_yolo()
        conf = kwargs.pop("conf", None) or conf_thres or self.cfg.conf_thres
        iou  = kwargs.pop("iou",  None) or iou_thres  or self.cfg.iou_thres
        kwargs.setdefault("verbose", False)
        return self._yolo.predict(x, conf=conf, iou=iou, **kwargs)

    def _ensure_yolo(self) -> None:
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(self.cfg.weights)
