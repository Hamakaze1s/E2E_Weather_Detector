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
    def predict(self, images: torch.Tensor, conf_thres: Optional[float] = None,
                iou_thres: Optional[float] = None) -> List[torch.Tensor]:
        """Run YOLOv8 inference and return per-image detection tensors.

        Args:
            images: float tensor (B, 3, H, W) in [0, 1].
            conf_thres: confidence threshold (defaults to cfg.conf_thres).
            iou_thres:  NMS IoU threshold    (defaults to cfg.iou_thres).

        Returns:
            List of (N, 6) tensors — one per image — with rows
            (cx, cy, w, h, confidence, class_id), coordinates normalized to [0, 1].
        """
        device = next(self.net.parameters()).device
        conf = conf_thres if conf_thres is not None else self.cfg.conf_thres
        iou  = iou_thres  if iou_thres  is not None else self.cfg.iou_thres

        # Ultralytics expects uint8 HWC numpy arrays; passing a float tensor
        # causes incorrect internal normalization and garbage predictions.
        imgs_list = []
        for im in images.detach().to(device):
            im_np = (im.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
            imgs_list.append(im_np)

        self._ensure_yolo()
        results = self._yolo.predict(imgs_list, verbose=False, conf=conf, iou=iou, device=device)

        out: List[torch.Tensor] = []
        for r in results:
            if r.boxes is None or r.boxes.shape[0] == 0:
                out.append(torch.zeros((0, 6), device=device, dtype=torch.float32))
                continue
            xyxy   = torch.as_tensor(r.boxes.xyxy, dtype=torch.float32, device=device)
            conf_t = torch.as_tensor(r.boxes.conf, dtype=torch.float32, device=device).view(-1, 1)
            cls_t  = torch.as_tensor(r.boxes.cls,  dtype=torch.float32, device=device).view(-1, 1)
            orig_h, orig_w = float(r.orig_shape[0]), float(r.orig_shape[1])
            cxcywh = _xyxy_to_cxcywh(xyxy)
            cxcywh[:, 0] /= orig_w
            cxcywh[:, 1] /= orig_h
            cxcywh[:, 2] /= orig_w
            cxcywh[:, 3] /= orig_h
            out.append(torch.cat([cxcywh, conf_t, cls_t], dim=1))
        return out

    def _ensure_yolo(self) -> None:
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(self.cfg.weights)
            # Re-bind our trained weights to the fresh runner
            runner = self._yolo
            if runner is not None and hasattr(runner, "model"):
                if hasattr(runner.model, "model"):
                    runner.model.model = self.net
                else:
                    runner.model = self.net


def _xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    """Convert (N,4) xyxy pixel boxes to (N,4) cxcywh pixel boxes."""
    x1, y1, x2, y2 = xyxy.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = (x2 - x1).clamp(min=0)
    h  = (y2 - y1).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1)
