"""
Composite Loss for End-to-End Adverse Weather Detection
========================================================

Core innovation: joint optimisation of image restoration and object detection
in a single backward pass.

Total loss
----------
    L_total = L_det + λ · L_rest

Detection branch (YOLOv8-style):
    L_det = w_box · L_CIoU  +  w_obj · L_BCE_obj  +  w_cls · L_BCE_cls

Restoration branch (weighted sum):
    L_rest = λ_rec   · L_L1               (pixel fidelity)
           + λ_ssim  · (1 − SSIM)         (structural similarity)
           + λ_cor   · (1 − Pearson r)    (global correlation)
           + λ_feat  · L_VGG              (perceptual / VGG-16 relu3_3)

Default weight settings used for the medium_32g_251225 run
-----------------------------------------------------------
    λ                = 0.5   (lambda_restoration: restoration ↔ detection trade-off)
    w_box            = 0.05  (CIoU box regression)
    w_obj            = 1.0   (objectness BCE)
    w_cls            = 0.5   (class BCE)
    λ_rec            = 1.0   (L1 reconstruction)
    λ_ssim           = 0.1   (SSIM structural)
    λ_cor            = 0.1   (Pearson correlation)
    λ_feat           = 1.0   (VGG-16 perceptual)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Utility: CIoU loss
# ---------------------------------------------------------------------------

def bbox_ciou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete IoU between two sets of boxes in [cx, cy, w, h] format."""
    px, py, pw, ph = pred.unbind(-1)
    tx, ty, tw, th = target.unbind(-1)
    pred_x1 = px - pw / 2;  pred_y1 = py - ph / 2
    pred_x2 = px + pw / 2;  pred_y2 = py + ph / 2
    target_x1 = tx - tw / 2; target_y1 = ty - th / 2
    target_x2 = tx + tw / 2; target_y2 = ty + th / 2

    inter_w = (torch.min(pred_x2, target_x2) - torch.max(pred_x1, target_x1)).clamp(min=0)
    inter_h = (torch.min(pred_y2, target_y2) - torch.max(pred_y1, target_y1)).clamp(min=0)
    inter   = inter_w * inter_h
    union   = pw * ph + tw * th - inter + eps
    iou     = inter / union

    cw = torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1)
    ch = torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1)
    c  = cw * cw + ch * ch + eps
    center_dist = (px - tx) ** 2 + (py - ty) ** 2

    v = (4 / math.pi ** 2) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (center_dist / c + alpha * v)


# ---------------------------------------------------------------------------
# Restoration sub-losses
# ---------------------------------------------------------------------------

def pearson_corr_loss(restored: torch.Tensor, clean: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 − Pearson correlation (per batch, averaged)."""
    r_mean = restored.mean(dim=[1, 2, 3], keepdim=True)
    c_mean = clean.mean(dim=[1, 2, 3], keepdim=True)
    r_std  = restored.std(dim=[1, 2, 3], keepdim=True)
    c_std  = clean.std(dim=[1, 2, 3], keepdim=True)
    corr   = ((restored - r_mean) * (clean - c_mean)).mean(dim=[1, 2, 3]) / (
        r_std.squeeze() * c_std.squeeze() + eps
    )
    return 1.0 - corr.mean()


def _gaussian_kernel(window_size: int, channels: int) -> torch.Tensor:
    sigma  = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss  = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k1d    = gauss / gauss.sum()
    k2d    = (k1d[:, None] * k1d[None, :]).view(1, 1, window_size, window_size)
    return k2d.repeat(channels, 1, 1, 1)


def ssim_loss(restored: torch.Tensor, clean: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """1 − SSIM (differentiable, Gaussian-weighted)."""
    c1, c2  = 0.01 ** 2, 0.03 ** 2
    pad     = window_size // 2
    ch      = restored.shape[1]
    w       = _gaussian_kernel(window_size, ch).to(restored.device)

    mu_x    = F.conv2d(restored,          w, padding=pad, groups=ch)
    mu_y    = F.conv2d(clean,             w, padding=pad, groups=ch)
    sig_x   = F.conv2d(restored * restored, w, padding=pad, groups=ch) - mu_x ** 2
    sig_y   = F.conv2d(clean    * clean,    w, padding=pad, groups=ch) - mu_y ** 2
    sig_xy  = F.conv2d(restored * clean,   w, padding=pad, groups=ch) - mu_x * mu_y

    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2) + 1e-6
    return 1 - (num / den).mean()


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG-16 features up to relu3_3 (layers[:16])."""

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.vgg = nn.Sequential(*list(vgg.features.children())[:16]).eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        norm = lambda x: (x - self.mean) / self.std
        return F.l1_loss(self.vgg(norm(restored)), self.vgg(norm(clean)))


# ---------------------------------------------------------------------------
# Detection loss (YOLOv8-compatible)
# ---------------------------------------------------------------------------

@dataclass
class DetWeights:
    """
    YOLOv8-style detection loss weights.

    Default values (medium_32g_251225):
        box  = 0.05   — CIoU regression
        obj  = 1.0    — objectness BCE
        cls  = 0.5    — class BCE
    """
    box: float = 0.05
    obj: float = 1.0
    cls: float = 0.5


class DetectionLoss(nn.Module):
    def __init__(self, weights: DetWeights, num_classes: int) -> None:
        super().__init__()
        self.w = weights
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="mean")
        self.num_classes = num_classes
        self.anchor_threshold = 4.0

    # ------------------------------------------------------------------
    # Target assignment (anchor-based, same as YOLOv5/v8 assign)
    # ------------------------------------------------------------------
    def _build_targets(
        self,
        predictions: List[Dict[str, Any]],
        targets:     List[Dict[str, torch.Tensor]],
    ) -> Tuple[list, list, list, list]:
        device = predictions[0]["pred"].device
        targets_cat_parts: List[torch.Tensor] = []
        for bi, tgt in enumerate(targets):
            if tgt["boxes"].numel() == 0:
                continue
            b = torch.full((tgt["boxes"].shape[0], 1), bi, device=device)
            targets_cat_parts.append(
                torch.cat([b, tgt["labels"].unsqueeze(1).float().to(device), tgt["boxes"].to(device)], dim=1)
            )
        targets_cat = (
            torch.cat(targets_cat_parts) if targets_cat_parts
            else torch.zeros((0, 6), device=device)
        )

        tbox_list, idx_list, anch_list, tcls_list = [], [], [], []
        for pd in predictions:
            pred, anchors, stride = pd["pred"], pd["anchors"].to(device), pd["stride"]
            bs, na, gh, gw, _ = pred.shape
            if targets_cat.shape[0] == 0:
                empty = torch.zeros(0, device=device, dtype=torch.long)
                tbox_list.append(torch.zeros((0, 4), device=device))
                idx_list.append((empty, empty, empty, empty))
                anch_list.append(torch.zeros((0, 2), device=device))
                tcls_list.append(empty)
                continue

            g = targets_cat.clone()
            g[:, 2] *= gw;  g[:, 3] *= gh;  g[:, 4] *= gw;  g[:, 5] *= gh
            r = torch.max(g[:, None, 4:6] / anchors[None],
                          anchors[None] / (g[:, None, 4:6] + 1e-7))
            mask = r.max(dim=2).values < self.anchor_threshold

            bl, al, gjl, gil, tbl, ancl, tcl = [], [], [], [], [], [], []
            for ai in range(anchors.shape[0]):
                m = mask[:, ai]
                if not m.any():
                    continue
                mg = g[m]
                gi = mg[:, 2].long().clamp_(0, gw - 1)
                gj = mg[:, 3].long().clamp_(0, gh - 1)
                bl.append(mg[:, 0].long())
                al.append(torch.full_like(bl[-1], ai))
                gjl.append(gj);  gil.append(gi)
                tbl.append(torch.cat([mg[:, 2:4], mg[:, 4:6]], 1))
                ancl.append(anchors[ai].expand_as(mg[:, 4:6]))
                tcl.append(mg[:, 1].long())

            if not bl:
                empty = torch.zeros(0, device=device, dtype=torch.long)
                tbox_list.append(torch.zeros((0, 4), device=device))
                idx_list.append((empty, empty, empty, empty))
                anch_list.append(torch.zeros((0, 2), device=device))
                tcls_list.append(empty)
                continue

            idx_list.append((torch.cat(bl), torch.cat(al), torch.cat(gjl), torch.cat(gil)))
            tbox_list.append(torch.cat(tbl))
            anch_list.append(torch.cat(ancl))
            tcls_list.append(torch.cat(tcl))

        return tbox_list, idx_list, anch_list, tcls_list

    def forward(
        self,
        predictions: List[Dict[str, Any]],
        targets:     List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = predictions[0]["pred"].device
        tbox_list, idx_list, anch_list, tcls_list = self._build_targets(predictions, targets)

        l_box = torch.zeros(1, device=device)
        l_obj = torch.zeros(1, device=device)
        l_cls = torch.zeros(1, device=device)

        for i, pd in enumerate(predictions):
            pred = pd["pred"]
            bs, na, gh, gw, no = pred.shape
            obj_tgt = torch.zeros((bs, na, gh, gw), device=device)
            b, a, gj, gi = idx_list[i]

            if b.numel():
                ps  = pred[b, a, gj, gi]
                grid = torch.stack([gi.float(), gj.float()], dim=1)
                pxy = (ps[:, :2].sigmoid() * 2.0 - 0.5) + grid
                pwh = ((ps[:, 2:4].sigmoid() * 2.0) ** 2) * anch_list[i]
                ciou = bbox_ciou(torch.cat([pxy, pwh], 1), tbox_list[i])
                l_box += (1.0 - ciou).mean()
                obj_tgt[b, a, gj, gi] = 1.0
                if no > 5 and self.num_classes > 1:
                    l_cls += self.bce_cls(
                        ps[:, 5:],
                        F.one_hot(tcls_list[i], self.num_classes).float()
                    )

            l_obj += self.bce_obj(pred[..., 4], obj_tgt)

        total = self.w.box * l_box + self.w.obj * l_obj + self.w.cls * l_cls
        return total, {"loss_box": l_box.detach(), "loss_obj": l_obj.detach(), "loss_cls": l_cls.detach()}


# ---------------------------------------------------------------------------
# Restoration loss
# ---------------------------------------------------------------------------

class RestorationLoss(nn.Module):
    """
    Weighted combination of pixel, structural, correlation, and perceptual losses.

    Default λ values (medium_32g_251225):
        rec        = 1.0   (L1 pixel)
        ssim       = 0.1   (1 − SSIM)
        cor        = 0.1   (1 − Pearson r)
        perceptual = 1.0   (VGG-16 feature)
    """

    def __init__(self, weights: Dict[str, float]) -> None:
        super().__init__()
        self.w = weights
        self.vgg = VGGPerceptualLoss()

    def forward(
        self, restored: torch.Tensor, clean: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dev = restored.device
        w   = self.w

        l_rec  = F.l1_loss(restored, clean)
        l_ssim = ssim_loss(restored, clean)       if w.get("ssim", 0)       > 0 else torch.zeros(1, device=dev)
        l_cor  = pearson_corr_loss(restored, clean) if w.get("cor", 0)      > 0 else torch.zeros(1, device=dev)
        l_feat = self.vgg(restored, clean)          if w.get("perceptual", 0) > 0 else torch.zeros(1, device=dev)

        total = (
            w.get("rec",        1.0) * l_rec
          + w.get("ssim",       0.0) * l_ssim
          + w.get("cor",        0.0) * l_cor
          + w.get("perceptual", 0.0) * l_feat
        )
        return total, {
            "loss_rec":  l_rec.detach(),
            "loss_ssim": l_ssim.detach(),
            "loss_cor":  l_cor.detach(),
            "loss_feat": l_feat.detach(),
        }


# ---------------------------------------------------------------------------
# Composite (joint) loss  ←  the key contribution
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """
    Joint detection + restoration loss for end-to-end training.

    Formula
    -------
        L_total = L_det + λ · L_rest

    where λ = ``lambda_restoration`` (default 0.5).

        L_det  = w_box·L_CIoU + w_obj·L_BCE_obj + w_cls·L_BCE_cls
        L_rest = λ_rec·L_L1  + λ_ssim·(1−SSIM) + λ_cor·(1−r) + λ_feat·L_VGG

    Both branches share the *same* backward pass — there is no disk I/O between
    the restoration and detection modules.  The restored tensor flows directly as
    a GPU Tensor into the YOLOv8 input, enabling end-to-end gradient flow.

    Parameters (medium_32g_251225 defaults)
    ----------------------------------------
        lambda_restoration = 0.5
        det_weights  : box=0.05, obj=1.0, cls=0.5
        rest_weights : rec=1.0, ssim=0.1, cor=0.1, perceptual=1.0
    """

    def __init__(
        self,
        det_weights:        Dict[str, float],
        rest_weights:       Dict[str, float],
        lambda_restoration: float,
        num_classes:        int,
    ) -> None:
        super().__init__()
        self.lambda_restoration = lambda_restoration
        self.det  = DetectionLoss(DetWeights(**det_weights), num_classes)
        self.rest = RestorationLoss(rest_weights)

    def forward(
        self,
        predictions: List[Dict[str, Any]],   # YOLOv8 multi-scale outputs
        restored:    torch.Tensor,             # output of Histoformer (on GPU)
        clean:       torch.Tensor,             # ground-truth clean image
        targets:     List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l_det,  det_info  = self.det(predictions, targets)
        l_rest, rest_info = self.rest(restored, clean)

        total = l_det + self.lambda_restoration * l_rest
        info  = {**det_info, **rest_info, "loss_total": total.detach(),
                 "lambda": torch.tensor(self.lambda_restoration)}
        return total, info
