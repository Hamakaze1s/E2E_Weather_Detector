#!/usr/bin/env python3
"""
One-Click Inference — E2E Adverse-Weather Detector
====================================================

Restores degraded weather images and detects objects in a single forward pass.

Quick start
-----------
    # 1. Download checkpoints (see README.md for Google Drive links), then:
    python inference.py \
        --restoration_ckpt  checkpoints/restoration_best.pt \
        --detection_ckpt    checkpoints/yolov8_best.pt \
        --input             sample_images/ \
        --output            output/

Each output image is saved as:  <stem>_result.jpg
  Left  half : weather-degraded input  (as-is)
  Right half : Histoformer-restored + YOLOv8 detection boxes

The side-by-side layout makes the trade-off immediately visible:
  • Image quality DROPS  (PSNR ↓) — the E2E model sacrifices pixel fidelity …
  • Detection accuracy RISES (mAP50 ↑) — … in favour of more accurate boxes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.restoration_net import RestorationConfig, HistoformerRestoration


# ── helpers ──────────────────────────────────────────────────────────────────

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_image(path: Path, size: int = 640) -> tuple[torch.Tensor, np.ndarray]:
    """Read an image and return (tensor [1,3,H,W] ∈[0,1], original_bgr_ndarray)."""
    bgr  = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # Pad to square, then resize
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[:h, :w] = rgb
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), bgr


def tensor_to_cv2(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) float tensor ∈[0,1] → HWC BGR uint8."""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def draw_boxes(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray) -> np.ndarray:
    """Draw bounding boxes on img (BGR)."""
    h, w = img.shape[:2]
    out  = img.copy()
    for (cx, cy, bw, bh), score, cls in zip(boxes, scores, cls_ids):
        x1 = int((cx - bw / 2) * w);  y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w);  y2 = int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"cls{int(cls)} {score:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out


# ── model loading ─────────────────────────────────────────────────────────────

def load_restoration(ckpt_path: str, device: torch.device) -> HistoformerRestoration:
    # Medium config (matches medium_32g_251225 training run)
    cfg = RestorationConfig(
        base_dim      = 16,
        num_blocks    = [1, 2, 2, 2],
        num_heads     = [1, 1, 2, 2],
        num_refine    = 1,
        ffn_expansion = 2.0,
        bias          = False,
        layernorm_type= "with_bias",
    )
    model = HistoformerRestoration(cfg).to(device).eval()
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Checkpoint is saved as {'model': OrderedDict(...), 'epoch': N}
    # Mirror the original load_state_dict_flexible: try each key with 'or' chain
    state = (
        ckpt.get("ema")
        or ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt.get("model")
        or ckpt
    )
    # Strip any "restoration." prefix if saved from E2EWeatherModel
    state = {k.replace("restoration.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[!] Restoration: {len(missing)} missing keys — checkpoint may not match model config")
    print(f"[✓] Restoration checkpoint loaded: {ckpt_path}")
    return model


def load_detection(ckpt_path: str, device: torch.device, base_weights: str = "yolov8n.pt"):
    """
    Load the fine-tuned YOLOv8 detection model for inference.

    Builds a YOLOv8Detector (identical to training), fuses Conv+BN layers
    (as done during training evaluation), then loads the full 254-key state
    dict — all keys match with 0 missing after fusion.

    Checkpoint format:  {'model': OrderedDict(254 keys), 'epoch': int}
    State dict keys split into two mirrors of the same weights:
        'net.<layer>.*'                      (127 fused-conv params)
        '_engine_model.model.<layer>.*'      (same 127 params, duplicate path)
    """
    from src.models.yolov8_detector import YOLOv8Config, YOLOv8Detector

    # Locate base yolov8n.pt
    ckpt_dir = Path(ckpt_path).parent
    candidates = [
        ckpt_dir / "yolov8n.pt",
        ckpt_dir.parent / "yolov8n.pt",
        Path(base_weights),
        Path(__file__).parent / "yolov8n.pt",
    ]
    base_path = next((str(c) for c in candidates if c.exists()), "yolov8n.pt")

    cfg      = YOLOv8Config(weights=base_path)
    detector = YOLOv8Detector(cfg, device=device)

    # Fuse Conv+BN — the checkpoint was saved after fusion
    if hasattr(detector, "_engine_model") and hasattr(detector._engine_model, "fuse"):
        detector._engine_model.fuse()

    detector.eval()

    # Load full state dict: after fusion both 'net.*' and '_engine_model.model.*'
    # map to the same fused Sequential → 0 missing keys
    raw   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw["model"]
    missing, unexpected = detector.load_state_dict(state, strict=False)
    if missing:
        print(f"[!] Detection: {len(missing)} missing keys")
    print(f"[✓] Detection checkpoint loaded:    {ckpt_path}")
    return detector


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_inference(
    restoration_ckpt: str,
    detection_ckpt:   str,
    input_path:       str,
    output_dir:       str,
    img_size:         int   = 640,
    conf_thres:       float = 0.25,
    iou_thres:        float = 0.45,
    device_str:       str   = "cuda",
    base_yolo:        str   = "yolov8n.pt",
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[i] Running on {device}")

    # Load models
    rest_model = load_restoration(restoration_ckpt, device)
    det_model  = load_detection(detection_ckpt, device, base_weights=base_yolo)

    # Collect input files
    inp = Path(input_path)
    if inp.is_file():
        files = [inp]
    elif inp.is_dir():
        files = sorted(p for p in inp.iterdir() if p.suffix.lower() in SUPPORTED)
    else:
        print(f"[!] Input path not found: {input_path}")
        sys.exit(1)

    if not files:
        print("[!] No supported image files found.")
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in files:
        print(f"  Processing: {img_path.name}")
        tensor, orig_bgr = load_image(img_path, size=img_size)
        tensor = tensor.to(device)

        # ── Step 1: Restoration (Histoformer) ─────────────────────────────
        with torch.no_grad():
            restored = rest_model(tensor).clamp(0.0, 1.0)  # residual may exceed 1.0

        # ── Step 2: Detection (YOLOv8) ──────────────────────────────────
        # The restored tensor flows DIRECTLY into YOLOv8 — no disk save.
        # predict() returns List[(N,6)]: cx cy w h conf cls, all normalized.
        preds_list = det_model.predict(restored, conf_thres=conf_thres, iou_thres=iou_thres)
        preds = preds_list[0]   # (N, 6) for this single image

        # ── Step 3: Compose side-by-side output ───────────────────────────
        weather_img  = tensor_to_cv2(tensor)
        restored_img = tensor_to_cv2(restored)

        if preds.numel() > 0:
            boxes_arr  = preds[:, :4].cpu().numpy()  # normalized cx cy w h
            scores_arr = preds[:, 4].cpu().numpy()
            cls_arr    = preds[:, 5].cpu().numpy()
            restored_img = draw_boxes(restored_img, boxes_arr, scores_arr, cls_arr)

        side_by_side = np.concatenate([weather_img, restored_img], axis=1)

        # Label the two halves
        h = side_by_side.shape[0]
        cv2.putText(side_by_side, "Input (degraded)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2)
        cv2.putText(side_by_side, "Restored + Detection (E2E)",
                    (img_size + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        out_path = out_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(out_path), side_by_side)
        print(f"  → saved: {out_path}")

    print(f"\n[✓] Done. Results in: {out_dir}")
    print("\nNote: Restored images have LOWER PSNR but HIGHER mAP50.")
    print("      This is the core E2E trade-off (see paper / README).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E2E Adverse-Weather Detector — one-click inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--restoration_ckpt", required=True,
                   help="Path to restoration_best.pt")
    p.add_argument("--detection_ckpt",   required=True,
                   help="Path to yolov8_best.pt")
    p.add_argument("--input",  default="sample_images/",
                   help="Image file or directory")
    p.add_argument("--output", default="output/",
                   help="Output directory")
    p.add_argument("--img_size",   type=int,   default=640)
    p.add_argument("--conf_thres", type=float, default=0.25)
    p.add_argument("--iou_thres",  type=float, default=0.45)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--base_yolo",  default="yolov8n.pt",
                   help="Base YOLOv8 architecture weights (auto-downloaded if not found locally)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        restoration_ckpt = args.restoration_ckpt,
        detection_ckpt   = args.detection_ckpt,
        input_path       = args.input,
        output_dir       = args.output,
        img_size         = args.img_size,
        conf_thres       = args.conf_thres,
        iou_thres        = args.iou_thres,
        device_str       = args.device,
        base_yolo        = args.base_yolo,
    )
