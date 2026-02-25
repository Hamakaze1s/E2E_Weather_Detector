# Sample Images

Place **3–5 adverse-weather images** here before running inference.

## Recommended sources (free, no registration required)

| Dataset | URL | Weather types |
|---------|-----|---------------|
| DAWN    | https://paperswithcode.com/dataset/dawn | fog, rain, snow, sand |
| RESIDE  | https://sites.google.com/view/reside-dehaze-datasets/ | haze/fog |
| Rain100H | https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html | heavy rain |

## Accepted formats
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

## Quick download example (DAWN subset, ~5 images)
```bash
# Download a few DAWN validation images via the official split
# (replace with any adverse-weather images you have)
wget -q -O sample_images/fog_001.jpg   "https://example.com/fog_001.jpg"
wget -q -O sample_images/rain_001.jpg  "https://example.com/rain_001.jpg"
wget -q -O sample_images/snow_001.jpg  "https://example.com/snow_001.jpg"
```

After placing images, run:
```bash
python inference.py \
    --restoration_ckpt checkpoints/restoration_best.pt \
    --detection_ckpt   checkpoints/yolov8_best.pt \
    --input            sample_images/ \
    --output           output/
```
