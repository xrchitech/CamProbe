# Probing into Camera Control of Video Models

[📄 arXiv](ARXIV_URL_PLACEHOLDER) | [🌐 Project Page](https://xrchitech.github.io/camprobe-page/)

This repository contains the code for CamProbe in **Probing into Camera Control of Video Models**, built on top of **HunyuanVideo-1.5**. We extend the original image-to-video pipeline with camera-aware warping and diffusion-time updates, enabling controllable camera motion without any additional training.


## ✨ Modes

- `custom`: load a camera trajectory from `--motion_path`
- `latent_bank`: use a predefined latent-bank motion id with default scale `0.003`
- `latent_bank_rotation`: use a predefined rotation motion id such as `arc_left` or `arc_right`


## 🚀 Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 python -u generate.py \
    --camera_mode custom \
    --motion_path motions/zoom_out.npz \
    --input_json selected-metadata.jsonl \
    --model_path tencent/HunyuanVideo-1.5 \
    --output_path outputs/custom
```

```bash
CUDA_VISIBLE_DEVICES=0 python -u generate.py \
    --camera_mode latent_bank_rotation \
    --action_id 6 \
    --rot_angle 0.02 \
    --input_json selected-metadata.jsonl \
    --model_path tencent/HunyuanVideo-1.5 \
    --output_path outputs/latent_bank_rotation
```

## 📝 Note

CamProbe is not tied to HunyuanVideo specifically. As long as the corresponding generation pipeline is modified to expose the same warping hook, the method can be run on other video models as well.
