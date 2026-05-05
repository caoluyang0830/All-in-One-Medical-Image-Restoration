---
language:
  - en
license: other
library_name: pytorch
tags:
  - medical-image-restoration
  - image-restoration
  - medical-imaging
  - transformer
  - mixture-of-experts
  - moe
  - fft
  - endoscopy
  - fundus
  - pet
  - ultrasound
  - x-ray
  - ct
  - mri
pipeline_tag: image-to-image
---

# Model Card for AMIFound

## Model Summary

**AMIFound** is an all-in-one medical image restoration model for unified restoration across multiple medical imaging modalities and degradation types.

The model supports a single-model restoration setting for the following modalities and tasks:

| Modality | Restoration task |
| --- | --- |
| Endoscopy | Low-light enhancement |
| Fundus | Spot-light artifact removal |
| PET | Noised-to-clean restoration |
| Ultrasound | Sound artifact removal |
| X-ray | Blur removal |
| CT | Metal artifact restoration |
| MR | Low-quality image restoration |

## Model Details

| Field | Description |
| --- | --- |
| Model name | AMIFound |
| Model type | Transformer-based Mixture-of-Experts architecture |
| Task type | Medical image restoration / image-to-image enhancement |
| Supported modalities | Endoscopy, Fundus, PET, Ultrasound, X-ray, CT, MR |
| Framework | PyTorch |
| Checkpoint name | `AMIFound_large` |
| Default checkpoint path | `checkpoints/AMIFound_large/last.ckpt` |
| Repository | `caoluyang0830/All-in-One-Medical-Image-Restoration` |
| License | Academic review and reproducibility purposes only |

## Architecture

AMIFound is based on a **Transformer-based Mixture-of-Experts (MoE) restoration architecture**.

The model uses an encoder-decoder restoration backbone with attention-based feature modeling. It further introduces frequency-aware components and expert-based adaptive restoration modules, allowing a single model to handle heterogeneous degradation patterns across different medical imaging modalities.

In summary, AMIFound can be described as:

> A Transformer-based MoE architecture for all-in-one medical image restoration.

## Intended Use

AMIFound is released for academic review, reproducibility, and research on medical image restoration.

Typical research use cases include:

- Evaluating all-in-one restoration across multiple medical imaging modalities
- Benchmarking unified restoration models under different degradation types
- Studying cross-modality generalization in medical image enhancement
- Reproducing the experimental results reported by the authors

## Data Format

The repository expects paired degraded and high-quality images for each modality.

Expected paired folders include:

| Degraded / low-quality folder | High-quality folder |
| --- | --- |
| `Endoscopy_dark` | `Endoscopy` |
| `Fundus_spot_light` | `Fundus` |
| `PET_denoised` | `PET` |
| `Ultrasound_sound_artifacts` | `Ultrasound` |
| `X_ray_blur` | `X_ray` |
| `CT_metal_artifacts` | `CT` |
| `MR_LQ` | `MR` |

The current split strategy in the repository is:

- **Training split**: 70%
- **Testing split**: 30%
- **Random seed**: 42

Before training or testing, users should update the hard-coded data paths in the repository according to their local environment.

## Installation

Create and activate a conda environment, then install PyTorch and dependencies:

```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Alternatively, run:

```bash
bash install.sh
```

## Training

Run the provided training script:

```bash
bash run_AMIFound_all_multiexpers2.sh
```

Equivalent command:

```bash
python src/train_all_multiexpers2.py \
  --model AMIFound \
  --batch_size 8 \
  --de_type MR CT X-ray Ultrasound PET Fundus Endoscopy \
  --trainset standard \
  --num_gpus 4 \
  --loss_type FFT \
  --fft_loss_weight 0.1 \
  --balance_loss_weight 0.01
```

## Testing

Run all task evaluations:

```bash
bash test_AMIFound_all.sh
```

Example single-task test:

```bash
python src/test_all_patch.py \
  --model AMIFound \
  --benchmarks Endoscopy \
  --checkpoint_id AMIFound_large \
  --de_type Endoscopy \
  --save_results
```

Saved outputs are written to:

```text
results/<checkpoint_id>/<benchmark>/
```

## Evaluation

The model can be evaluated using common image restoration metrics, including:

- PSNR
- SSIM
- LPIPS

When reporting results, please specify the modality, degradation type, checkpoint, data split, and evaluation protocol.

## Limitations

AMIFound is designed for research and reproducibility. Performance may vary when applied to datasets, scanners, institutions, or degradation patterns that differ from the experimental setting. Additional validation is recommended before applying the model to new data or deployment scenarios.

## Citation

If you use AMIFound in your research, please cite the corresponding paper or repository.

```bibtex
@misc{amifound2026,
  title        = {AMIFound: All-in-One Medical Image Restoration},
  author       = {AMIFound Authors},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/caoluyang0830/All-in-One-Medical-Image-Restoration}
}
```

Please replace this placeholder citation with the official citation once available.

## License

Copyright (c) 2026 AMIFound authors. All rights reserved.

This software is provided for academic review and reproducibility purposes only. No permission is granted to use, copy, modify, distribute, sublicense, or sell this software, in whole or in part, without prior written permission from the copyright holders.
