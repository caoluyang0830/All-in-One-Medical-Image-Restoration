# AMIFound: All-in-One Medical Image Restoration

AMIFound is a multi-task medical image restoration project built with PyTorch and Lightning.  
It supports a single model for multiple degradation types and reports **PSNR / SSIM / LPIPS** during evaluation.

## Features

- One unified model for multiple restoration tasks
- Multi-task training with mixed degradation types
- Evaluation with optional image saving
- Automatic tiled inference for large images in `test_all_patch.py`

## Supported Modalities

- `Endoscopy` (low-light enhancement)
- `Fundus` (spot-light artifact removal)
- `PET` (noised-to-clean)
- `Ultrasound` (sound artifact removal)
- `X-ray` (blur removal)
- `CT` (metal artifact restoration)
- `MR` (low-quality restoration)


## Project Structure

```text
.
├── checkpoints/
├── results/
├── src/
│   ├── train_all_multiexpers2.py
│   ├── test_all_patch.py
│   ├── options2.py
│   ├── data/
│   ├── net/
│   └── utils/
├── run_AMIFound_all_multiexpers2.sh
├── test_AMIFound_all.sh
├── requirements.txt
└── install.sh
```

## Installation

1. Create and activate a conda environment.
2. Install PyTorch (CUDA 11.8 example) and dependencies:

```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Or run:

```bash
bash install.sh
```

## Data Preparation

Important: this repository currently contains **hard-coded absolute paths** in `src/data/dataset_utils_all.py` and `src/options2.py`.  
Before training/testing, update paths to your local machine.

Main default data root used in code:

- `/data1/luyang/data/extracted_top50_samples/`

Expected paired folders:

- `Endoscopy_dark` and `Endoscopy`
- `Fundus_spot_light` and `Fundus`
- `PET_denoised` and `PET`
- `Ultrasound_sound_artifacts` and `Ultrasound`
- `X_ray_blur` and `X_ray`
- `CT_metal_artifacts` and `CT`
- `MR_LQ` and `MR`

The current split strategy in code is a fixed random **70% train / 30% test** split (`seed=42`).

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

The script prints:

- `PSNR`
- `SSIM`
- `LPIPS`

Saved outputs are written to:

```text
results/<checkpoint_id>/<benchmark>/
```

## Pretrained Checkpoint

This repo includes:

- `checkpoints/AMIFound_large/last.ckpt`

Use `--checkpoint_id AMIFound_large` for evaluation.

## Notes

- `test_all_patch.py` includes tile-based fallback for very large images.
- Some helper comments/logs in source files are in Chinese; functionality is unchanged.

## Citation

If this code helps your research, please cite your AMIFound paper/repository entry here.

```bibtex
@misc{amifound,
  title={AMIFound: All-in-One Medical Image Restoration},
  author={Your Name et al.},
  year={2026},
  howpublished={GitHub repository}
}
```
