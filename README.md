# AMIFound: An All-in-One Foundation Model for Multimodal Medical Image Restoration and Enhancement

Official PyTorch implementation accompanying the paper **“An All in One Foundation Model for Multimodal Medical Image Restoration and Enhancement.”**

**Repository:** <https://github.com/caoluyang0830/All-in-One-Medical-Image-Restoration>

AMIFound is a unified medical image restoration model that uses one set of model weights for seven restoration and enhancement tasks across seven imaging modalities. The repository contains the model implementation, training and paired-evaluation code, degradation-generation utilities, pretrained weights hosted externally, and sample paired data.

## Quick Start

Follow [Installation](#installation), download the [pretrained model and sample data](#pretrained-model-and-sample-data), and complete [Data preparation](#data-preparation). You can then run the example in [Paired evaluation](#paired-evaluation), or follow [Training](#training) to train AMIFound.

## Supported tasks

| Modality | Restoration or enhancement task | Low-quality folder | Reference folder |
| --- | --- | --- | --- |
| Endoscopy | Low-light enhancement | `Endoscopy_dark` | `Endoscopy` |
| Fundus photography | Spot-light artifact removal | `Fundus_spot_light` | `Fundus` |
| PET | Noise reduction | `PET_denoised` | `PET` |
| Ultrasound | Acoustic artifact removal | `Ultrasound_sound_artifacts` | `Ultrasound` |
| X-ray | Deblurring | `X_ray_blur` | `X_ray` |
| CT | Metal artifact reduction | `CT_metal_artifacts` | `CT` |
| MR | Restoration of low-quality MR images | `MR_LQ` | `MR` |

## What is included

```text
.
├── src/
│   ├── data/                         # Paired training and evaluation datasets
│   ├── net/                          # AMIFound network implementations
│   ├── utils/                        # Losses, metrics, image I/O, and schedulers
│   ├── train_all_multiexpers2.py     # Multi-task training entry point
│   ├── test_all_patch.py             # Paired evaluation entry point
│   └── options2.py                   # Command-line options
├── modiality/                        # Scripts for generating task-specific degradations
├── run_AMIFound_all_multiexpers2.sh  # Example training command
├── test_AMIFound_all.sh              # Example evaluation commands
├── MODEL_CARD.md                     # Model information and usage summary
├── requirements.txt                  # Python dependencies
└── video/AMIFound.mp4                # Demonstration video
```

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/caoluyang0830/All-in-One-Medical-Image-Restoration.git
cd All-in-One-Medical-Image-Restoration

conda create -n AMIFound python=3.10 -y
conda activate AMIFound
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Training and evaluation are designed for an NVIDIA GPU with CUDA. The commands above use CUDA 11.8 as an example.

Alternatively, after creating and activating the Conda environment, install the dependencies with:

```bash
bash install.sh
```

## Pretrained model and sample data

- [Pretrained model](https://drive.google.com/drive/folders/11DfmFFPeKrW1KHtx_MAW3NtyhZOjI262)
- [Sample paired data](https://drive.google.com/drive/folders/1f1IPbx_4sMiT2JzZKWsce0UIVt8a7KQa)

Download the pretrained `last.ckpt` checkpoint and place it as follows:

```text
checkpoints/AMIFound_large/last.ckpt
```

The value passed to `--checkpoint_id` is the directory name under `checkpoints/`.

## Data preparation

The standard AMIFound loaders expect paired low-quality and reference PNG images. Within each folder pair, use identical filenames and image dimensions so that sorting produces the correct pairs.

An example layout is:

```text
DATA_ROOT/
├── Endoscopy_dark/
├── Endoscopy/
├── Fundus_spot_light/
├── Fundus/
├── PET_denoised/
├── PET/
├── Ultrasound_sound_artifacts/
├── Ultrasound/
├── X_ray_blur/
├── X_ray/
├── CT_metal_artifacts/
├── CT/
├── MR_LQ/
└── MR/
```

### Path configuration

Before training or evaluation, configure the modality-specific input and reference folder locations in `src/data/dataset_utils_all.py` for your local `DATA_ROOT`.

The repository uses a fixed random split for each paired modality:

- 70% for training
- 30% for evaluation
- random seed 42

## Training

Run the training entry point from the repository root. The following is the configuration represented by the provided example script:

```bash
python src/train_all_multiexpers2.py \
  --model AMIFound \
  --batch_size 8 \
  --de_type MR CT X-ray Ultrasound PET Fundus Endoscopy \
  --trainset standard \
  --num_gpus 4 \
  --loss_type fft \
  --fft_loss_weight 0.1 \
  --balance_loss_weight 0.01
```

Checkpoints are written to:

```text
checkpoints/<training_timestamp>/
```

The direct Python command above is recommended. The supplied `run_AMIFound_all_multiexpers2.sh` can also be adapted to the local Conda environment and repository location.

## Paired evaluation

The released evaluation entry point computes PSNR, SSIM, and LPIPS against paired reference images. Run one modality per invocation so that the benchmark and data-loader task are aligned:

```bash
python src/test_all_patch.py \
  --model AMIFound \
  --benchmarks Endoscopy \
  --checkpoint_id AMIFound_large \
  --de_type Endoscopy \
  --save_results
```

To evaluate all seven tasks sequentially:

```bash
for task in Endoscopy Fundus PET Ultrasound X-ray CT MR; do
  python src/test_all_patch.py \
    --model AMIFound \
    --benchmarks "$task" \
    --checkpoint_id AMIFound_large \
    --de_type "$task" \
    --save_results
done
```

Metrics are printed to the terminal. Restored images are saved to:

```text
results/<checkpoint_id>/<benchmark>/
```

## Usage

This release supports:

- reproduction of the paired-data training protocol;
- quantitative paired evaluation using PSNR, SSIM, and LPIPS;
- saving restored outputs during paired evaluation;
- generation of task-specific degraded images using the scripts in `modiality/`; and
- research on unified restoration across medical imaging modalities.

## Citation

If you use this repository, please cite the accompanying paper **“An All in One Foundation Model for Multimodal Medical Image Restoration and Enhancement.”**

## License

Please see [LICENSE](LICENSE) for the terms governing use of this repository.

## Contact

For questions about the code or pretrained model, please open an issue in this repository.
