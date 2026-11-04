# AMIFound: Original Medical Image Dataset Access and Download Toolkit

This toolkit organizes access links to the original datasets listed in the paper's **Table E35 (MRI), Table E36 (CT), Table E37 (X-ray / Endoscopy / Fundus / Ultrasound / PET), and Table A2 (disease diagnosis)**. It provides an **index of original data sources and automated downloading** where supported.

## 📁 Files

| File | Description |
| --- | --- |
| [`download_amifound.py`](download_amifound.py) | Command-line dataset listing, downloading, and logging; supports selecting sources by modality, name, or ID. |
| [`amifound_sources.csv`](amifound_sources.csv) | **Benchmark and diagnostic dataset source entries** compiled from the paper. |
| [`DISCLAIMER_EN.md`](DISCLAIMER_EN.md) | Data use terms and disclaimer (English). |
| [`requirements.txt`](requirements.txt) | Python dependencies. |

## ⚙️ Installation

Python 3.10+ is recommended:

```bash
python -m pip install -r requirements.txt
```

## 📥 Usage

Run the following commands as needed:

```bash
# List all original data sources reported in the paper
python download_amifound.py list
```

```bash
# List data sources for one of the seven modalities (across both use cases)
python download_amifound.py list --modality MRI
```

```bash
# List only the original sources used for the paired restoration benchmark
python download_amifound.py list --scope benchmark
```

```bash
# Show the official access link, download method, and output directory for BUSI ultrasound data
python download_amifound.py plan --scope benchmark --name BUSI
```

```bash
# Download the BUSI dataset used in the benchmark
python download_amifound.py download --scope benchmark --name BUSI --accept-terms
```

```bash
# Download the corresponding BrainMRI files from Zenodo
python download_amifound.py download --scope benchmark --name BrainMRI --accept-terms
```

```bash
# Process ultrasound sources: attempt automatic Kaggle downloads and list the remaining manual access links
python download_amifound.py download --scope benchmark --modality Ultrasound --all --accept-terms
```

```bash
# Batch-download only sources supported by the Kaggle CLI or Zenodo API (potentially large; use caution)
python download_amifound.py download --all --auto-only --accept-terms
```

```bash
# List only the downstream diagnosis data sources
python download_amifound.py list --scope diagnosis
```

Files are saved to `./AMIFound_raw_sources/` by default. Specify another directory with `--out /data/AMIFound_raw_sources`. Archives downloaded through Kaggle are retained without automatic extraction. Results are written to `download_log.jsonl`. A successful source download creates a `.download_complete.json` marker; subsequent runs skip sources with that marker. Delete the marker to download the source again. Zenodo downloads support resuming with `.part` files and MD5 verification when checksum metadata is available.

## 🔁 Next Steps for Constructing HQ–LQ Pairs

Appendix A.1.2 of the paper describes modality-specific degradation models for all seven modalities: CT metal artifacts, MRI k-space truncation, PET low-dose noise, X-ray blur, endoscopy low illumination, ultrasound reverberation and acoustic shadowing, and fundus reflection artifacts. **`download_amifound.py` downloads only the original source data.** It does not handle dataset splitting, image preprocessing, experimental subset selection, degradation parameter sampling, or reconstruction of HQ–LQ pairs. To reproduce the experiments, consult the publicly released code and the original paper:

- Paper DOI: https://doi.org/10.1038/s42256-026-01310-z

## ⚖️ Legal and Ethical Considerations

All original data remain subject to the licenses and data use agreements of their respective providers. Please read [DISCLAIMER_EN.md](DISCLAIMER_EN.md) carefully. Inclusion of a source in the paper does not imply permission to sublicense or freely commercialize its data. The medical images must not be used for clinical diagnosis without appropriate validation.

*中文说明请参阅 [README_CN.md](README_CN.md)；中文免责声明请参阅 [DISCLAIMER_CN.md](DISCLAIMER_CN.md)。*
