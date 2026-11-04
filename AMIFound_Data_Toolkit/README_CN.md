# AMIFound：原始医学影像数据集获取脚本

本工具依据论文 PDF 中 **Table E35（MRI）、E36（CT）、E37（X-ray / Endoscopy / Fundus / Ultrasound / PET）及 Table A2（疾病诊断）** 整理原始数据集的访问入口。它提供 **原始数据来源索引 + 自动下载**。

## 📁 文件

| 文件 | 说明 |
| --- | --- |
| [`download_amifound.py`](download_amifound.py) | 命令行索引、下载与日志记录；支持按模态、名称和 ID 选择。 |
| [`amifound_sources.csv`](amifound_sources.csv) | 从论文 PDF 列出的 **benchmark 数据来源条目 + 诊断来源条目**。 |
| [`DISCLAIMER_CN.md`](DISCLAIMER_CN.md) | 数据使用与免责声明。 |
| [`requirements.txt`](requirements.txt) | Python 依赖。 |

## ⚙️ 安装

推荐 Python 3.10+：

```bash
python -m pip install -r requirements.txt
```


## 📥 使用

以下命令可根据需要分别执行：

```bash
# 列出论文里的全部原始来源
python download_amifound.py list
```

```bash
# 查看七类模态中的某一类（包括两种用途的来源）
python download_amifound.py list --modality MRI
```

```bash
# 仅查看论文的配对复原 benchmark 所用原始来源
python download_amifound.py list --scope benchmark
```

```bash
# 查看超声 BUSI 的官方入口、下载方式和目标目录
python download_amifound.py plan --scope benchmark --name BUSI
```

```bash
# 下载 benchmark 的 BUSI
python download_amifound.py download --scope benchmark --name BUSI --accept-terms
```

```bash
# 下载 Zenodo 的 BrainMRI 对应文件
python download_amifound.py download --scope benchmark --name BrainMRI --accept-terms
```

```bash
# 处理超声相关来源：Kaggle 数据可自动尝试下载；其余会生成手动入口清单
python download_amifound.py download --scope benchmark --modality Ultrasound --all --accept-terms
```

```bash
# 批量下载仅明确支持 Kaggle CLI / Zenodo API 的来源（可能很大，谨慎操作）
python download_amifound.py download --all --auto-only --accept-terms
```

```bash
# 只查看 downstream diagnosis 数据来源
python download_amifound.py list --scope diagnosis
```

默认输出到 `./AMIFound_raw_sources/`，可用 `--out /data/AMIFound_raw_sources` 指定存储路径。默认保留 Kaggle 返回的压缩文件，不自动解压；运行结果记录于 `download_log.jsonl`。成功完成的来源会产生 `.download_complete.json` 标记，重复运行会跳过该来源；删除标记可以重新下载，Zenodo 支持 `.part` 续传及 MD5 校验（若元数据提供）。

## 🔁 构建 HQ–LQ 对所需的下一步

论文 Appendix A.1.2 提供了七模态的退化建模思路（CT 金属伪影、MRI k-space 截断、PET 低剂量噪声、X-ray 模糊、内镜低照度、超声混响与声影、眼底反射伪影）。`download_amifound.py` **仅下载上游原始数据**，不负责数据拆分、图像预处理、实验子集筛选、退化参数采样或重建 HQ–LQ 对。若要复现实验，应结合公开代码和原始论文说明进行核对：

- 论文 DOI：https://doi.org/10.1038/s42256-026-01310-z

## ⚖️ 法律与伦理

所有原始数据归其对应来源许可及数据使用协议管理。请认真阅读 [DISCLAIMER_CN.md](DISCLAIMER_CN.md)，不可因为来源被列入论文就认为其已获可转授权或可任意商业化使用。相关医学影像不得用于未经验证的临床诊断。
  
For English documentation, please refer to [README_EN.md](README_EN.md), and for the English disclaimer, please refer to [DISCLAIMER_EN.md](DISCLAIMER_EN.md).
