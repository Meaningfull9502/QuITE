<div align="center">

# [ICML 2026] QuITE: Query-Based Irregular Time Series Embedding

[![Conference](https://img.shields.io/badge/ICML-2026-1f6feb?style=flat-square)](https://icml.cc/Conferences/2026)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

**Official PyTorch implementation** of  *QuITE: Query-Based Irregular Time Series Embedding* (ICML 2026).

A plug-and-play **input-embedding** module that lets any standard MTS backbone — PatchTST, PatchMixer, TMix, iTransformer, S-Mamba, TimeXer — handle **Irregular Multivariate Time Series (IMTS)** without architectural changes or artificial value generation.

📄 [**Paper (arXiv)**](https://arxiv.org/abs/2605.28166) · 📑 [**OpenReview**](https://openreview.net/forum?id=ILQGHFvEoo) · 💻 [**Code**](https://github.com/Meaningfull9502/QuITE) · 📜 [**Citation**](#-citation)

</div>

---

## 🔥 News

- **2026-05-01** — **QuITE** is accepted by **ICML 2026** 🎉

---

## 🧐 Overview

Irregular Multivariate Time Series (IMTS) are common in healthcare, industrial monitoring, and climatology, yet they break the uniform-sampling assumption baked into standard MTS embeddings. We address this at the **input-embedding** stage.

- **QuITE** *(paper §4)* — a plug-and-play module. A small set of learnable **query tokens** aggregates irregular observations through a single masked self-attention layer.
- **QuITE++** *(paper §5)* — a hierarchical extension: query-based patch embedding → patch-level self-attention → variable-level self-attention → cross-attention decoder over future-time queries.

Across **7 benchmarks × 6 MTS backbones**, plugging QuITE in yields average relative gains of **up to 54.7 % in forecasting** and **up to 15.8 % in classification**; QuITE++ achieves the **best performance on 20 / 24 forecasting settings** (paper Tables 2-4).

<p align="center">
  <img src="figs/forecasting.png" width="780"/>
  <br/>
  <em><b>Figure 1 (a).</b> Effectiveness of QuITE on forecasting — averaged over all datasets.</em>
</p>

<p align="center">
  <img src="figs/classification.png" width="780"/>
  <br/>
  <em><b>Figure 1 (b).</b> Effectiveness of QuITE on classification — averaged over all datasets.</em>
</p>

---

## 🧠 Model

<p align="center">
  <img src="figs/QuITE.png" width="900"/>
  <br/>
  <em><b>Figure 2.</b> Overall framework of <b>QuITE</b>. Learnable query tokens aggregate irregular observations through a single self-attention layer.</em>
</p>

<p align="center">
  <img src="figs/model.jpg" width="900"/>
  <br/>
  <em><b>Figure 3.</b> Overall architecture of <b>QuITE++</b>. A hierarchical encoder models intra-variable patch-level (B) and inter-variable (C) interactions via learnable query tokens.</em>
</p>

---

## 📊 Datasets

We use **4 forecasting** + **3 classification** benchmarks, following [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN) for forecasting and [Raindrop](https://github.com/mims-harvard/Raindrop) for classification preprocessing. Place all data under `../data/` (sibling of this repository).

### Forecasting (paper Appendix A.1)

| Dataset | # Samples | # Vars | Avg. Length | Missing |
|---|---:|---:|---:|---:|
| Human Activity | 5,400  | 12 | 120 | 75.0 % |
| USHCN          | 26,736 |  5 | 163 | 77.9 % |
| PhysioNet      | 12,000 | 36 |  74 | 88.4 % |
| MIMIC-III      | 23,457 | 96 |  46 | 96.7 % |

### Classification (paper Appendix A.2)

| Dataset | # Samples | # Vars | # Classes | Missing |
|---|---:|---:|---:|---:|
| P19 | 38,803 | 34 | 2 (binary)   | 94.9 % |
| P12 | 11,988 | 36 | 2 (binary)   | 88.4 % |
| PAM | 5,333  | 17 | 8 (activity) | 60.0 % |

### Acquisition

- **PhysioNet / Human Activity** — auto-downloaded by the code.
- **USHCN** — use the preprocessed `small_chunked_sporadic.csv` from [GRU-ODE-Bayes](https://github.com/edebrouwer/gru_ode_bayes).
- **MIMIC-III** — request raw data via [PhysioNet](https://physionet.org/content/mimiciii/1.4/) (credentialed), then run the [Neural Flows preprocessing](https://github.com/mbilos/neural-flows-experiments/blob/master/nfe/experiments/gru_ode_bayes/data_preproc/mimic_prep.ipynb).
- **P12 / P19 / PAM** — use the Raindrop-processed splits ([P19](https://doi.org/10.6084/m9.figshare.19514338.v1), [P12](https://doi.org/10.6084/m9.figshare.19514341.v1)).

---

## 🛠 Installation

Tested on **Python 3.10+** and **PyTorch 2.0+**.

```bash
git clone https://github.com/Meaningfull9502/QuITE.git
cd QuITE
pip install -r requirements.txt
pip install mamba-ssm  # only required for the S-Mamba backbone
```

---

## ⚡ Quick Start

```bash
# QuITE + iTransformer on PhysioNet (forecasting, 24 → 24)
python train_forecasting.py --dataset physionet --history 24 \
    --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
    --batch_size 64 --lr 1e-3 --seed 1 --gpu 0 \
    --irr_emb --model itransformer --mode quite
```

```bash
# QuITE++ on the same setting
python train_quite_plus.py --dataset physionet --history 24 \
    --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 2 \
    --batch_size 64 --lr 1e-3 --seed 1 --gpu 0
```

```bash
# QuITE + PatchTST on P19 (classification)
python train_classification.py --dataset P19 \
    --patch_size 3.75 --stride 3.75 --hid_dim 64 --nhead 4 --nlayer 3 \
    --batch_size 64 --lr 1e-3 --epoch 1000 --gpu 0 \
    --irr_emb --model patchtst --mode quite
```

> Per-dataset hyperparameters follow paper Table C.1 and are encoded in `jobs/*.sh`. Run `python <script>.py --help` for the full argument list.

---

## 🔁 Reproducing Paper Results

```bash
bash jobs/run_forecasting.sh     # Table 2  — 6 backbones × 12 (dataset, horizon) × 5 seeds
bash jobs/run_quite_plus.sh      # Table 4  — QuITE++ on 12 settings × 5 seeds
bash jobs/run_classification.sh  # Table 3  — 6 backbones × 3 datasets
```

All runs use Adam, `lr=1e-3`, **patience=50**, **seeds {1..5}**, **MSE loss** (forecasting), **CE loss** (classification) — paper §6.1.

---

## 🧱 Supported Backbones

> 🔑 **Key rule.** When QuITE is plugged into any of the six MTS backbones, we fix
> `--hid_dim 64` and `--nhead 4` (paper §6.1). This isolates the gains to QuITE
> itself rather than to extra backbone capacity. Per-backbone `--nlayer` follows
> paper Appendix B.1 and is set automatically by `jobs/*.sh`. `--nhead` is N/A
> for the non-attention backbones (PatchMixer / TMix / S-Mamba).

| Family | `--model` | Token | `--nlayer` | `--nhead` | Standalone `--hid_dim` | QuITE-equipped `--hid_dim` |
|---|---|---|:---:|:---:|:---:|:---:|
| **Patch** | `patchtst` | per-patch (Transformer) | 3 | 4 | 256 | **64** |
| **Patch** | `patchmixer` | per-patch (CNN, single-layer) | 1 | — | 256 | **64** |
| **Patch** | `tmix` | per-patch (MLP, TSMixer-style) | 2 | — | 128 | **64** |
| **Variate** | `itransformer` | per-variable (inverted Transformer) | 3 | 4 | 512 | **64** |
| **Variate** | `s_mamba` | per-variable (bidirectional Mamba) | 2 | — | 256 | **64** |
| **Hybrid** | `timexer` | per-patch + per-variable exogenous | 3 | 4 | 256 | **64** |

For **classification**, all six backbones run at `--hid_dim 64` in both standalone and QuITE-equipped form (paper Appendix B.1).

> **TimeXer.** As the hybrid backbone, TimeXer instantiates two embeddings internally (`patch_embedding` + `variate_embedding`); the chosen `--mode` is applied to both.

### QuITE++ per-(dataset, horizon) settings

QuITE++ is tuned per dataset via grid search over `--hid_dim ∈ {32, 64}`, `--nlayer ∈ {1, 2, 3}`, `--nhead ∈ {1, 2, 4, 8}` (paper §6.1). The selected values used to produce paper Table 4 are baked into `jobs/run_quite_plus.sh`:

| Dataset | Horizon | `--hid_dim` | `--nlayer` | `--nhead` |
|---|---|:---:|:---:|:---:|
| **Activity**  | 3000 → 1000 | 64 | 3 | 8 |
| **Activity**  | 2000 → 2000 | 64 | 3 | 4 |
| **Activity**  | 1000 → 3000 | 64 | 2 | 2 |
| **USHCN**     | 24 → 1  | 32 | 1 | 2 |
| **USHCN**     | 24 → 6  | 32 | 1 | 4 |
| **USHCN**     | 24 → 12 | 64 | 2 | 2 |
| **PhysioNet** | 12 → 36 | 64 | 3 | 2 |
| **PhysioNet** | 24 → 24 | 64 | 2 | 4 |
| **PhysioNet** | 36 → 12 | 64 | 1 | 4 |
| **MIMIC-III** | 12 → 36 | 32 | 3 | 4 |
| **MIMIC-III** | 24 → 24 | 32 | 3 | 4 |
| **MIMIC-III** | 36 → 12 | 32 | 1 | 4 |

---

## 🎛 Embedding Modes (`--mode`)

| `--mode` | Meaning | Paper | Pair with `--irr_emb`? |
|---|---|---|:---:|
| **`quite`** | **QuITE** — query-based irregular embedding (main method) | Eq. 5-13 | ✅ |
| `mean` | Mean Pooling baseline | Table 5 | ✅ |
| `mtand` | mTAND attention baseline | Table 5 | ✅ |
| `add` | value + time embedding | Table 5 | ❌ |
| `concat` | value ‖ time embedding | Table 5 | ❌ |
| `False` | vanilla backbone embedding (no time conditioning) | — | ❌ |

---

## 📜 Citation

```bibtex
@inproceedings{lim2026quite,
  title         = {Qu{ITE}: Query-based Irregular Time-series Embedding},
  author        = {Lim, JungHoon},
  booktitle     = {Forty-third International Conference on Machine Learning},
  year          = {2026},
  url           = {https://openreview.net/forum?id=ILQGHFvEoo},
  eprint        = {2605.28166},
  archivePrefix = {arXiv}
}
```

---

## 🙏 Acknowledgements

Built on top of [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN), [Raindrop](https://github.com/mims-harvard/Raindrop), [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba), [Hi-Patch](https://github.com/qianlima-lab/Hi-Patch), and [PyOmniTS](https://github.com/Ladbaby/PyOmniTS).
