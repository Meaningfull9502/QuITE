<div align="center">

# [ICML 2026] QuITE: Query-Based Irregular Time Series Embedding

[![Conference](https://img.shields.io/badge/ICML-2026-1f6feb?style=flat-square)](https://icml.cc/Conferences/2026)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

**Official PyTorch implementation** of  *QuITE: Query-Based Irregular Time Series Embedding* (ICML 2026).

A plug-and-play **input-embedding** module that lets any standard MTS backbone — PatchTST, PatchMixer, TMix, iTransformer, S-Mamba, TimeXer — handle **Irregular Multivariate Time Series (IMTS)** without architectural changes or artificial value generation.

📄 [**Paper (OpenReview)**](https://openreview.net/forum?id=ILQGHFvEoo) · 💻 [**Code**](https://github.com/Meaningfull9502/QuITE) · 📜 [**Citation**](#-citation)

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

| Forecasting | # Samples | # Vars | Missing |   | Classification | # Samples | # Vars | # Classes | Missing |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| Human Activity | 5,400 | 12 | 75.0 % |   | P19 | 38,803 | 34 | 2 | 94.9 % |
| USHCN          | 26,736 | 5 | 77.9 % |   | P12 | 11,988 | 36 | 2 | 88.4 % |
| PhysioNet      | 12,000 | 36 | 88.4 % |   | PAM | 5,333 | 17 | 8 | 60.0 % |
| MIMIC-III      | 23,457 | 96 | 96.7 % |   |     |        |    |   |        |

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
    --patch_size 3.75 --stride 3.75 --hid_dim 64 --nhead 2 --nlayer 3 \
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

QuITE-equipped models are standardized to `--hid_dim 64` and `--nhead 4` (forecasting) / `--nhead 2` (classification). The per-backbone `--nlayer` follows paper Appendix B.1 and is set automatically by the batch scripts.

| Family | `--model` | Token | `--nlayer` |
|---|---|---|:---:|
| **Patch** | `patchtst` | per-patch (Transformer) | 3 |
| **Patch** | `patchmixer` | per-patch (CNN, single-layer) | 1 |
| **Patch** | `tmix` | per-patch (MLP, TSMixer-style) | 2 |
| **Variate** | `itransformer` | per-variable (inverted Transformer) | 3 |
| **Variate** | `s_mamba` | per-variable (bidirectional Mamba) | 2 |
| **Hybrid** | `timexer` | per-patch + per-variable exogenous | 3 |

> **QuITE++.** Tuned per dataset via grid search over `--hid_dim ∈ {32, 64}`, `--nlayer ∈ {1, 2, 3}`, `--nhead ∈ {1, 2, 4, 8}` (paper §6.1).

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
  title     = {Qu{ITE}: Query-based Irregular Time-series Embedding},
  author    = {Lim, JungHoon},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=ILQGHFvEoo}
}
```

---

## 🙏 Acknowledgements

Built on top of [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN), [Raindrop](https://github.com/mims-harvard/Raindrop), [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba), [Hi-Patch](https://github.com/qianlima-lab/Hi-Patch), and [PyOmniTS](https://github.com/Ladbaby/PyOmniTS).
