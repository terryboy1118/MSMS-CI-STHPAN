# MSMS-CI-STHPAN

Quantitative stock selection remains a complex challenge due to the multi-structural and non-stationary nature of financial markets. To address this, we propose **MSMS-DTW** (Market-Segmented Multi-Scale Dynamic Time Warping), a similarity-fusion method that enhances temporal alignment and interpretability.

MSMS-DTW segments the market timeline using the **turning points of a benchmark stock** and restricts DTW computation to stocks within the same segment, thus preserving **phase-specific dynamics**.

By incorporating **multi-scale segmentation** and **DTW normalization**, our method captures localized synchronization patterns and mitigates the temporal-distortion issues present in global DTW. We further integrate MSMS-DTW into the **CI-STHPAN** (Channel-Independent Spatio-Temporal Hypergraph Pretrained Attention Network) framework, leveraging channel-wise independence and hypergraph structures for robust stock-relation modeling.

> 📈 **On the NASDAQ dataset**, the enhanced model achieves  
> **Internal Rate of Return (IRR):** 0.92297  
> **Sharpe Ratio (SR):** 2.20432  
> — significantly outperforming diverse baselines.

---

## 🔍 Keywords

`Stock Selection` · `Dynamic Time Warping` · `Market Segmentation` · `Hypergraph Neural Network` · `Transformer`

---

## 📚 Table of Contents

- [📦 Installation](#-installation)  
- [🚀 Quick Example](#-quick-example)  
- [🧠 Model Overview](#-model-overview)  
- [📊 Experimental Results](#-experimental-results)  
- [📁 Project Structure](#-project-structure)

---

## 📦 Installation

```bash
# 1. Create and activate conda env
conda create -n ci-sthpann python=3.10.12
conda activate ci-sthpann

# 2. Install PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install PyTorch Geometric (PyG)
conda install pyg -c pyg

# 4. Other Python packages
pip install -r requirements.txt
```

---

## 🚀 Quick Example

Demonstration of how to pre-train and fine-tune the model on hypergraphs constructed using Wiki-based relations (NASDAQ dataset).

```bash
# Pre-train the model
cd CI-STHPAN_self_supervised
bash scripts/pretrain/pre_graph_dtw.sh

# Fine-tune the model (choose script as needed)
bash scripts/finetune/[28]graph_dtw.sh
```

---

## 🧠 Model Overview

<p align="center">
  <img src="figures/model_architecture.png" alt="Model Overview" width="800"/>
</p>

- **CI-STHPAN** serves as the base architecture for time-series and graph relational modeling.
- **MSMS-DTW** handles market-aware segmentation and similarity measurement.
- The constructed **hypergraph** captures multi-stock relations with higher-order dynamics.

---

## 📊 Experimental Results

| Dataset | Model                 | IRR     | Sharpe Ratio |
|---------|------------------------|---------|---------------|
| NASDAQ | CI-STHPAN (baseline)  | 0.67712 | 1.32892       |
| NASDAQ | MSMS-CI-STHPAN (ours) | **0.92297** | **2.20432**       |
| NYSE   | CI-STHPAN (baseline)  | 0.61387 | 1.15443       |
| NYSE   | MSMS-CI-STHPAN (ours) | **0.83225** | **1.90761**       |

> 🔬 Metrics include Internal Rate of Return (IRR) and Sharpe Ratio (SR). Experiments are conducted under fixed seeds and repeated runs.
---

For any questions, please contact `lin.syuan@example.com`. Contributions and pull requests are welcome!
