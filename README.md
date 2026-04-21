Official implementation of "Photonic Flow Matching". (Coming soon.)
<div align="center">
  <img src="assets/teaser.png" width="100%">

<h1>Photonic Flow Matching</h1>

**Nanxing Chen***, **Fenglei Wang***, **Shuo Wang***, **Jintian Hu*#**, **Yuxiang Sun**, **Chuang Yang**,
**Geyang Qu**, **Xuan Yu**, **Jiangyi Li**, **Jie Xhang**, **Qingbo Yang**, **Kairui Cao#**,
**Jun Guan**, **Shengjie Wang**, **Chaoran Huang**, **Yubin Fan**, **Qinghai Song#**

</div>

<div align="center">

[![Paper]
[![PDF]
[![Project]
[![License]
</div>

https://github.com/user-attachments/assets/34583132-3b91-495b-ac9e-aa05286136ec

-----

### 🗺️ Meet Photonic Flow Matching! We've built a optoelectronic generative model for Text-to-Image! 🏗️🌍

Photonic Flow Matching has focused on:

- **Wave-optics generative law**: Architecturally unifies coordinate grounding, dense geometric cues, and long-range drift correction within a single streaming framework through anchor context, pose-reference window, and trajectory memory.
- **High-Efficiency Streaming Inference**: A feed-forward architecture with paged KV cache attention, enabling stable inference at ~20 FPS on 518×378 resolution over long sequences exceeding 10,000 frames.
- **State-of-the-Art Reconstruction**: Superior performance on diverse benchmarks compared to both existing streaming and iterative optimization-based approaches.

---

# ⚙️ Quick Start

## Installation

**1. Create conda environment**

```bash
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map
```

**2. Install PyTorch (CUDA 12.8)**

```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

> For other CUDA versions, see [PyTorch Get Started](https://pytorch.org/get-started/locally/).

**3. Install lingbot-map**

> Photonic Flow Matching (PFM) formulates image generation as a physically realizable transport process under paraxial wave optics.  
> In PFM, a sample is treated as the full normalized intensity distribution, while free-space propagation and programmable phase modulation induce a transport map on the sample space.

---

## Announcements

- **[YYYY-MM-DD]** Initial release of the PFM codebase.
- **[YYYY-MM-DD]** Pretrained checkpoints released.
- **[YYYY-MM-DD]** Paper and supplementary information released.

---

## Overview

This repository provides the implementation of **Photonic Flow Matching (PFM)**, a wave-optics-grounded framework for continuity-driven generative transport.

The core idea of PFM is:

- the normalized optical intensity evolves according to a continuity law;
- the transport velocity is induced by the optical phase gradient;
- free-space propagation and phase modulation together define a physically constrained generative process.

This repository includes:

- training and inference code for PFM;
- simulated optical propagation modules;
- configurable phase modulation layers;
- demo scripts for sample generation;
- evaluation scripts for common generative metrics.

---

## Quickstart

We provide the software implementations of:

- **PFM training**
- **PFM inference / generation**
- **Simulated optical propagation**
- **Phase-mask-conditioned transport**
- **Evaluation pipeline**

---

## Installation

### 1. Create Conda Environment

```bash
conda create --name pfm python=3.11
conda activate pfm
