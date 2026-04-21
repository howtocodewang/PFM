Official implementation of "Photonic Flow Matching". (Coming soon.)
<div align="center">
  <img src="assets/teaser.png" width="100%">

<h1>Photonic Flow Matching</h1>

**Nanxing Chen***, **Fenglei Wang***, **Shuo Wang***, **Jintian Hu*#**, **Yuxiang Sun**, **Chuang Yang**,
**Geyang Qu**, **Xuan Yu**, **Jiangyi Li**, **Jie Xhang**, **Qingbo Yang**, **Kairui Cao#**,
**Jun Guan**, **Shengjie Wang**, **Chaoran Huang**, **Yubin Fan**, **Qinghai Song#**

</div>
<div align="center">
[**Paper**](YOUR_PAPER_LINK) | [**Project Page**](YOUR_PROJECT_PAGE) | [**Supplementary Information**](YOUR_SI_LINK) | [**Pretrained Weights**](YOUR_CHECKPOINT_LINK)
</div>

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
