# 🚗 LightDrive: Real-time Autonomous Driving VLM & VLA Framework

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-F9AB00.svg)

**LightDrive** is a highly efficient Vision-Language-Action (VLA) framework for autonomous driving, built upon the NuScenes dataset. 

Moving away from the computationally expensive Bird's Eye View (BEV) transformations relied upon by conventional autonomous driving models, we introduce **Ray-centric Fusion**—a technique that directly injects camera geometry into 2D image patches. This repository provides the complete data generation and training pipelines for both our **VLM (Vision-Language Model)** for scene understanding, and our **VLA (Vision-Language-Action)** model for direct vehicle control (action and speed).

## ✨ Key Features

* **Hierarchical Teacher-LLM Data Engine**: Utilizes `LLaVA-1.5` for individual multi-camera perception and `Qwen2.5` for global reasoning to automatically generate high-quality text and action labels for autonomous driving.
* **BEV-free Ray-centric Fusion**: Transforms intrinsic and extrinsic camera matrices into multi-dimensional vectors and injects them directly into vision tokens, embedding 3D spatial awareness without computational bottlenecks.
* **Multi-Task VLA Architecture**: Employs a Dual-Head structure (Action Classification + Speed Regression) leveraging the hidden states of the Causal LLM's final generated token. This allows the model to simultaneously output scene descriptions and direct vehicle control commands with a latency of under 100ms.
* **Optimized for Multi-GPU**: Fully supports PyTorch Distributed Data Parallel (DDP), Gradient Accumulation, and Automatic Mixed Precision (AMP, bfloat16) for fast and stable distributed training.

---

## 📂 Repository Structure

This repository consists of three core components:

```text
├── data_engine/
│   └── generate_data.py    # [Data] Auto-generates driving label data (JSON) using LLaVA & Qwen
├── models/
│   ├── train_vlm.py        # [VLM] Training script for text-based scene description and reasoning
│   └── train_vla.py        # [VLA] Training script for text reasoning + action prediction + speed control
├── scripts/
│   └── train_run.sh        # Example Bash script for 8-GPU distributed training
└── README.md
