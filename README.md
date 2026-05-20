# SLFP-ViT: Small Logarithmic Floating-Point Vision Transformer

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.0+](https://img.shields.io/badge/pytorch-1.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-DLUT-yellow.svg)](https://opensource.org/licenses/DLUT)

This project implements the **ViT-B/16 (Vision Transformer)** architecture based on **SLFP (Small Logarithmic Floating-Point)** arithmetic using Python and PyTorch. It is designed to simulate low-precision floating-point operations for efficient Transformer acceleration on hardware like FPGAs.

---

## 🌟 Project Specifications

*   **Supported Model**: `ViT-B/16` (Vision Transformer Base, Patch Size 16).
*   **Activation Function**: `GELU` (optimized for Transformer architectures).
*   **Optimizers**: `Adam`, `SGD`.
*   **Target Dataset**: `CIFAR-100`.
*   **Core Arithmetic**: Small Logarithmic Floating-Point (SLFP) simulation.

---

## 📖 Theoretical Background

The implementation of SLFP arithmetic in this project is inspired by the following research:

1.  *"Small Logarithmic Floating-Point Multiplier Based on FPGA and Its Application on MobileNet"*
2.  *"FPGA-Friendly Architecture of Processing Elements For Efficient and Accurate CNNs"*

These papers provide the foundation for hardware-friendly processing elements that this repository simulates in a software environment.

---

## 🚀 Getting Started

### Prerequisites
Ensure your environment meets the following requirements:
- **Python**: 3.6+
- **PyTorch**: 1.0+
- **torchvision**: 0.2.2+

### Installation

---

## 📂 Project Structure

```text
.
├── models/             # ViT-B/16 model definitions with SLFP layers
├── training/           # Training scripts for CIFAR-100
├── core/               # SLFP quantization logic and custom functions
│   ├── gelu_slfp.py    # SLFP-optimized GELU implementation
│   └── quant_utils.py  # Quantization helpers
├── train_cifar100.py   # Main entry point for training
└── eval_cifar100.py    # Main entry point for evaluation
```

## Accuracy of Quantized vit_b_16 functions on Cifar100.

Accuracy of Quantized vit_b_16 functions on Cifar100.

|      Model      |      Dataset     |      W-bit     |     A-bit     |        Top-1 Accuracy        |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Vit_b_16  | cifar100  |   32bits(FP32)   |    32bits(FP32)  |              92.48%              |
| Vit_b_16  | cifar100  |    8bits(SLFP)    |    8bits(SLFP)   |              91.68%              |

## Weight model download

[MLP-PEFT-acc0.9248-Epoch20.pth](https://fs-im-kefu.7moor-fs1.com/ly/4d2c3f00-7d4c-11e5-af15-41bf63ae4ea0/1741268997495/MLP-PEFT-acc0.9248-Epoch20.pth)   
[ViT-b16-CIFAR100-Epoch10-PEFT-SLFP-QAT-acc0.9168.pth](https://fs-im-kefu.7moor-fs1.com/ly/4d2c3f00-7d4c-11e5-af15-41bf63ae4ea0/1741269345629/ViT-b16-CIFAR100-Epoch10-PEFT-SLFP-QAT-acc0.9168.pth)

## Reference

https://github.com/DLUT-IIS/CNN_Accelerator
https://github.com/ZipperLii/Fine-tuning-of-ViT-B16-on-CIFAR-100








