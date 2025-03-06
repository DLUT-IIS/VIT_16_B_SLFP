# Project Description

Implement a Vision Transformer based on SLFP (Small Logarithmic Floating-Point) using python.  

**Note:** Supported network models: vit_b_16.;   

**Note:** Supported activation functions include: gelu.;  

**Note:** Optimisers include:Adam, SGD.;  

**Note:** Datasets include: Cifar100.  


*Related articles on SLFP:*

*"Small Logarithmic Floating-Point Multiplier Based on FPGA and Its Application on MobileNet"*  

*"FPGA-Friendly Architecture of Processing Elements For Efficient and Accurate CNNs"*  

****

# User Guide

Please replace ’https://github.com/DLUT-IIS/VIT_16_B_SLFP' with ‘your_github_name/your_repository’ in all links. ’

## Pre-development Configuration Requirement

Dependencies:
- Python 3.6+
- PyTorch 1.0+
- torchvision 0.2.2+

## **Installation steps**

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone 

```sh
git clone https://github.com/DLUT-IIS/VIT_16_B_SLFP
```

## Accuracy of Quantized vit_b_16 functions on Cifar100.

Accuracy of Quantized vit_b_16 functions on Cifar100.

|      Model      |      Dataset     |      W-bit     |     A-bit     |        Top-1 Accuracy        |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Vit_b_16  | cifar100  |   32bits(FP32)   |    32bits(FP32)  |              92.48%              |
| Vit_b_16  | cifar100  |    8bits(SLFP)    |    8bits(SLFP)   |              91.68%              |

# Reference

https://github.com/DLUT-IIS/CNN_Accelerator

https://github.com/ZipperLii/Fine-tuning-of-ViT-B16-on-CIFAR-100

# Author
Shize Zhang







