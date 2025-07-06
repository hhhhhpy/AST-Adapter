# AST-Adapter
Official implementation of [AST-Adapter: Parameter-Efficient Video-to-video Transfer Learning with Adaptive Spatiotemporal Information Bias]



![图片2](https://github.com/user-attachments/assets/eafdc0f4-f43c-4e14-ab6a-de536f0fbe6e)


# Overview
AST-Adapter is a parameter-efficient video-to-video transfer learning method that leverages the inherent spatiotemporal properties of video data to achieve superior performance across diverse video datasets, while requiring the tuning of only a minimal number of parameters.
The core contributions of our work include：

▶️ A novel metric to quantify spatiotemporal information bias within video datasets.

▶️ An adaptive selection mechanism to select the optimal adapter for each layer.

# Installation

```bash
conda create -n ast python=3.8
pip install -r requirements.txt
```

# Datasets 
We follow [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/DATASET.md) to prepare the datasets. 
# Metric Calculation
```bash
CUDA_VISIBLE_DEVICES=0 bash metric.sh
```
# Model Evaluation
```bash
python model.py
```

