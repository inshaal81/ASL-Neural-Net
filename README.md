# ASL Neural Net

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inshaal81/SignSync/blob/main/notebooks/ASL_Neural_Net_Colab.ipynb)

A deep neural network built from scratch with NumPy to recognize American Sign Language (ASL) alphabet gestures. This project demonstrates fundamental deep learning concepts without relying on high-level frameworks like TensorFlow or PyTorch.

## Overview

ASL Neural Net classifies static hand gestures representing 24 letters of the ASL alphabet (excludes J and Z which require motion). The model achieves **77.97% test accuracy** using a fully-connected neural network.

## Features

- **From Scratch Implementation** - Built with NumPy to understand the fundamentals
- **Multi-class Classification** - 24 letter classes using softmax activation
- **Configurable Architecture** - Easy to experiment with different layer sizes
- **Model Persistence** - Save and load trained models
- **Comprehensive Evaluation** - Confusion matrices, per-class metrics, error analysis

## Architecture

```
INPUT (784) → DENSE(128) → RELU → DENSE(64) → RELU → DENSE(24) → SOFTMAX
```

**Total Parameters:** 103,320

**Key Components:**
- He initialization for better convergence
- Categorical cross-entropy loss
- Mini-batch gradient descent
- Per-image normalization for distribution shift handling

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 100.00% |
| Test Accuracy | **77.97%** |
| Classes | 24 (A-Y, excluding J, Z) |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/inshaal81/SignSync.git
cd ASL-Neural-Net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download Sign Language MNIST from Kaggle:
https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Place the CSV files in `data/datasets/`:
- `sign_mnist_train.csv`
- `sign_mnist_test.csv`

### Train

```bash
python scripts/train.py
```

### Evaluate

```bash
python scripts/evaluate.py
```

## Project Structure

```
ASL-Neural-Net/
├── src/
│   ├── model.py          # DeepNeuralNetwork class
│   ├── utils.py          # Data loading, normalization
│   └── visualizations.py # Plotting utilities
├── scripts/
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── notebooks/
│   └── ASL_Neural_Net_Colab.ipynb  # Interactive demo (Google Colab)
└── data/datasets/        # Dataset files (not committed)
```

## License

MIT License
