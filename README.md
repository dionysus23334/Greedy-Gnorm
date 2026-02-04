# Greedy-Gnorm: A Gradient Matrix Norm-Based Alternative to Attention Entropy for Head Pruning

## Getting Started

Clone this repository and change into the project directory:

```bash

git clone https://github.com/dionysus23334/Greedy-Gnorm.git
cd Greedy-Gnorm

```

All experiments are implemented in Jupyter notebooks (`.ipynb`).

We recommend using **Python 3.9+**.

### Environment Setup

```bash
conda create -n prune-test python=3.9 -y
conda activate prune-test
```
#### CPU / GPU Execution

This implementation supports both CPU and GPU execution.

#### GPU Execution (Recommended)

If a CUDA-enabled GPU is available, we strongly recommend installing the GPU version of PyTorch and running the notebooks on GPU.

GPU execution significantly reduces runtime, as the pruning procedure involves repeated forward and backward passes through the model.

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install transformers==4.36.2 datasets==2.16.1 textpruner==1.1.post2 tqdm==4.66.1 matplotlib==3.8.2 pandas==2.1.4
```

Alternatively, core dependencies are listed in `requirements.txt`.
