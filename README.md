# Greedy-Gnorm: A Gradient Matrix Norm-Based Alternative to Attention Entropy for Head Pruning

## Getting Started

All experiments are implemented in Jupyter notebooks (`.ipynb`).

We recommend using **Python 3.9+**.

### Environment Setup

```bash
conda create -n prune-test python=3.9 -y
conda activate prune-test
```

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

```bash
pip install transformers==4.36.2 datasets==2.16.1 textpruner==1.1.post2 tqdm==4.66.1 matplotlib==3.8.2 pandas==2.1.4
```

Alternatively, core dependencies are listed in `requirements.txt`.
