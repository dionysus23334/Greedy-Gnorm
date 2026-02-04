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

---

## Reproducing Plots and Tables

All experiments in this repository are implemented as **Jupyter notebooks (`.ipynb`)**.
Each notebook can be executed independently unless otherwise specified.

Unless noted otherwise, notebooks are expected to be run from the repository root directory.

---

### Data Preparation

#### `data_processing.ipynb`

**Purpose**
Prepare datasets and model inputs for downstream pruning experiments.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* In-memory processed datasets
* No files are written to disk

---

### Gradient Analysis and Diagnostics

#### `gradient_test.ipynb`

**Purpose**
Validate gradient computation and inspect numerical stability for attention parameters.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Printed diagnostic information
* Gradient sanity-check results

---

### Greedy G-norm–Based Pruning

#### `greedy_pruning.ipynb`

**Purpose**
Perform greedy attention head pruning based on gradient matrix norms (G-norm).

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Accuracy after each pruning iteration
* Pruned attention head masks
* Optional figures visualizing pruning trajectories

---

#### `InverseGreedyGnormPruning.ipynb`

**Purpose**
Apply inverse greedy pruning based on G-norm criteria.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Accuracy under inverse pruning
* Final pruning configurations

---

### Baseline and Comparative Pruning Methods

#### `random_pruning.ipynb`

**Purpose**
Provide a random pruning baseline for comparison.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Accuracy under random pruning
* Baseline comparison results

---

#### `AE_pruning.ipynb`

**Purpose**
Perform attention head pruning based on activation energy (AE).

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Accuracy after AE-based pruning
* Pruning masks

---

#### `AE_inverse_pruning.ipynb`

**Purpose**
Perform inverse pruning based on activation energy.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Accuracy under inverse AE pruning
* Pruning configurations

---

### Multi-Model Experiments

#### `four_models.ipynb`

**Purpose**
Evaluate pruning behavior across multiple Transformer architectures.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Accuracy comparisons across models
* Architecture-specific pruning results

---

#### `greedyprune(BERT,ALBERT,ROBERTA,XLM-ROBERTA).ipynb`

**Purpose**
Apply greedy G-norm pruning to multiple Transformer models in a unified setting.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Consolidated pruning results across models
* Comparative accuracy statistics

---

### Custom and Numerical Stability Experiments

#### `customBERTpruning.ipynb`

**Purpose**
Demonstrate customized pruning strategies for BERT models.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Accuracy metrics
* Custom pruning masks

---

#### `solve_underflow.ipynb`

**Purpose**
Investigate and mitigate numerical underflow issues encountered during pruning.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Diagnostic outputs
* Numerical stability validation results

---

### Result Aggregation and Visualization

#### `show_pruning_result.ipynb`

**Purpose**
Visualize pruning trajectories and performance trends.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Figures summarizing pruning behavior
* Plots saved to `figures/` (if enabled)

---

#### `Pruning_summary.ipynb`

**Purpose**
Summarize and aggregate results from multiple pruning strategies.

**Usage**

```text
Open and run all cells in the notebook.
```

**Outputs**

* Consolidated summary tables
* Final comparison figures

---

## Outputs

The notebooks generate outputs corresponding to the experimental results reported in the manuscript, including:

* Accuracy before and after attention head pruning
* Pruning trajectories under different strategies
* Comparative results across Transformer architectures
* Diagnostic plots and summary figures

Outputs are either displayed directly within the notebooks or saved to the following directories (if enabled):

* `figures/`
* `experiments_results/`

---


