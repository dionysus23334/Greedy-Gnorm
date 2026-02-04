# Greedy-Gnorm: A Gradient Matrix Norm-Based Alternative to Attention Entropy for Head Pruning
## Table of Contents
[Getting Started](#getting-started)  
&nbsp;&nbsp;↳ [Environment Setup](#environment-setup)  
&nbsp;&nbsp;↳ [CPU / GPU Execution](#cpu--gpu-execution)

[Reproducing Plots and Tables](#reproducing-plots-and-tables)  
&nbsp;&nbsp;↳ [Data Preparation](#data-preparation)  
&nbsp;&nbsp;↳ [Gradient Analysis and Diagnostics](#gradient-analysis-and-diagnostics)  
&nbsp;&nbsp;↳ [Greedy G-norm–Based Pruning](#greedy-g-normbased-pruning)  
&nbsp;&nbsp;↳ [Baseline and Comparative Pruning Methods](#baseline-and-comparative-pruning-methods)  
&nbsp;&nbsp;↳ [Multi-Model Experiments](#multi-model-experiments)  
&nbsp;&nbsp;↳ [Custom and Numerical Stability Experiments](#custom-and-numerical-stability-experiments)  
&nbsp;&nbsp;↳ [Result Aggregation and Visualization](#result-aggregation-and-visualization)

[Outputs](#outputs)  
&nbsp;&nbsp;↳ [Random Pruning Results](#random-pruning-results-excel-files)  
&nbsp;&nbsp;↳ [Accuracy vs. Pruned Heads](#accuracy-vs-pruned-heads-csv-files)  
&nbsp;&nbsp;↳ [Notes on Reproducibility](#notes-on-reproducibility)

[Figures and Visualizations](#figures-and-visualizations)  
&nbsp;&nbsp;↳ [Illustrative Figures](#illustrative-figures-manuscript-diagrams)  
&nbsp;&nbsp;↳ [Experimental Result Visualizations](#experimental-result-visualizations)

---

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



### Greedy Gnorm Based Pruning

#### `greedy_pruning.ipynb` — Greedy Gradient-Norm Pruning (Prototype)

This notebook provides an **early exploratory implementation** of greedy attention head pruning based on **Q/K/V gradient norms**, primarily used for **method validation and debugging**, rather than final experimental reporting.


#### `greedyprune(BERT,ALBERT,ROBERTA,XLM-ROBERTA).ipynb` — Greedy Gradient-Norm-Based Pruning

This notebook implements a **greedy attention head pruning algorithm based on gradient norm statistics**, where attention heads are iteratively removed according to their estimated contribution to the model output.

**Purpose**
Perform **greedy attention head pruning** using a **QKV gradient-norm–based importance score**, evaluated after each pruning step.

**Method Overview**

At each pruning iteration:

1. Compute per-head gradient norms for **Query (Q), Key (K), and Value (V)** projection matrices.
2. Aggregate head importance as the **product of Q, K, and V gradient norms**, followed by normalization.
3. Select the **least important remaining attention head** in a greedy manner.
4. Prune the selected head using a structured attention head mask.
5. Re-evaluate downstream task accuracy after pruning.

To correctly handle **previously pruned heads**, the implementation expands reduced-weight tensors back to the original dimensionality by inserting zero blocks, ensuring consistent per-head norm estimation throughout the pruning process.

**Supported Models**

* BERT
* ALBERT
* RoBERTa
* XLM-RoBERTa

Each model uses its corresponding pretrained checkpoint and task-specific dataset.

**Usage**

```text
Open and run all cells in the notebook sequentially.
```

The pruning procedure is fully self-contained and does not rely on precomputed importance scores.

**Results**

The notebook produces the following output files:

* `pruned_heads_accuracy(BERT)basedonGreedyGnorm.csv`
* `pruned_heads_accuracy(ALBERT)basedonGreedyGnorm.csv`
* `pruned_heads_accuracy(ROBERTA)basedonGreedyGnorm.csv`
* `pruned_heads_accuracy(XLM_ROBERTA)basedonGreedyGnorm.csv`

Each CSV file records downstream task accuracy as a function of the number of pruned attention heads under greedy gradient-norm–based pruning.






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




#### `AE_pruning.ipynb` — Attention Entropy Pruning

This notebook implements **Attention Entropy (AE)–based attention head pruning** for Transformer models, where heads with higher AE scores are pruned first.

**Purpose**
Perform AE-based greedy pruning and evaluate downstream performance degradation.

**Usage**

```text
Open and run all cells in the notebook.
```

**Results**

* `pruned_heads_accuracy(BERT)basedonAE.csv`
* `pruned_heads_accuracy(ALBERT)basedonAE.csv`
* `pruned_heads_accuracy(ROBERTA)basedonAE.csv`
* `pruned_heads_accuracy(XLM_ROBERTA)basedonAE.csv`

Each file records downstream accuracy as a function of the number of pruned attention heads.




#### `AE_inverse_pruning.ipynb` — Attention Entropy Inverse Pruning


This notebook implements **inverse Attention Entropy (AE)–based attention head pruning** for Transformer models, where heads are removed in ascending order of their AE scores.

**Purpose**
Perform inverse pruning based on activation energy.

**Usage**

```text
Open and run all cells in the notebook.
```

**Results**

* `pruned_heads_accuracy(BERT)basedonAEinverse.csv`
* `pruned_heads_accuracy(ALBERT)basedonAEinverse.csv`
* `pruned_heads_accuracy(ROBERTA)basedonAEinverse.csv`
* `pruned_heads_accuracy(XLM_ROBERTA)basedonAEinverse.csv`

Each file records downstream accuracy as a function of the number of pruned attention heads.





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

This repository includes precomputed experimental results corresponding to the pruning experiments reported in the manuscript.
These files allow reproducing plots and tables **without rerunning the full pruning procedure**, which can be computationally expensive.

### Random Pruning Results (Excel Files)

The following `.xlsx` files contain results from **random attention head pruning experiments**, used as baselines for comparison.

**Files**

* `ALBERTrandompruned_heads_data_25x13_1group.xlsx`
* `BERTrandompruned_heads_data_50x25_1group.xlsx`
* `BERTrandompruned_heads_data_50x25_2group.xlsx`
* `BERTrandompruned_heads_data_50x25_concated.xlsx`
* `ROBERTArandompruned_heads_data_25x25_1group.xlsx`
* `XLM_ROBERTArandompruned_heads_data_25x25_1group.xlsx`

**Description**

* Each file records accuracy results under randomly selected attention head pruning.
* Different files correspond to different model architectures and experimental groupings.
* These results are used as **random pruning baselines** in comparative analyses.

**Usage**

* Loaded by analysis and visualization notebooks (e.g., `show_pruning_result.ipynb`)
* Used to compute averaged baseline performance curves


### Accuracy vs. Pruned Heads (CSV Files)

The following `.csv` files record **model accuracy as a function of the number of pruned attention heads**, under different pruning strategies.

Each file corresponds to a specific:

* Model architecture
* Pruning criterion



#### ALBERT

* `pruned_heads_accuracy(ALBERT)basedonAE.csv`
* `pruned_heads_accuracy(ALBERT)basedonAEinverse.csv`
* `pruned_heads_accuracy(ALBERT)basedonGreedyGnorm.csv`
* `pruned_heads_accuracy(ALBERT)basedonInverseGreedyGnorm.csv`



#### BERT

* `pruned_heads_accuracy(BERT)basedonAE.csv`
* `pruned_heads_accuracy(BERT)basedonAEinverse.csv`
* `pruned_heads_accuracy(BERT)basedonGreedyGnorm.csv`
* `pruned_heads_accuracy(BERT)basedonInverseGreedyGnorm.csv`



#### RoBERTa

* `pruned_heads_accuracy(ROBERTA)basedonAE.csv`
* `pruned_heads_accuracy(ROBERTA)basedonAEinverse.csv`
* `pruned_heads_accuracy(ROBERTA)basedonGreedyGnorm.csv`
* `pruned_heads_accuracy(ROBERTA)basedonInverseGreedyGnorm.csv`


#### XLM-RoBERTa

* `pruned_heads_accuracy(XLM_ROBERTA)basedonAE.csv`
* `pruned_heads_accuracy(XLM_ROBERTA)basedonAEinverse.csv`
* `pruned_heads_accuracy(XLM_ROBERTA)basedonGreedyGnorm.csv`
* `pruned_heads_accuracy(XLM_ROBERTA)basedonInverseGreedyGnorm.csv`


**Description**

* Each CSV file contains:

  * Number of pruned attention heads
  * Corresponding model accuracy
* Files are organized by **pruning strategy**:

  * Activation Energy (AE)
  * Inverse AE
  * Greedy G-norm
  * Inverse Greedy G-norm

**Usage**

* Used to generate accuracy–pruning curves
* Directly consumed by visualization and summary notebooks (e.g., `show_pruning_result.ipynb`, `Pruning_summary.ipynb`)



### Notes on Reproducibility

* The provided result files allow reproducing figures and tables **without rerunning full pruning experiments**.
* Full experiments can still be reproduced by executing the corresponding notebooks, but this may require substantial computation time, especially on CPU.

---

## Figures and Visualizations

This repository includes both **illustrative figures** used in the manuscript and **experimental result visualizations** generated from pruning experiments.



### Illustrative Figures (Manuscript Diagrams)

The following figures are **conceptual illustrations** included in the manuscript to explain the pruning methodology.
These figures were created using **draw.io** and are **not generated by code**.

**Files**

* `gradientpooling.pdf`
* `gradientpooling.png`
* `AlbertPruning.pdf`
* `AlbertPruning.png`

**Description**

* `gradientpooling.*`:
  Illustrates the gradient pooling mechanism used to compute importance scores for attention heads.
* `AlbertPruning.*`:
  Provides a schematic overview of attention head pruning in ALBERT.

**Notes**

* These figures are **static illustrations** for explanatory purposes.
* They are not produced by running the notebooks and are included directly for completeness and reference.



### Experimental Result Visualizations

The following figures visualize the **empirical results** of attention head pruning experiments.

**Files**

* `PruningResults.pdf`
* `PruningResults.png`
* `RandomPruning.pdf`
* `RandomPruning.png`
* `allfinalsolutions.pdf`
* `allfinalsolutions.png`

**Description**

* `PruningResults.*`:
  Accuracy trends under different pruning strategies.
* `RandomPruning.*`:
  Performance under random attention head pruning (baseline).
* `allfinalsolutions.*`:
  Aggregated comparison of pruning strategies across models.

**Usage**

* These figures are generated or reproduced using result files (`.csv`, `.xlsx`) and visualization notebooks such as:

  * `show_pruning_result.ipynb`
  * `Pruning_summary.ipynb`



### Notes on Reproducibility

* Illustrative figures (`gradientpooling.*`, `AlbertPruning.*`) are **manually designed diagrams** and are not reproduced by code.
* Experimental figures can be regenerated by running the corresponding notebooks, provided that the required result files or full pruning experiments are available.

---

