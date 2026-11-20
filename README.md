Growing Feedforward Neural Network (GFNN)
Synaptic Divergence-Guided Topological Adaptation
This repository accompanies research on a Growing Feedforward Neural Network (GFNN), a constructive learning framework that adapts hidden-layer capacity during training using a Synaptic Divergence-guided topology adaptation procedure.

The proposed framework combines:
1. Synaptic Divergence: a neuron-level activation-dispersion measure used to identify representational heterogeneity.
2. Topological Spot Selection: selection of a high-divergence region for localized structural expansion.
3. Function-preserving neuron insertion: insertion of new neurons while preserving the network output at the insertion point, subject to the stated initialization procedure.
4. Fibonacci-regulated growth: regulation of the amount of structural expansion.
Reproducibility note: This README documents the experimental design described in the associated manuscript. The source scripts should be checked before treating every command, dependency, hyperparameter, or random seed below as verified. Exact values should be reported from the scripts, not inferred.

Repository contents
Core implementation
- GFNN architecture and Synaptic Divergence-guided topology adaptation — GFNN_Paper.py
Benchmark scripts
- Iris — Iris.py
- Breast Cancer Wisconsin benchmark configuration — Breast-Cancer-Wisconsin-2.py
- Wine — WINE-RESULT.py
- Digits — Digits-dataset.py
- Additional Breast Cancer Wisconsin implementation — Breast-Cancer-Wisconsin-dataset.py
- Fashion-MNIST experiment — MNIST.py
Comparative methods and evaluation
- Random Forest — Random-Forest.py
- XGBoost — XGBoost.py
- Comparison across 30 runs — Comparisonw-GFNN.py
- Gradient-descent implementation and testing — Gradient-Descent.py
Benchmark datasets
The manuscript describes five classification benchmarks:
Dataset	Samples	Features	Classes
Iris	150	4	3
Breast Cancer Wisconsin	569	30	2
Wine	178	13	3
Digits	1,797	64	10
Fashion-MNIST balanced subset	1,000	784	10


The Fashion-MNIST experiment is described as a class-balanced subset containing 100 images per class.
Experimental protocol described in the manuscript
- 30 independent runs.
- Class-stratified train/test partitions.
- Feature standardization using training-partition statistics only, followed by application of those statistics to the test partition.
- Hyperbolic tangent (tanh) hidden-layer activation.
- Deterministic full-batch gradient descent.
- Fixed-topology MLP baselines with hidden widths 8, 16, 32, and 64.
- Evaluation of accuracy, F1-score, final hidden-layer width, trainable parameter count, training time, inference latency, and paired statistical tests.
The manuscript describes paired Wilcoxon signed-rank and paired Student's t-tests, with Holm–Bonferroni correction for multiple comparisons. Consult the manuscript and source scripts for the exact test implementation and adjusted p-values.
Environment setup
The scripts should be run with a Python environment containing the libraries they import. A provisional dependency list is supplied in requirements.txt; it has not been verified against the source files. After checking every script, pin the tested versions and update that file.

Create and activate a virtual environment:
python -m venv .venv
Windows:
.venv\Scripts\activate
macOS/Linux:
source .venv/bin/activate
Install the provisional dependencies:
python -m pip install --upgrade pip
pip install -r requirements.txt
Running scripts
The following are suggested commands based on the supplied filenames. Run each script from the directory containing it, and check whether it requires additional data files, arguments, or configuration.
python GFNN_Paper.py
python Iris.py
python WINE-RESULT.py
python Digits-dataset.py
python MNIST.py
python Random-Forest.py
python XGBoost.py
python Comparisonw-GFNN.py
python Gradient-Descent.py
The repository contains two Breast Cancer Wisconsin scripts; inspect their intended roles before deciding whether both should be run.
Reproducibility checklist
Before using this repository to reproduce the paper, record and publish:
- Python version and operating system.
- Exact package versions.
- Random seeds for all 30 runs.
- Train/test split proportions and split-generation procedure.
- Data preprocessing and any subset-selection procedure.
- GFNN initialization, learning rate, epochs per stage, divergence threshold, growth schedule, and stopping rule.
- Baseline model hyperparameters.
- Hardware and timing protocol.
- Commands used to generate each table and figure.
- Raw run-level results and the statistical-analysis procedure.
Do not claim exact reproducibility until these details and the corresponding outputs are available.
Citation
Please cite the associated manuscript if you use this code in academic work:

Nurmagambetov, D., Eset, A., and Eset, I.
A Synaptic Divergence-Guided Growing Feedforward Neural Network via Topological Adaptation.
Submitted manuscript.

Replace this provisional citation with the final journal citation and DOI after publication. Confirm the author order and spelling against the final manuscript before release.
License
The license is GNU General Public License v.3.0 
Contact
Corresponding author: Dias Nurmagambetov, Ph.D.
Department of Mathematics, Prairie View A&M University
Prairie View, Texas, USA
Email and ORCID: add or verify the details in the final manuscript.

| File | |
| --- | --- |
| [`CD-GFNN-Main.py`](./CD-GFNN-Main.py) | [![Run in Python Compiler](https://pythoncompiler.io/badge-run.svg)](https://pythoncompiler.io/?gh=drdiaz-max/python-practice/main/CD-GFNN-Main.py) |
| [`IRIS-WINE-DIGITS.py`](./IRIS-WINE-DIGITS.py) | [![Run in Python Compiler](https://pythoncompiler.io/badge-run.svg)](https://pythoncompiler.io/?gh=drdiaz-max/python-practice/main/IRIS-WINE-DIGITS.py) |
| [`MLP-8-MLP-16-MLP-32.py`](./MLP-8-MLP-16-MLP-32.py) | [![Run in Python Compiler](https://pythoncompiler.io/badge-run.svg)](https://pythoncompiler.io/?gh=drdiaz-max/python-practice/main/MLP-8-MLP-16-MLP-32.py) |
| [`dataset_VS_GFNN.py`](./dataset_VS_GFNN.py) | [![Run in Python Compiler](https://pythoncompiler.io/badge-run.svg)](https://pythoncompiler.io/?gh=drdiaz-max/python-practice/main/dataset_VS_GFNN.py) |

Made with [pythoncompiler.io](https://pythoncompiler.io/?utm_source=github&utm_medium=practice-repo) - the free online Python compiler.
<!-- /pythoncompiler:files -->

<!-- pythoncompiler:files -->
## Run these files in your browser

No setup needed. Each badge opens the file in a free online Python editor and runs it there.

| File | |
| --- | --- |
| [`CD-GFNN-Main.py`](./CD-GFNN-Main.py) | [![Run in Python Compiler](https://pythoncompiler.io/badge-run.svg)](https://pythoncompiler.io/?gh=drdiaz-max/python-practice/main/CD-GFNN-Main.py) |
| [`IRIS-WINE-DIGITS.py`](./IRIS-WINE-DIGITS.py) | [![Run in Python Compiler](https://pythoncompiler.io/badge-run.svg)](https://pythoncompiler.io/?gh=drdiaz-max/python-practice/main/IRIS-WINE-DIGITS.py) |
| [`MLP-8-MLP-16-MLP-32.py`](./MLP-8-MLP-16-MLP-32.py) | [![Run in Python Compiler](https://pythoncompiler.io/badge-run.svg)](https://pythoncompiler.io/?gh=drdiaz-max/python-practice/main/MLP-8-MLP-16-MLP-32.py) |
| [`dataset_VS_GFNN.py`](./dataset_VS_GFNN.py) | [![Run in Python Compiler](https://pythoncompiler.io/badge-run.svg)](https://pythoncompiler.io/?gh=drdiaz-max/python-practice/main/dataset_VS_GFNN.py) |
| [`ingle-Neuron-Linear-Fibonacci-Exponential-Growth.py`](./ingle-Neuron-Linear-Fibonacci-Exponential-Growth.py) | [![Run in Python Compiler](https://pythoncompiler.io/badge-run.svg)](https://pythoncompiler.io/?gh=drdiaz-max/python-practice/main/ingle-Neuron-Linear-Fibonacci-Exponential-Growth.py) |

Made with [pythoncompiler.io](https://pythoncompiler.io/?utm_source=github&utm_medium=practice-repo) - the free online Python compiler.
<!-- /pythoncompiler:files -->
