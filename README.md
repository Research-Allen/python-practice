# SD-GFNN: Synaptic Divergence-Guided Growing Feedforward Neural Network

https://zenodo.org/badge/DOI/10.5281/zenodo.24407104.svg](https://doi.org/10.5281/zenodo.24407104)

## Overview

SD-GFNN (Synaptic Divergence-Guided Growing Feedforward Neural Network) is a constructive learning framework that automatically adapts hidden-layer topology during training through divergence-guided structural growth.

Unlike conventional feedforward neural networks that require manual selection of hidden-layer width, SD-GFNN incrementally expands network capacity according to Synaptic Divergence, a neuron-level measure of representational heterogeneity and downstream synaptic flux variability.

The framework combines:

- Synaptic Divergence-guided growth
- Topological Spot localization
- Function-preserving neuron insertion
- Fibonacci-regulated capacity expansion
- Conditional finite structural termination
- Adaptive architecture construction without manual topology selection

---

## Repository Structure

```text
SD-GFNN
│
├── Docs/
│   ├── CITATION.bib
│   ├── CITATION.cff
│   └── LICENSE
│
├── Experiments/
│   ├── CD-GFNN-Main.py
│   ├── dataset_VS_GFNN.py
│   ├── IRIS-WINE-DIGITS.py
│   ├── MLP-8-MLP-16-MLP-32.py
│   ├── SD-GFNN-vs-CC-vs-DNC.py
│   ├── Fashion-MNIST.py
│   └── Single-Neuron-Linear-Fibonacci-Exponential-Growth.py
│
├── Manuscript/
│   └── CD-GFNNPaper.pdf
│
├── Results/
│   ├── experimental-results.md
│   ├── statistical-tests.csv
│   ├── results.csv
│   ├── SD-GFNN vs DNC vs CC.txt
│   └── ForFashion-MNIST.txt
│
├── README.md
└── requirements.txt
```

---

## Main Contributions

- Automatic hidden-layer growth without manual tuning.
- Synaptic Divergence criterion for adaptive capacity allocation.
- Function-preserving neuron insertion to maintain output continuity.
- Fibonacci-based growth strategy that limits capacity overshoot.
- Theoretical analysis of invariance, convergence, and bounded growth.
- Statistical validation through repeated experimental trials.

---

## Experimental Evaluation

### Benchmark Datasets

- Iris
- Wine
- Breast Cancer Wisconsin
- Digits
- Fashion-MNIST

### Classical Constructive Neural Network Baselines

- Cascade-Correlation (Fahlman & Lebiere, 1990)
- Dynamic Node Creation (Ash, 1989)

### Fixed-Topology Baselines

- MLP(8)
- MLP(16)
- MLP(32)

---

## Representative Results

### Constructive Neural Network Comparison

| Method | Mean Accuracy |
|----------|----------:|
| Cascade-Correlation | 97.49% |
| Dynamic Node Creation | 97.60% |
| SD-GFNN | 97.72% |

### Fashion-MNIST Evaluation

| Method | Mean Accuracy |
|----------|----------:|
| MLP(32) | 90.40% |
| SD-GFNN | 90.13% |

The results indicate that divergence-guided growth achieves predictive performance comparable to classical constructive neural networks while maintaining favorable computational efficiency.

---

## Installation

```bash
git clone https://github.com/Research-Allen/SD-GFNN.git

cd SD-GFNN

pip install -r requirements.txt
```

---

## Running Experiments

### Main SD-GFNN Experiment

```bash
python Experiments/CD-GFNN-Main.py
```

### Constructive Neural Network Benchmark

```bash
python Experiments/SD-GFNN-vs-CC-vs-DNC.py
```

### Fashion-MNIST Evaluation

```bash
python Experiments/Fashion-MNIST.py
```

---

## Manuscript

The accompanying research manuscript is available in:

```text
Manuscript/CD-GFNNPaper.pdf
```

---

## Citation

If you use SD-GFNN in your research, please cite:

```bibtex
@software{nurmagambetov2026sdgfnn,
  author    = {Dias Nurmagambetov, Ph.D., Allen Eset, Isaiah Eset},
  title     = {SD-GFNN: Synaptic Divergence-Guided Growing Feedforward Neural Network},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24407104},
  url       = {https://doi.org/10.5281/zenodo.24407104}
}
```

Concept DOI:

https://doi.org/10.5281/zenodo.24407104

Version DOI:

https://doi.org/10.5281/zenodo.24407105

---

## License

MIT License
