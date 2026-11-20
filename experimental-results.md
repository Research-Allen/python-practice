# SD-GFNN Consolidated Experimental Results

This document compiles all verified benchmark metrics, structural growth statistics, statistical significance tests, and execution traces across **30 independent runs** (`seeds 0..29`) for the **Synaptic Divergence-Guided Growing Feedforward Neural Network (SD-GFNN)**.

---

## 1. SD-GFNN Classification & Structural Adaptation Summary (30 Independent Runs)

| Dataset | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | Avg. Growth Events | Avg. Best Neurons | Avg. Terminal Nodes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** | $96.22 \pm 3.41$ | $96.68$ | $96.22$ | $96.18 \pm 3.49$ | $99.67 \pm 0.70$ | $1.13$ | $8.13$ | $9.13$ |
| **Breast Cancer** | $97.72 \pm 1.53$ | $97.77$ | $97.72$ | $97.71 \pm 1.54$ | $99.46 \pm 0.42$ | $1.27$ | $8.27$ | $9.27$ |
| **Wine** | $97.22 \pm 3.21$ | $97.45$ | $97.22$ | $97.20 \pm 3.27$ | $99.83 \pm 0.30$ | $1.13$ | $8.13$ | $9.13$ |
| **Digits** | $94.94 \pm 1.06$ | $95.09$ | $94.94$ | $94.93 \pm 1.06$ | $99.70 \pm 0.14$ | $2.20$ | $9.60$ | $11.30$ |

---

## 2. Breast Cancer Wisconsin: SD-GFNN vs. Fixed-Topology MLP Baselines (30 Runs)

### 2.1 Classification Performance

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SD-GFNN** | $97.72 \pm 1.53$ | $97.77$ | $97.72$ | $97.71$ | $99.46$ |
| **MLP(8)** | $97.75 \pm 1.48$ | $97.80$ | $97.75$ | $97.74$ | $99.46$ |
| **MLP(16)** | $97.40 \pm 1.54$ | $97.46$ | $97.40$ | $97.39$ | $99.48$ |
| **MLP(32)** | $97.63 \pm 1.28$ | $97.68$ | $97.63$ | $97.62$ | $99.47$ |

### 2.2 Paired Statistical Significance Tests ($N = 30$)

| Baseline | Mean Diff | Paired $t$-test $p$ | Wilcoxon $p$ | Cohen's $d$ | 95% CI Lower | 95% CI Upper |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP(8)** | $-0.000292$ | $0.813022$ | $0.916051$ | $-0.043579$ | $-0.002798$ | $+0.002213$ |
| **MLP(16)** | $+0.003216$ | $0.019044$ | $0.052001$ | $+0.453393$ | $+0.000567$ | $+0.005865$ |
| **MLP(32)** | $+0.000877$ | $0.572515$ | $0.565559$ | $+0.104218$ | $-0.002266$ | $+0.004020$ |

---

## 3. Tree-Based Ensemble Baselines (30 Independent Runs)

| Dataset | Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Iris** | Random Forest (300) | $95.78 \pm 3.84$ | $96.24$ | $95.78$ | $95.74 \pm 3.89$ | $99.63 \pm 0.50$ |
| **Iris** | XGBoost (300) | $93.44 \pm 3.60$ | $93.97$ | $93.44$ | $93.40 \pm 3.65$ | $98.43 \pm 1.66$ |
| **Wine** | Random Forest (300) | $98.52 \pm 1.56$ | $98.64$ | $98.52$ | $98.52 \pm 1.57$ | $99.94 \pm 0.14$ |
| **Wine** | XGBoost (300) | $98.24 \pm 1.68$ | $98.40$ | $98.24$ | $98.25 \pm 1.67$ | $99.90 \pm 0.20$ |
| **Digits** | Random Forest (300) | $97.68 \pm 0.61$ | $97.74$ | $97.68$ | $97.67 \pm 0.62$ | $99.95 \pm 0.02$ |
| **Digits** | XGBoost (300) | $97.22 \pm 0.77$ | $97.30$ | $97.22$ | $97.22 \pm 0.77$ | $99.95 \pm 0.03$ |

---

## 4. Single-Run Growth Trajectory Trace (Breast Cancer, `seed=42`)

```text
Round=01 | Neurons=  8 | ValAcc=0.9670 | Dmax=0.8148 | Gain=inf
Round=02 | Neurons=  9 | ValAcc=0.9780 | Dmax=0.8397 | Gain=0.010989
Round=03 | Neurons= 10 | ValAcc=0.9890 | Dmax=0.8481 | Gain=0.010989
Round=04 | Neurons= 12 | ValAcc=0.9670 | Dmax=0.8311 | Gain=-0.021978

STOP: Gain=-0.021978 (Best Checkpoint Retained: 10 Hidden Neurons, ValAcc=0.9890)
```