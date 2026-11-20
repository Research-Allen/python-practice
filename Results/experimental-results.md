# SD-GFNN Consolidated Experimental Results

## 1. Classification & Structural Adaptation Summary

|Dataset|Accuracy|Precision|Recall|F1|ROC-AUC|Growth Events|Best Neurons|Terminal Nodes|
|---|---|---|---|---|---|---|---|---|
|Iris|96.22±3.41|96.68|96.22|96.18±3.49|99.67±0.70|1.13|8.13|9.13|
|Breast Cancer|97.72±1.53|97.77|97.72|97.71±nan|99.46±nan|1.27|8.27|9.27|
|Wine|97.22±3.21|97.45|97.22|97.20±3.27|99.83±0.30|1.13|8.13|9.13|
|Digits|94.94±1.06|95.09|94.94|94.93±1.06|99.70±0.14|2.20|9.60|11.30|

## 2. Tree-Based Ensemble Baselines

|Dataset|Model|Accuracy|F1|ROC-AUC|
|---|---|---|---|---|
|Iris|Random Forest (300)|95.78±3.84|95.74±3.89|99.63±0.50|
|Iris|XGBoost (300)|93.44±3.60|93.40±3.65|98.43±1.66|
|Wine|Random Forest (300)|98.52±1.56|98.52±1.57|99.94±0.14|
|Wine|XGBoost (300)|98.24±1.68|98.25±1.67|99.90±0.20|
|Digits|Random Forest (300)|97.68±0.61|97.67±0.62|99.95±0.02|
|Digits|XGBoost (300)|97.22±0.77|97.22±0.77|99.95±0.03|
|Breast Cancer|MLP(8)|97.75±1.48|97.74±nan|99.46±nan|
|Breast Cancer|MLP(16)|97.40±1.54|97.39±nan|99.48±nan|
|Breast Cancer|MLP(32)|97.63±1.28|97.62±nan|99.47±nan|