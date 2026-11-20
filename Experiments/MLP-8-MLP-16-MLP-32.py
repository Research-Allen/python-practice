import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import t

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================================================
# SYNAPTIC DIVERGENCE
# ==========================================================

def compute_synaptic_divergence(model, X):

    z = np.dot(X, model.coefs_[0]) + model.intercepts_[0]

    h = np.tanh(z)

    h_mean = np.mean(h, axis=0)

    return np.mean(
        np.abs(h - h_mean),
        axis=0
    )


# ==========================================================
# SD-GFNN
# ==========================================================

class SDGFNN:

    def __init__(
        self,
        tau=0.10,
        epsilon=0.01,
        min_improvement=0.0005,
        initial_neurons=8,
        max_growth_rounds=15,
        random_state=0
    ):

        self.tau = tau
        self.epsilon = epsilon
        self.min_improvement = min_improvement
        self.initial_neurons = initial_neurons
        self.max_growth_rounds = max_growth_rounds
        self.random_state = random_state

        self.best_model = None
        self.best_val_acc = -np.inf

        self.best_neurons = None
        self.terminal_neurons = None
        self.growth_events = 0

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val
    ):

        neurons = self.initial_neurons

        threshold = self.tau + self.epsilon

        prev_acc = None

        fib1 = 1
        fib2 = 1

        for r in range(self.max_growth_rounds):

            model = MLPClassifier(
                hidden_layer_sizes=(neurons,),
                activation="tanh",
                solver="adam",
                max_iter=1000,
                random_state=self.random_state
            )

            model.fit(
                X_train,
                y_train
            )

            val_pred = model.predict(X_val)

            val_acc = accuracy_score(
                y_val,
                val_pred
            )

            d_max = np.max(
                compute_synaptic_divergence(
                    model,
                    X_val
                )
            )

            acc_gain = (
                np.inf
                if prev_acc is None
                else val_acc - prev_acc
            )

            prev_acc = val_acc

            if (
                val_acc > self.best_val_acc
            ):

                self.best_val_acc = val_acc
                self.best_model = model
                self.best_neurons = neurons

            if d_max <= threshold:

                self.terminal_neurons = neurons
                break

            if (
                r > 0
                and
                acc_gain <= self.min_improvement
            ):

                self.terminal_neurons = neurons
                break

            old_neurons = neurons

            if r < 2:

                neurons += 1

            else:

                growth = fib1 + fib2

                fib1, fib2 = (
                    fib2,
                    growth
                )

                neurons += growth

            if neurons > old_neurons:
                self.growth_events += 1

        if self.terminal_neurons is None:
            self.terminal_neurons = neurons

    def predict(self, X):

        return self.best_model.predict(X)

    def predict_proba(self, X):

        return self.best_model.predict_proba(X)


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    pred = model.predict(X_test)

    prob = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "Accuracy":
            accuracy_score(
                y_test,
                pred
            ),
        "Precision":
            precision_score(
                y_test,
                pred,
                average="weighted"
            ),
        "Recall":
            recall_score(
                y_test,
                pred,
                average="weighted"
            ),
        "F1":
            f1_score(
                y_test,
                pred,
                average="weighted"
            ),
        "ROC_AUC":
            roc_auc_score(
                y_test,
                prob
            )
    }


# ==========================================================
# STATISTICS
# ==========================================================

def cohen_d_paired(a, b):

    diff = np.array(a) - np.array(b)

    return (
        np.mean(diff)
        /
        np.std(diff, ddof=1)
    )


def confidence_interval(
    values,
    alpha=0.05
):

    values = np.array(values)

    n = len(values)

    mean = np.mean(values)

    se = (
        np.std(values, ddof=1)
        /
        np.sqrt(n)
    )

    tcrit = t.ppf(
        1 - alpha / 2,
        n - 1
    )

    return (
        mean - tcrit * se,
        mean + tcrit * se
    )


def compare_models(
    sd_scores,
    baseline_scores,
    baseline_name
):

    t_stat, p_t = ttest_rel(
        sd_scores,
        baseline_scores
    )

    try:

        w_stat, p_w = wilcoxon(
            sd_scores,
            baseline_scores
        )

    except:

        w_stat = np.nan
        p_w = np.nan

    effect = cohen_d_paired(
        sd_scores,
        baseline_scores
    )

    diff = (
        np.array(sd_scores)
        -
        np.array(baseline_scores)
    )

    ci_low, ci_high = confidence_interval(
        diff
    )

    return [

        baseline_name,

        np.mean(diff),

        p_t,

        p_w,

        effect,

        ci_low,

        ci_high

    ]


# ==========================================================
# DATA
# ==========================================================

X, y = load_breast_cancer(
    return_X_y=True
)

N_RUNS = 30

results = {

    "SD-GFNN": [],
    "MLP(8)": [],
    "MLP(16)": [],
    "MLP(32)": []

}

growth_events = []
best_neurons = []
terminal_neurons = []

# ==========================================================
# EXPERIMENTS
# ==========================================================

for seed in range(N_RUNS):

    X_train_full, X_test, y_train_full, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=seed
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.20,
            stratify=y_train_full,
            random_state=seed
        )
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    # SD-GFNN

    sdgfnn = SDGFNN(
        random_state=seed
    )

    sdgfnn.fit(
        X_train,
        y_train,
        X_val,
        y_val
    )

    results["SD-GFNN"].append(
        evaluate_model(
            sdgfnn,
            X_test,
            y_test
        )
    )

    growth_events.append(
        sdgfnn.growth_events
    )

    best_neurons.append(
        sdgfnn.best_neurons
    )

    terminal_neurons.append(
        sdgfnn.terminal_neurons
    )

    # MLP BASELINES

    for h in [8, 16, 32]:

        mlp = MLPClassifier(
            hidden_layer_sizes=(h,),
            activation="tanh",
            solver="adam",
            max_iter=1000,
            random_state=seed
        )

        mlp.fit(
            X_train,
            y_train
        )

        name = f"MLP({h})"

        results[name].append(
            evaluate_model(
                mlp,
                X_test,
                y_test
            )
        )

# ==========================================================
# PERFORMANCE TABLE
# ==========================================================

perf_rows = []

for model_name in results:

    acc = [
        x["Accuracy"]
        for x in results[model_name]
    ]

    prec = [
        x["Precision"]
        for x in results[model_name]
    ]

    rec = [
        x["Recall"]
        for x in results[model_name]
    ]

    f1 = [
        x["F1"]
        for x in results[model_name]
    ]

    auc = [
        x["ROC_AUC"]
        for x in results[model_name]
    ]

    perf_rows.append([

        model_name,

        f"{np.mean(acc):.4f} ± {np.std(acc):.4f}",

        f"{np.mean(prec):.4f}",

        f"{np.mean(rec):.4f}",

        f"{np.mean(f1):.4f}",

        f"{np.mean(auc):.4f}"

    ])

perf_df = pd.DataFrame(

    perf_rows,

    columns=[

        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
)

print("\n")
print("=" * 100)
print("PERFORMANCE RESULTS")
print("=" * 100)
print(perf_df.to_string(index=False))

# ==========================================================
# SIGNIFICANCE TESTS
# ==========================================================

sd_acc = [
    r["Accuracy"]
    for r in results["SD-GFNN"]
]

stats_rows = []

for baseline in [

    "MLP(8)",
    "MLP(16)",
    "MLP(32)"

]:

    baseline_acc = [

        r["Accuracy"]
        for r in results[baseline]

    ]

    stats_rows.append(

        compare_models(
            sd_acc,
            baseline_acc,
            baseline
        )

    )

stats_df = pd.DataFrame(

    stats_rows,

    columns=[

        "Baseline",
        "Mean Diff",
        "Paired t-test p",
        "Wilcoxon p",
        "Cohen d",
        "95% CI Lower",
        "95% CI Upper"
    ]
)

print("\n")
print("=" * 100)
print("STATISTICAL COMPARISON: SD-GFNN vs BASELINES")
print("=" * 100)
print(stats_df.to_string(index=False))

# ==========================================================
# SD-GFNN STRUCTURAL RESULTS
# ==========================================================

print("\n")
print("=" * 100)
print("SD-GFNN STRUCTURE")
print("=" * 100)

print(
    f"Average Growth Events    : "
    f"{np.mean(growth_events):.2f}"
)

print(
    f"Average Best Neurons     : "
    f"{np.mean(best_neurons):.2f}"
)

print(
    f"Average Terminal Neurons : "
    f"{np.mean(terminal_neurons):.2f}"
)

print("=" * 100)