# ==========================================================
# SD-GFNN vs CC vs DNC
# 30-RUN CONSTRUCTIVE NEURAL NETWORK BENCHMARK
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_rel,
    wilcoxon
)

from sklearn.datasets import (
    load_breast_cancer
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.neural_network import (
    MLPClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
        tau=0.05,
        epsilon=0.005,
        min_improvement=0.0005,
        initial_neurons=8,
        max_growth_rounds=25,
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

        threshold = (
            self.tau +
            self.epsilon
        )

        prev_acc = None

        fib1 = 1
        fib2 = 1

        for r in range(
            self.max_growth_rounds
        ):

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

            val_pred = model.predict(
                X_val
            )

            val_acc = accuracy_score(
                y_val,
                val_pred
            )

            divergence = compute_synaptic_divergence(
                model,
                X_val
            )

            d_max = np.max(divergence)

            acc_gain = (
                np.inf
                if prev_acc is None
                else val_acc - prev_acc
            )

            prev_acc = val_acc

            if val_acc > self.best_val_acc:

                self.best_val_acc = val_acc
                self.best_model = model
                self.best_neurons = neurons

            if d_max <= threshold:

                self.terminal_neurons = neurons
                break

            if (
                r > 0 and
                acc_gain <= self.min_improvement
            ):

                self.terminal_neurons = neurons
                break

            old_neurons = neurons

            if r < 2:

                growth = 1

            else:

                growth = fib1 + fib2
                fib1, fib2 = fib2, growth

            neurons += growth

            if neurons > old_neurons:

                self.growth_events += 1

        if self.terminal_neurons is None:

            self.terminal_neurons = neurons

        return self

    def predict(self, X):

        return self.best_model.predict(X)

# ==========================================================
# CASCADE CORRELATION
# ==========================================================

class CascadeCorrelation:

    def __init__(
        self,
        max_neurons=50,
        patience=5,
        random_state=0
    ):

        self.max_neurons = max_neurons
        self.patience = patience
        self.random_state = random_state

        self.best_score = -np.inf

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val
    ):

        no_improve = 0

        for neurons in range(
            1,
            self.max_neurons + 1
        ):

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

            pred = model.predict(X_val)

            acc = accuracy_score(
                y_val,
                pred
            )

            if acc > self.best_score:

                self.best_score = acc
                self.best_model = model
                self.best_neurons = neurons

                no_improve = 0

            else:

                no_improve += 1

            if no_improve >= self.patience:
                break

        return self

    def predict(self, X):

        return self.best_model.predict(X)

# ==========================================================
# DYNAMIC NODE CREATION
# ==========================================================

class DynamicNodeCreation:

    def __init__(
        self,
        start_neurons=1,
        max_neurons=50,
        error_threshold=0.01,
        random_state=0
    ):

        self.start_neurons = start_neurons
        self.max_neurons = max_neurons
        self.error_threshold = error_threshold
        self.random_state = random_state

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val
    ):

        neurons = self.start_neurons

        prev_error = np.inf

        self.best_score = -np.inf

        while neurons <= self.max_neurons:

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

            pred = model.predict(X_val)

            acc = accuracy_score(
                y_val,
                pred
            )

            error = 1.0 - acc

            if acc > self.best_score:

                self.best_score = acc
                self.best_model = model
                self.best_neurons = neurons

            if abs(
                prev_error - error
            ) < self.error_threshold:

                neurons += 1

            prev_error = error

            neurons += 1

        return self

    def predict(self, X):

        return self.best_model.predict(X)

# ==========================================================
# DATA
# ==========================================================

X, y = load_breast_cancer(
    return_X_y=True
)

# ==========================================================
# RUNS
# ==========================================================

RUNS = 30

results = []

for seed in range(RUNS):

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

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    methods = {
        "CC": CascadeCorrelation(random_state=seed),
        "DNC": DynamicNodeCreation(random_state=seed),
        "SD-GFNN": SDGFNN(random_state=seed)
    }

    row = {"Seed": seed}

    for name, model in methods.items():

        start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val
        )

        train_time = (
            time.perf_counter() - start
        )

        pred = model.predict(X_test)

        row[f"{name}_Accuracy"] = accuracy_score(
            y_test,
            pred
        )

        row[f"{name}_Precision"] = precision_score(
            y_test,
            pred
        )

        row[f"{name}_Recall"] = recall_score(
            y_test,
            pred
        )

        row[f"{name}_F1"] = f1_score(
            y_test,
            pred
        )

        row[f"{name}_Neurons"] = getattr(
            model,
            "best_neurons",
            np.nan
        )

        row[f"{name}_Time"] = train_time

    results.append(row)

    print(
        f"Run {seed+1:02d} | "
        f"CC={row['CC_Accuracy']:.4f} | "
        f"DNC={row['DNC_Accuracy']:.4f} | "
        f"SD={row['SD-GFNN_Accuracy']:.4f}"
    )

# ==========================================================
# SAVE ALL RUNS
# ==========================================================

df = pd.DataFrame(results)

df.to_csv(
    "all_30_runs_constructive.csv",
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

def summarize(x):

    mean = np.mean(x)

    std = np.std(
        x,
        ddof=1
    )

    ci = (
        1.96 *
        std /
        np.sqrt(
            len(x)
        )
    )

    return [
        mean,
        std,
        mean - ci,
        mean + ci
    ]

summary = []

for method in [
    "CC",
    "DNC",
    "SD-GFNN"
]:

    acc = summarize(
        df[f"{method}_Accuracy"]
    )

    summary.append([

        method,

        acc[0],
        acc[1],
        acc[2],
        acc[3],

        np.mean(
            df[f"{method}_F1"]
        ),

        np.mean(
            df[f"{method}_Neurons"]
        ),

        np.mean(
            df[f"{method}_Time"]
        )
    ])

summary_df = pd.DataFrame(
    summary,
    columns=[
        "Method",
        "Accuracy_Mean",
        "Accuracy_STD",
        "CI95_Low",
        "CI95_High",
        "F1_Mean",
        "Avg_Neurons",
        "Avg_Time"
    ]
)

summary_df.to_csv(
    "constructive_summary.csv",
    index=False
)


print("\n=== SUMMARY ===\n")
print(summary_df)

# ======================================================
# STATISTICS
# ==========================================================

for baseline in [
    "CC",
    "DNC"
]:

    t_stat, t_p = ttest_rel(
        df["SD-GFNN_Accuracy"],
        df[f"{baseline}_Accuracy"]
    )

    w_stat, w_p = wilcoxon(
        df["SD-GFNN_Accuracy"],
        df[f"{baseline}_Accuracy"]
    )

    print(
        f"\nSD-GFNN vs {baseline}"
    )

    print(
        f"t-test p = {t_p:.6f}"
    )

    print(
        f"Wilcoxon p = {w_p:.6f}"
    )
