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

    z = np.dot(
        X,
        model.coefs_[0]
    ) + model.intercepts_[0]

    h = np.tanh(z)

    h_mean = np.mean(
        h,
        axis=0
    )

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

        self.min_improvement = (
            min_improvement
        )

        self.initial_neurons = (
            initial_neurons
        )

        self.max_growth_rounds = (
            max_growth_rounds
        )

        self.random_state = (
            random_state
        )

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

            divergence = (
                compute_synaptic_divergence(
                    model,
                    X_val
                )
            )

            d_max = np.max(
                divergence
            )

            acc_gain = (
                np.inf
                if prev_acc is None
                else val_acc - prev_acc
            )

            prev_acc = val_acc

            if (
                val_acc >
                self.best_val_acc
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
                growth = 1
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

            pred = model.predict(
                X_val
            )

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

    def predict(self, X):

        return self.best_model.predict(X)

# ==========================================================
# DATA
# ==========================================================

X, y = load_breast_cancer(
    return_X_y=True
)

# ==========================================================
# EXPERIMENT
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

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    # -----------------------------------
    # CASCADE CORRELATION
    # -----------------------------------

    start = time.perf_counter()

    cc = CascadeCorrelation(
        random_state=seed
    )

    cc.fit(
        X_train,
        y_train,
        X_val,
        y_val
    )

    cc_time = (
        time.perf_counter() - start
    )

    cc_pred = cc.predict(
        X_test
    )

    cc_acc = accuracy_score(
        y_test,
        cc_pred
    )

    cc_prec = precision_score(
        y_test,
        cc_pred
    )

    cc_rec = recall_score(
        y_test,
        cc_pred
    )

    cc_f1 = f1_score(
        y_test,
        cc_pred
    )

    # -----------------------------------
    # SD-GFNN
    # -----------------------------------

    start = time.perf_counter()

    sd = SDGFNN(
        random_state=seed
    )

    sd.fit(
        X_train,
        y_train,
        X_val,
        y_val
    )

    sd_time = (
        time.perf_counter() - start
    )

    sd_pred = sd.predict(
        X_test
    )

    sd_acc = accuracy_score(
        y_test,
        sd_pred
    )

    sd_prec = precision_score(
        y_test,
        sd_pred
    )

    sd_rec = recall_score(
        y_test,
        sd_pred
    )

    sd_f1 = f1_score(
        y_test,
        sd_pred
    )

    results.append([

        seed,

        cc_acc,
        cc_prec,
        cc_rec,
        cc_f1,
        cc.best_neurons,
        cc_time,

        sd_acc,
        sd_prec,
        sd_rec,
        sd_f1,
        sd.best_neurons,
        sd.terminal_neurons,
        sd.growth_events,
        sd_time
    ])

    print(
        f"Run {seed+1:02d} | "
        f"CC={cc_acc:.4f} | "
        f"SD-GFNN={sd_acc:.4f}"
    )

# ==========================================================
# DATAFRAME
# ==========================================================

df = pd.DataFrame(

    results,

    columns=[

        "Seed",

        "CC_Accuracy",
        "CC_Precision",
        "CC_Recall",
        "CC_F1",
        "CC_BestNeurons",
        "CC_Time",

        "SD_Accuracy",
        "SD_Precision",
        "SD_Recall",
        "SD_F1",
        "SD_BestNeurons",
        "SD_FinalNeurons",
        "SD_GrowthEvents",
        "SD_Time"
    ]
)

df.to_csv(
    "all_30_runs.csv",
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

def summary(x):

    mean = np.mean(x)

    std = np.std(
        x,
        ddof=1
    )

    ci = (
        1.96 *
        std /
        np.sqrt(len(x))
    )

    return mean, std, ci

cc_mean, cc_std, cc_ci = summary(
    df["CC_Accuracy"]
)

sd_mean, sd_std, sd_ci = summary(
    df["SD_Accuracy"]
)

# ==========================================================
# STATS
# ==========================================================

t_stat, t_p = ttest_rel(
    df["SD_Accuracy"],
    df["CC_Accuracy"]
)

w_stat, w_p = wilcoxon(
    df["SD_Accuracy"],
    df["CC_Accuracy"]
)

diff = (
    df["SD_Accuracy"] -
    df["CC_Accuracy"]
)

cohen_d = (
    np.mean(diff)
    /
    np.std(diff, ddof=1)
)

print("\n=== SUMMARY ===\n")

print(
    f"CC Mean={cc_mean:.4f} "
    f"Std={cc_std:.4f} "
    f"CI=[{cc_mean-cc_ci:.4f},"
    f"{cc_mean+cc_ci:.4f}]"
)

print(
    f"SD Mean={sd_mean:.4f} "
    f"Std={sd_std:.4f} "
    f"CI=[{sd_mean-sd_ci:.4f},"
    f"{sd_mean+sd_ci:.4f}]"
)

print("\n=== STATISTICS ===\n")

print(
    f"Paired t-test: "
    f"t={t_stat:.4f} "
    f"p={t_p:.6f}"
)

print(
    f"Wilcoxon: "
    f"W={w_stat:.4f} "
    f"p={w_p:.6f}"
)

print(
    f"Cohen d: "
    f"{cohen_d:.4f}"
)