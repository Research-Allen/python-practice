# ==========================================================
# SIMPLIFIED SD-GFNN
# Fashion-MNIST Pilot Experiment
# 10 Runs
# ==========================================================

import gzip
import struct
import time
import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_rel,
    wilcoxon
)

from sklearn.model_selection import (
    train_test_split
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
# IDX LOADER
# ==========================================================

def load_idx_images(filename):

    with gzip.open(filename, "rb") as f:

        magic, num, rows, cols = struct.unpack(
            ">IIII",
            f.read(16)
        )

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        images = data.reshape(
            num,
            rows * cols
        )

        return (
            images.astype(np.float32)
            / 255.0
        )


def load_idx_labels(filename):

    with gzip.open(filename, "rb") as f:

        magic, num = struct.unpack(
            ">II",
            f.read(8)
        )

        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        return labels

# ==========================================================
# LOAD FASHION-MNIST
# ==========================================================

X_train_full = load_idx_images(
    "train-images-idx3-ubyte.gz"
)

y_train_full = load_idx_labels(
    "train-labels-idx1-ubyte.gz"
)

X_test = load_idx_images(
    "t10k-images-idx3-ubyte.gz"
)

y_test = load_idx_labels(
    "t10k-labels-idx1-ubyte.gz"
)

# ==========================================================
# SPEEDUP
# ==========================================================

X_train_full, _, y_train_full, _ = (
    train_test_split(
        X_train_full,
        y_train_full,
        train_size=5000,
        stratify=y_train_full,
        random_state=42
    )
)

print(
    "Training subset:",
    X_train_full.shape
)

# ==========================================================
# SYNAPTIC DIVERGENCE
# ==========================================================

def compute_divergence(
    model,
    X
):

    z = np.dot(
        X,
        model.coefs_[0]
    ) + model.intercepts_[0]

    h = np.tanh(z)

    mean_h = np.mean(
        h,
        axis=0
    )

    return np.max(
        np.mean(
            np.abs(
                h - mean_h
            ),
            axis=0
        )
    )

# ==========================================================
# SIMPLIFIED SD-GFNN
# ==========================================================

class FastSDGFNN:

    def __init__(
        self,
        random_state=0
    ):

        self.random_state = random_state

        self.best_model = None
        self.best_acc = -1

        self.best_neurons = 8

        self.growth_events = 0

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val
    ):

        neurons = 8

        for step in range(6):

            model = MLPClassifier(

                hidden_layer_sizes=(
                    neurons,
                ),

                activation="tanh",

                solver="adam",

                max_iter=20,

                early_stopping=True,

                n_iter_no_change=5,

                random_state=self.random_state
            )

            model.fit(
                X_train,
                y_train
            )

            val_pred = model.predict(
                X_val
            )

            acc = accuracy_score(
                y_val,
                val_pred
            )

            if acc > self.best_acc:

                self.best_acc = acc

                self.best_model = model

                self.best_neurons = (
                    neurons
                )

            div = compute_divergence(
                model,
                X_val
            )

            if div < 0.05:
                break

            neurons += 4

            self.growth_events += 1

    def predict(self, X):

        return self.best_model.predict(X)

# ==========================================================
# 10 RUNS
# ==========================================================

RUNS = 10

rows = []

for seed in range(RUNS):

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.20,
            stratify=y_train_full,
            random_state=seed
        )
    )

    # ----------------------------------
    # SD-GFNN
    # ----------------------------------

    start = time.perf_counter()

    sd = FastSDGFNN(
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

    sd_f1 = f1_score(
        y_test,
        sd_pred,
        average="macro"
    )

    # ----------------------------------
    # MLP32
    # ----------------------------------

    start = time.perf_counter()

    mlp = MLPClassifier(

        hidden_layer_sizes=(32,),

        activation="tanh",

        solver="adam",

        max_iter=20,

        early_stopping=True,

        n_iter_no_change=5,

        random_state=seed
    )

    mlp.fit(
        X_train,
        y_train
    )

    mlp_time = (
        time.perf_counter() - start
    )

    mlp_pred = mlp.predict(
        X_test
    )

    mlp_acc = accuracy_score(
        y_test,
        mlp_pred
    )

    mlp_f1 = f1_score(
        y_test,
        mlp_pred,
        average="macro"
    )

    rows.append([

        seed,

        sd_acc,
        sd_f1,
        sd.best_neurons,
        sd.growth_events,
        sd_time,

        mlp_acc,
        mlp_f1,
        mlp_time
    ])

    print(
        f"Run {seed+1:02d}: "
        f"SD={sd_acc:.4f} "
        f"MLP={mlp_acc:.4f}"
    )

# ==========================================================
# RESULTS
# ==========================================================

df = pd.DataFrame(

    rows,

    columns=[

        "Seed",

        "SD_Acc",
        "SD_F1",
        "SD_Neurons",
        "SD_Growth",
        "SD_Time",

        "MLP_Acc",
        "MLP_F1",
        "MLP_Time"
    ]
)

df.to_csv(
    "fashion_results.csv",
    index=False
)

# ==========================================================
# STATISTICS
# ==========================================================

t_stat, t_p = ttest_rel(
    df["SD_Acc"],
    df["MLP_Acc"]
)

w_stat, w_p = wilcoxon(
    df["SD_Acc"],
    df["MLP_Acc"]
)

print("\n====== RESULTS ======\n")

print(df.describe())

print("\nT-test p =", t_p)

print("Wilcoxon p =", w_p)