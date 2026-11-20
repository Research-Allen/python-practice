import warnings
warnings.filterwarnings("ignore")

import numpy as np

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

        print("\nGrowth Process")
        print("-" * 70)

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

            print(
                f"Round={r+1:02d} | "
                f"Neurons={neurons:3d} | "
                f"ValAcc={val_acc:.4f} | "
                f"Dmax={d_max:.4f} | "
                f"Gain={acc_gain:.6f}"
            )

            # ----------------------------------
            # STOPPING RULES
            # ----------------------------------

            if d_max <= threshold:

                self.terminal_neurons = neurons

                print(
                    f"\nSTOP: Dmax={d_max:.4f} "
                    f"<= Threshold={threshold:.4f}"
                )

                break

            if (
                r > 0
                and
                acc_gain <= self.min_improvement
            ):

                self.terminal_neurons = neurons

                print(
                    f"\nSTOP: Gain={acc_gain:.6f}"
                )

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
# DATA
# ==========================================================

X, y = load_breast_cancer(
    return_X_y=True
)

# ==========================================================
# SINGLE EXPERIMENT
# ==========================================================

seed = 42

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

# ==========================================================
# TRAIN SD-GFNN
# ==========================================================

model = SDGFNN(
    random_state=seed
)

model.fit(
    X_train,
    y_train,
    X_val,
    y_val
)

# ==========================================================
# TEST
# ==========================================================

pred = model.predict(
    X_test
)

prob = model.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred
)

recall = recall_score(
    y_test,
    pred
)

f1 = f1_score(
    y_test,
    pred
)

auc = roc_auc_score(
    y_test,
    prob
)

# ==========================================================
# RESULTS
# ==========================================================

print("\n")
print("=" * 70)
print("SD-GFNN RESULTS")
print("=" * 70)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"ROC-AUC           : {auc:.4f}")

print("-" * 70)

print(
    f"Growth Events     : "
    f"{model.growth_events}"
)

print(
    f"Best Neurons      : "
    f"{model.best_neurons}"
)

print(
    f"Terminal Neurons  : "
    f"{model.terminal_neurons}"
)

print(
    f"Best Val Accuracy : "
    f"{model.best_val_acc:.4f}"
)

print("=" * 70)