import warnings
warnings.filterwarnings("ignore")

import numpy as np

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_digits
)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    label_binarize
)

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================================================
# SETTINGS
# ==========================================================

N_RUNS = 30

# ==========================================================
# SYNAPTIC DIVERGENCE
# ==========================================================

def compute_synaptic_divergence(model, X):

    z = (
        np.dot(X, model.coefs_[0])
        + model.intercepts_[0]
    )

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
# DATASET EVALUATION
# ==========================================================

def evaluate_dataset(
    X,
    y,
    dataset_name
):

    classes = np.unique(y)

    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    auc_list = []

    growth_list = []
    best_neuron_list = []
    terminal_neuron_list = []

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

        model = SDGFNN(
            random_state=seed
        )

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val
        )

        pred = model.predict(
            X_test
        )

        prob = model.predict_proba(
            X_test
        )

        acc_list.append(
            accuracy_score(
                y_test,
                pred
            )
        )

        prec_list.append(
            precision_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )
        )

        rec_list.append(
            recall_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )
        )

        f1_list.append(
            f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )
        )

        y_bin = label_binarize(
            y_test,
            classes=classes
        )

        auc_list.append(
            roc_auc_score(
                y_bin,
                prob,
                average="macro",
                multi_class="ovr"
            )
        )

        growth_list.append(
            model.growth_events
        )

        best_neuron_list.append(
            model.best_neurons
        )

        terminal_neuron_list.append(
            model.terminal_neurons
        )

    print("\n")
    print("=" * 80)
    print(f"{dataset_name} + SD-GFNN")
    print("=" * 80)

    print(
        f"Accuracy Mean          : "
        f"{np.mean(acc_list):.4f}"
    )

    print(
        f"Accuracy Std           : "
        f"{np.std(acc_list):.4f}"
    )

    print(
        f"Precision Mean         : "
        f"{np.mean(prec_list):.4f}"
    )

    print(
        f"Recall Mean            : "
        f"{np.mean(rec_list):.4f}"
    )

    print(
        f"F1 Mean                : "
        f"{np.mean(f1_list):.4f}"
    )

    print(
        f"F1 Std                 : "
        f"{np.std(f1_list):.4f}"
    )

    print(
        f"ROC-AUC Mean           : "
        f"{np.mean(auc_list):.4f}"
    )

    print(
        f"ROC-AUC Std            : "
        f"{np.std(auc_list):.4f}"
    )

    print(
        f"Average Growth Events  : "
        f"{np.mean(growth_list):.2f}"
    )

    print(
        f"Average Best Neurons   : "
        f"{np.mean(best_neuron_list):.2f}"
    )

    print(
        f"Average Terminal Nodes : "
        f"{np.mean(terminal_neuron_list):.2f}"
    )

    print("=" * 80)

# ==========================================================
# LOAD DATASETS
# ==========================================================

iris = load_iris()

wine = load_wine()

digits = load_digits()

# ==========================================================
# RUN EXPERIMENTS
# ==========================================================

evaluate_dataset(
    iris.data,
    iris.target,
    "IRIS"
)

evaluate_dataset(
    wine.data,
    wine.target,
    "WINE"
)

evaluate_dataset(
    digits.data,
    digits.target,
    "DIGITS"
)