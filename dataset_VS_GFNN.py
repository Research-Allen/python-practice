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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# ==========================================================
# SETTINGS
# ==========================================================

N_RUNS = 30

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_dataset(
    X,
    y,
    model,
    title
):

    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    auc_list = []

    classes = np.unique(y)

    for seed in range(N_RUNS):

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                stratify=y,
                random_state=seed
            )
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

        model.fit(
            X_train,
            y_train
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
                average="weighted"
            )
        )

        rec_list.append(
            recall_score(
                y_test,
                pred,
                average="weighted"
            )
        )

        f1_list.append(
            f1_score(
                y_test,
                pred,
                average="weighted"
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

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Accuracy Mean : "
        f"{np.mean(acc_list):.4f}"
    )

    print(
        f"Accuracy Std  : "
        f"{np.std(acc_list):.4f}"
    )

    print(
        f"Precision Mean: "
        f"{np.mean(prec_list):.4f}"
    )

    print(
        f"Recall Mean   : "
        f"{np.mean(rec_list):.4f}"
    )

    print(
        f"F1 Mean       : "
        f"{np.mean(f1_list):.4f}"
    )

    print(
        f"F1 Std        : "
        f"{np.std(f1_list):.4f}"
    )

    print(
        f"ROC-AUC Mean  : "
        f"{np.mean(auc_list):.4f}"
    )

    print(
        f"ROC-AUC Std   : "
        f"{np.std(auc_list):.4f}"
    )

    print("=" * 70)

# ==========================================================
# LOAD DATASETS
# ==========================================================

iris = load_iris()

wine = load_wine()

digits = load_digits()

# ==========================================================
# RANDOM FOREST
# ==========================================================

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=0,
    n_jobs=-1
)

evaluate_dataset(
    iris.data,
    iris.target,
    rf,
    "IRIS + RANDOM FOREST"
)

evaluate_dataset(
    wine.data,
    wine.target,
    rf,
    "WINE + RANDOM FOREST"
)

evaluate_dataset(
    digits.data,
    digits.target,
    rf,
    "DIGITS + RANDOM FOREST"
)

# ==========================================================
# XGBOOST
# ==========================================================

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=0
)

evaluate_dataset(
    iris.data,
    iris.target,
    xgb,
    "IRIS + XGBOOST"
)

evaluate_dataset(
    wine.data,
    wine.target,
    xgb,
    "WINE + XGBOOST"
)

evaluate_dataset(
    digits.data,
    digits.target,
    xgb,
    "DIGITS + XGBOOST"
)
