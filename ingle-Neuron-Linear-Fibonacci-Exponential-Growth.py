import time
import numpy as np

from scipy.stats import ttest_rel, wilcoxon
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# =====================================================
# Growth Strategies
# =====================================================

def single(stage):
    return 1

def linear(stage):
    return stage

def fibonacci(stage):
    if stage <= 2:
        return 1
    a, b = 1, 1
    for _ in range(stage - 2):
        a, b = b, a + b
    return b

def exponential(stage):
    return 2 ** (stage - 1)


# =====================================================
# Train Strategy
# =====================================================

def run_strategy(strategy_func, seed):

    digits = load_digits()

    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.2,
        random_state=seed,
        stratify=digits.target
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    hidden = 0
    best_acc = 0

    for stage in range(1, 9):

        hidden += strategy_func(stage)

        clf = MLPClassifier(
            hidden_layer_sizes=(hidden,),
            max_iter=300,
            random_state=seed
        )

        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)

        best_acc = accuracy_score(y_test, pred)

    return best_acc


# =====================================================
# Multiple Runs
# =====================================================

RUNS = 30

single_acc = []
linear_acc = []
fib_acc = []
exp_acc = []

for seed in range(RUNS):

    single_acc.append(run_strategy(single, seed))
    linear_acc.append(run_strategy(linear, seed))
    fib_acc.append(run_strategy(fibonacci, seed))
    exp_acc.append(run_strategy(exponential, seed))

single_acc = np.array(single_acc)
linear_acc = np.array(linear_acc)
fib_acc = np.array(fib_acc)
exp_acc = np.array(exp_acc)


# =====================================================
# Summary
# =====================================================

def summarize(name, x):

    mean = np.mean(x)
    std = np.std(x)

    ci95 = 1.96 * std / np.sqrt(len(x))

    print(
        f"{name:<12}"
        f"Mean={mean:.4f} "
        f"Std={std:.4f} "
        f"CI95=[{mean-ci95:.4f},{mean+ci95:.4f}]"
    )

print("\n=== RESULTS OVER 30 RUNS ===\n")

summarize("Single", single_acc)
summarize("Linear", linear_acc)
summarize("Fibonacci", fib_acc)
summarize("Exponential", exp_acc)


# =====================================================
# Paired t-tests
# =====================================================

print("\n=== PAIRED T-TESTS ===\n")

comparisons = [
    ("Fib vs Single", fib_acc, single_acc),
    ("Fib vs Linear", fib_acc, linear_acc),
    ("Fib vs Exp", fib_acc, exp_acc)
]

for name, a, b in comparisons:

    t, p = ttest_rel(a, b)

    print(
        f"{name:<18}"
        f"t={t:.4f} "
        f"p={p:.6f}"
    )


# =====================================================
# Wilcoxon
# =====================================================

print("\n=== WILCOXON ===\n")

for name, a, b in comparisons:

    stat, p = wilcoxon(a, b)

    print(
        f"{name:<18}"
        f"W={stat:.4f} "
        f"p={p:.6f}"
    )


# =====================================================
# Cohen d
# =====================================================

print("\n=== EFFECT SIZE ===\n")

for name, a, b in comparisons:

    diff = a - b

    d = np.mean(diff) / np.std(diff)

    print(
        f"{name:<18}"
        f"d={d:.4f}"
    )