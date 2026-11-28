import time
import numpy as np
from scipy.stats import wilcoxon
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from AGNN import OptimalCrossTowerAGNN, make_two_spirals, relu


class SearchAblatedAGNN(OptimalCrossTowerAGNN):
    """Сравнение стратегий выбора связей: Hybrid Golden vs OMP vs Exhaustive."""

    def __init__(self, search_strategy="hybrid_golden", **kwargs):
        super().__init__(**kwargs)
        assert search_strategy in ("hybrid_golden", "omp", "exhaustive")
        self.search_strategy = search_strategy
        self.search_time_s = 0.0

    def select_connections_golden(self, candidate_cols, local_idx, local_gaps, y_idx):
        t0 = time.perf_counter()
        if self.search_strategy == "hybrid_golden":
            res = super().select_connections_golden(candidate_cols, local_idx, local_gaps, y_idx)
            self.search_time_s += time.perf_counter() - t0
            return res

        M = len(candidate_cols)
        if M <= 2:
            self.search_time_s += time.perf_counter() - t0
            return candidate_cols

        y_loc = y_idx[local_idx]
        dom_cls = int(np.bincount(y_loc).argmax())
        target_bin = (y_loc == dom_cls).astype(float)
        base_err = float(np.mean((0.5 - target_bin) ** 2))

        if self.search_strategy == "exhaustive":
            g_cent = local_gaps - np.mean(local_gaps)
            g_norm = np.linalg.norm(g_cent) + 1e-8
            scored_cols = []
            for col in candidate_cols:
                act = self.coords[local_idx, col]
                a_cent = act - np.mean(act)
                a_norm = np.linalg.norm(a_cent) + 1e-8
                corr = abs(float(np.dot(a_cent, g_cent) / (a_norm * g_norm)))
                scored_cols.append((corr * (0.25 + float(np.std(act))), col))
            scored_cols.sort(key=lambda x: x[0], reverse=True)
            ordered_cols = [col for _, col in scored_cols]

            def eval_cost(k):
                sub_cols = ordered_cols[:k]
                sub_in = self.coords[local_idx][:, sub_cols]
                c_in = (
                    np.mean(sub_in[target_bin == 1], axis=0)
                    if np.any(target_bin == 1)
                    else np.mean(sub_in, axis=0)
                )
                w = c_in.reshape(-1, 1)
                b = -0.5 * float(np.dot(c_in, c_in))
                act_proto = relu(sub_in @ w + b).flatten()
                if np.std(act_proto) < 1e-8:
                    loss_k = base_err
                else:
                    p_norm = (act_proto - np.min(act_proto)) / (
                        np.max(act_proto) - np.min(act_proto) + 1e-8
                    )
                    loss_k = float(np.mean((p_norm - target_bin) ** 2))
                return loss_k + self.sparsity_penalty * (k / M)

            losses = [eval_cost(k) for k in range(2, M + 1)]
            k_opt = int(np.argmin(losses) + 2)
            self.search_time_s += time.perf_counter() - t0
            return ordered_cols[:k_opt]

        if self.search_strategy == "omp":
            # Жадное ортогональное сопоставление (Orthogonal Matching Pursuit) по остатку невязки
            X_cand = self.coords[local_idx][:, candidate_cols]
            X_cent = X_cand - np.mean(X_cand, axis=0, keepdims=True)
            norms = np.linalg.norm(X_cent, axis=0) + 1e-8
            X_normed = X_cent / norms

            residual = (local_gaps - np.mean(local_gaps)).copy()
            selected_local = []
            remaining = set(range(M))

            best_cost = np.inf
            best_subset = candidate_cols[:2]

            for step in range(1, M + 1):
                rem_list = list(remaining)
                corrs = np.abs(X_normed[:, rem_list].T @ residual)
                best_j = rem_list[int(np.argmax(corrs))]
                selected_local.append(best_j)
                remaining.remove(best_j)

                # Ортогональная проекция остатка на выбранное подпространство
                A_s = X_normed[:, selected_local]
                coef, _, _, _ = np.linalg.lstsq(A_s, local_gaps - np.mean(local_gaps), rcond=None)
                residual = (local_gaps - np.mean(local_gaps)) - A_s @ coef

                if step >= 2:
                    sub_cols = [candidate_cols[j] for j in selected_local]
                    sub_in = self.coords[local_idx][:, sub_cols]
                    c_in = (
                        np.mean(sub_in[target_bin == 1], axis=0)
                        if np.any(target_bin == 1)
                        else np.mean(sub_in, axis=0)
                    )
                    w = c_in.reshape(-1, 1)
                    b = -0.5 * float(np.dot(c_in, c_in))
                    act_proto = relu(sub_in @ w + b).flatten()
                    if np.std(act_proto) < 1e-8:
                        loss_k = base_err
                    else:
                        p_norm = (act_proto - np.min(act_proto)) / (
                            np.max(act_proto) - np.min(act_proto) + 1e-8
                        )
                        loss_k = float(np.mean((p_norm - target_bin) ** 2))
                    cost_k = loss_k + self.sparsity_penalty * (step / M)
                    if cost_k < best_cost:
                        best_cost = cost_k
                        best_subset = list(sub_cols)

            self.search_time_s += time.perf_counter() - t0
            return best_subset


def run_search_ablation(dataset_name="Two Spirals (32D)", n_splits=15):
    if dataset_name == "Two Spirals (32D)":
        X, y = make_two_spirals(seed=42)
        max_axes, lr = 20, 0.04
    elif dataset_name == "UCI Digits (64D)":
        X, y = load_digits(return_X_y=True)
        max_axes, lr = 56, 0.05
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    out_dim = len(np.unique(y))
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    strategies = ["hybrid_golden", "omp", "exhaustive"]
    labels = {
        "hybrid_golden": "Hybrid Golden (Ours)",
        "omp": "OMP (Greedy Orthogonal)",
        "exhaustive": "Exhaustive Search",
    }

    res = {s: {"acc": [], "f1": [], "links": [], "search_ms": []} for s in strategies}

    for seed, (tr, te) in enumerate(sss.split(X, y)):
        sc = StandardScaler()
        X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
        y_tr, y_te = y[tr], y[te]

        for s in strategies:
            model = SearchAblatedAGNN(
                search_strategy=s,
                output_dim=out_dim,
                max_axes=max_axes,
                max_units=45,
                learning_rate=lr,
                max_bridges_per_unit=4,
                seed=seed,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            stats = model.get_topology_stats()

            res[s]["acc"].append(accuracy_score(y_te, preds))
            res[s]["f1"].append(f1_score(y_te, preds, average="macro"))
            res[s]["links"].append(stats["active_links"])
            res[s]["search_ms"].append(model.search_time_s * 1000.0)

    print(f"\n=== Search Strategy Ablation: {dataset_name} ({n_splits} splits) ===")
    base_acc = np.array(res["hybrid_golden"]["acc"])

    for s in strategies:
        acc = np.array(res[s]["acc"])
        f1 = np.array(res[s]["f1"])
        links = np.array(res[s]["links"])
        s_ms = np.array(res[s]["search_ms"])

        if s == "hybrid_golden":
            stat_str = "---"
        else:
            diff = base_acc - acc
            w, t, l = int(np.sum(diff > 1e-12)), int(np.sum(np.abs(diff) <= 1e-12)), int(np.sum(diff < -1e-12))
            p_w = 1.0 if np.allclose(diff, 0) else wilcoxon(diff)[1]
            stat_str = f"W/T/L={w}/{t}/{l}, Wilcoxon p={p_w:.4e}"

        print(
            f"{labels[s]:24s} | Acc={acc.mean():.4f}±{acc.std():.4f} | "
            f"Macro-F1={f1.mean():.4f}±{f1.std():.4f} | Links={links.mean():.1f} | "
            f"SearchTime={s_ms.mean():.2f} ms | {stat_str}"
        )


if __name__ == "__main__":
    run_search_ablation("Two Spirals (32D)", n_splits=15)
    run_search_ablation("UCI Digits (64D)", n_splits=15)
