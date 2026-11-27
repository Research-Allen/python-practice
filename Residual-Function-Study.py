import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.datasets import load_digits, load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from AGNN import OptimalCrossTowerAGNN, make_two_spirals


class ResidualAblatedAGNN(OptimalCrossTowerAGNN):
    """Сравнение функций невязки: Probability Gap vs Margin vs Cross-Entropy."""

    def __init__(self, residual_type="prob_gap", **kwargs):
        super().__init__(**kwargs)
        assert residual_type in ("prob_gap", "margin", "cross_entropy")
        self.residual_type = residual_type

    def _compute_residuals(self, probs, y_idx):
        n = len(y_idx)
        p_correct = probs[np.arange(n), y_idx]

        if self.residual_type == "prob_gap":
            return 1.0 - p_correct

        if self.residual_type == "margin":
            probs_other = probs.copy()
            probs_other[np.arange(n), y_idx] = -1.0
            p_best_other = np.max(probs_other, axis=1)
            # Сдвиг (1 + margin)/2 переводит [-1, 1] в [0, 1] с сохранением порога ошибки > 0.5
            return 0.5 * (1.0 + (p_best_other - p_correct))

        if self.residual_type == "cross_entropy":
            ce = -np.log(np.clip(p_correct, 1e-8, 1.0))
            # Мягкая нормировка в [0, 1) через 1 - exp(-ce) сохраняет монотонность и масштаб порогов
            return ce / (1.0 + ce)

    def build_global_candidates(
        self, X_sub, y_idx, tabu_actions, cluster_size=18, min_seed_dist=0.42
    ):
        probs = self.forward_sub(X_sub)
        gaps = self._compute_residuals(probs, y_idx)
        hashes = self.hasher.hash_points(X_sub)
        global_pool = []

        err_thr = 0.5 if self.residual_type == "margin" else 0.35

        for sid, sdata in self.seeds.items():
            if sdata["level"] >= 5:
                continue
            next_lvl = sdata["level"] + 1
            action_key = ("DEEPEN", sid, next_lvl)
            if action_key in tabu_actions:
                continue

            sub_axes = sdata["sub_axes"]
            dists = np.linalg.norm(X_sub[:, sub_axes] - sdata["center_sub"], axis=1)
            local_idx = np.argsort(dists)[:cluster_size]
            local_gaps = gaps[local_idx]
            err_count = np.sum(local_gaps > err_thr)

            if err_count >= 3:
                base_score = (
                    float(np.mean(local_gaps))
                    * np.log1p(err_count)
                    * (1.0 + float(np.std(local_gaps)))
                )
                center_full = np.mean(X_sub[local_idx], axis=0)
                global_pool.append({
                    "base_score": base_score,
                    "score": base_score,
                    "action": "DEEPEN",
                    "seed_id": sid,
                    "local_idx": local_idx,
                    "local_gaps": local_gaps,
                    "sub_axes": sub_axes,
                    "center_full": center_full,
                    "action_key": action_key,
                })

        thr = np.percentile(gaps, 85)
        work_mask = gaps >= thr
        if np.any(work_mask):
            dome_dim = min(self.base_dim, 5)
            for h in np.unique(hashes[work_mask]):
                h_int = int(h)
                action_key = ("NEW_SEED", h_int)
                if action_key in tabu_actions:
                    continue

                b_idx = np.where((hashes == h_int) & work_mask)[0]
                if len(b_idx) < 3:
                    continue

                if self.base_dim <= dome_dim:
                    sub_axes = list(range(self.base_dim))
                else:
                    rel_compact = np.std(X_sub, axis=0) / (np.std(X_sub[b_idx], axis=0) + 1e-8)
                    sub_axes = sorted(np.argsort(rel_compact)[::-1][:dome_dim].tolist())

                epicenter_sub = X_sub[b_idx[np.argmax(gaps[b_idx])]][sub_axes]
                too_close = any(
                    sdata["sub_axes"] == sub_axes
                    and np.linalg.norm(epicenter_sub - sdata["center_sub"]) < min_seed_dist
                    for sdata in self.seeds.values()
                )
                if too_close:
                    continue

                dists = np.linalg.norm(X_sub[:, sub_axes] - epicenter_sub, axis=1)
                local_idx = np.argsort(dists)[:cluster_size]
                local_gaps = gaps[local_idx]
                err_count = np.sum(local_gaps > err_thr)

                if err_count >= 3:
                    base_score = (
                        float(np.mean(local_gaps))
                        * np.log1p(err_count)
                        * (1.0 + float(np.std(local_gaps)))
                    )
                    center_full = np.mean(X_sub[local_idx], axis=0)
                    global_pool.append({
                        "base_score": base_score,
                        "score": base_score,
                        "action": "NEW_SEED",
                        "seed_id": None,
                        "local_idx": local_idx,
                        "local_gaps": local_gaps,
                        "sub_axes": sub_axes,
                        "center_full": center_full,
                        "action_key": action_key,
                    })

        if not global_pool:
            return []

        centers = np.array([c["center_full"] for c in global_pool])
        base_scores = np.array([c["base_score"] for c in global_pool])
        if len(centers) > 1:
            pair_dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            nonzero = pair_dists[pair_dists > 0]
            sigma_zone = max(0.5, float(np.percentile(nonzero, 30))) if len(nonzero) > 0 else 1.0
        else:
            sigma_zone = 1.0

        for i, cand in enumerate(global_pool):
            pt_dists = np.linalg.norm(X_sub - cand["center_full"], axis=1)
            nn_idx = np.argsort(pt_dists)[:250]
            kernel_pts = np.exp(-0.5 * (pt_dists[nn_idx] / sigma_zone) ** 2)
            regional_error_mass = float(
                np.sum(gaps[nn_idx] * kernel_pts) / (np.sum(kernel_pts) + 1e-8)
            )
            if len(centers) > 1:
                c_dists = np.linalg.norm(centers - cand["center_full"], axis=1)
                kernel_cands = np.exp(-0.5 * (c_dists / sigma_zone) ** 2)
                kernel_cands[i] = 0.0
                neighbor_support = float(np.sum(base_scores * kernel_cands))
            else:
                neighbor_support = 0.0

            cand["score"] = (
                cand["base_score"] * (1.0 + 0.6 * regional_error_mass)
                + 0.35 * neighbor_support
            )

        global_pool.sort(key=lambda c: c["score"], reverse=True)
        return global_pool


def run_residual_study(dataset_name="Two Spirals (32D)", n_splits=15):
    if dataset_name == "Two Spirals (32D)":
        X, y = make_two_spirals(seed=42)
        max_axes, lr = 20, 0.04
    elif dataset_name == "Wine (13D)":
        X, y = load_wine(return_X_y=True)
        max_axes, lr = 13, 0.04
    elif dataset_name == "UCI Digits (64D)":
        X, y = load_digits(return_X_y=True)
        max_axes, lr = 56, 0.05
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    out_dim = len(np.unique(y))
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    res_types = ["prob_gap", "margin", "cross_entropy"]
    labels = {
        "prob_gap": "Probability Gap (1 - p)",
        "margin": "Margin (p_best - p_true)",
        "cross_entropy": "Cross-Entropy (-log p)",
    }

    res = {r: {"acc": [], "f1": [], "links": [], "units": []} for r in res_types}

    for seed, (tr, te) in enumerate(sss.split(X, y)):
        sc = StandardScaler()
        X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
        y_tr, y_te = y[tr], y[te]

        for r in res_types:
            model = ResidualAblatedAGNN(
                residual_type=r,
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

            res[r]["acc"].append(accuracy_score(y_te, preds))
            res[r]["f1"].append(f1_score(y_te, preds, average="macro"))
            res[r]["links"].append(stats["active_links"])
            res[r]["units"].append(stats["units"])

    print(f"\n=== Residual Function Study: {dataset_name} ({n_splits} splits) ===")
    base_acc = np.array(res["prob_gap"]["acc"])

    for r in res_types:
        acc = np.array(res[r]["acc"])
        f1 = np.array(res[r]["f1"])
        links = np.array(res[r]["links"])
        units = np.array(res[r]["units"])

        if r == "prob_gap":
            stat_str = "---"
        else:
            diff = base_acc - acc
            w, t, l = int(np.sum(diff > 1e-12)), int(np.sum(np.abs(diff) <= 1e-12)), int(np.sum(diff < -1e-12))
            p_w = 1.0 if np.allclose(diff, 0) else wilcoxon(diff)[1]
            stat_str = f"W/T/L={w}/{t}/{l}, Wilcoxon p={p_w:.4e}"

        print(
            f"{labels[r]:25s} | Acc={acc.mean():.4f}±{acc.std():.4f} | "
            f"Macro-F1={f1.mean():.4f}±{f1.std():.4f} | Units={units.mean():.1f} | "
            f"Links={links.mean():.1f} | {stat_str}"
        )


if __name__ == "__main__":
    run_residual_study("Two Spirals (32D)", n_splits=15)
    run_residual_study("Wine (13D)", n_splits=15)
    run_residual_study("UCI Digits (64D)", n_splits=15)
