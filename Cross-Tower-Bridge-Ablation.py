import time
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from AGNN import OptimalCrossTowerAGNN, relu


class AblatedBridgeAGNN(OptimalCrossTowerAGNN):
    """Subclass of OptimalCrossTowerAGNN with a complete toggle for all cross-tower bridges."""

    def __init__(self, enable_bridges=True, **kwargs):
        super().__init__(**kwargs)
        self.enable_bridges = enable_bridges
        if not enable_bridges:
            self.max_bridges_per_unit = 0

    def apply_growth(self, X_sub, y_idx, action, seed_id, local_idx, local_gaps, sub_axes):
        if self.enable_bridges:
            return super().apply_growth(X_sub, y_idx, action, seed_id, local_idx, local_gaps, sub_axes)

        # Isolated-tower growth (no same_3d lateral bridges, no same_lvl_peers)
        self.forward_sub(X_sub)
        center_sub = np.mean(X_sub[local_idx][:, sub_axes], axis=0)
        spread_sub = np.mean(np.std(X_sub[local_idx][:, sub_axes], axis=0))

        if action == "NEW_SEED":
            sid = self.next_seed_id
            self.next_seed_id += 1
            level = 3
            sigma = float(np.clip(spread_sub * 1.4, 0.25, 1.0))

            unit = {
                "type": "local_dome",
                "seed_id": sid,
                "level": level,
                "sub_axes": list(sub_axes),
                "center": center_sub.copy(),
                "sigma": sigma,
                "bridge_indices": [],
                "bridge_weights": np.zeros((0, 1)),
                "bridge_mask": np.zeros((0, 1)),
            }
            self.seeds[sid] = {
                "center_sub": center_sub.copy(),
                "sub_axes": list(sub_axes),
                "unit_indices": [len(self.units)],
                "level": level,
            }
            self.units.append(unit)
        else:
            sid = seed_id
            seed = self.seeds[sid]
            own_cols = [self.base_dim + u_idx for u_idx in seed["unit_indices"]]
            new_level = seed["level"] + 1
            # Strictly own-tower units + base input axes (zero cross-tower peers)
            all_candidate_cols = list(range(self.base_dim)) + own_cols
            input_indices = self.select_connections_golden(
                all_candidate_cols, local_idx, local_gaps, y_idx
            )

            sub_in = self.coords[local_idx][:, input_indices]
            center_in = np.mean(sub_in, axis=0)
            spread_in = np.std(sub_in, axis=0)

            in_dim = len(input_indices)
            new_w = (center_in + 0.08 * spread_in * self.rng.randn(in_dim)).reshape(in_dim, 1)
            new_w /= np.linalg.norm(new_w) + 1e-8
            new_b = np.array([[-float(center_in @ new_w)]])

            seed["level"] = new_level
            seed["unit_indices"].append(len(self.units))

            unit = {
                "type": "deep_axis",
                "seed_id": sid,
                "level": new_level,
                "sub_axes": list(seed["sub_axes"]),
                "center": center_sub.copy(),
                "input_indices": input_indices,
                "weights": new_w,
                "mask": np.ones((in_dim, 1)),
                "bias": new_b,
            }
            self.units.append(unit)

        new_out_w = self.rng.randn(1, self.output_dim) * 0.25
        dom_class = int(np.bincount(y_idx[local_idx]).argmax())
        new_out_w[0, dom_class] += 0.55

        self.W_out = np.vstack([self.W_out, new_out_w])
        self.out_mask = np.vstack([self.out_mask, np.ones((1, self.output_dim))])

    def optimize_inter_tower_bridges(self, X_tr_sub, y_tr_oh, X_val_sub, y_val, y_val_oh):
        if not self.enable_bridges:
            return 0
        return super().optimize_inter_tower_bridges(X_tr_sub, y_tr_oh, X_val_sub, y_val, y_val_oh)


def run_bridge_ablation(max_axes=56, n_splits=15):
    X, y = load_digits(return_X_y=True)
    out_dim = len(np.unique(y))
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    res = {
        True: {"acc": [], "f1": [], "links": [], "bridges": [], "units": []},
        False: {"acc": [], "f1": [], "links": [], "bridges": [], "units": []},
    }

    for seed, (tr, te) in enumerate(sss.split(X, y)):
        sc = StandardScaler()
        X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
        y_tr, y_te = y[tr], y[te]

        for enable_bridges in [True, False]:
            model = AblatedBridgeAGNN(
                enable_bridges=enable_bridges,
                output_dim=out_dim,
                max_axes=max_axes,
                max_units=45,
                learning_rate=0.05,
                max_bridges_per_unit=4 if enable_bridges else 0,
                seed=seed,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            stats = model.get_topology_stats()

            res[enable_bridges]["acc"].append(accuracy_score(y_te, preds))
            res[enable_bridges]["f1"].append(f1_score(y_te, preds, average="macro"))
            res[enable_bridges]["links"].append(stats["active_links"])
            res[enable_bridges]["bridges"].append(stats["cross_tower_bridges"])
            res[enable_bridges]["units"].append(stats["units"])

        print(
            f"Split {seed+1:2d}/{n_splits} | "
            f"Full: Acc={res[True]['acc'][-1]:.4f} (Links={res[True]['links'][-1]}, Br={res[True]['bridges'][-1]}) | "
            f"NoBr: Acc={res[False]['acc'][-1]:.4f} (Links={res[False]['links'][-1]}, Br={res[False]['bridges'][-1]})",
            flush=True,
        )

    acc_on, acc_off = np.array(res[True]["acc"]), np.array(res[False]["acc"])
    f1_on, f1_off = np.array(res[True]["f1"]), np.array(res[False]["f1"])
    links_on, links_off = np.array(res[True]["links"]), np.array(res[False]["links"])
    br_on, br_off = np.array(res[True]["bridges"]), np.array(res[False]["bridges"])

    diff = acc_on - acc_off
    wins, ties, losses = int(np.sum(diff > 1e-12)), int(np.sum(np.abs(diff) <= 1e-12)), int(np.sum(diff < -1e-12))
    _, p_t = ttest_rel(acc_on, acc_off)
    p_w = 1.0 if np.allclose(diff, 0) else wilcoxon(diff)[1]

    print(f"\n=== Bridge Ablation Summary: UCI Digits (max_axes={max_axes}, {n_splits} splits) ===")
    print(f"Full AGNN (Bridges ON)  : Acc={acc_on.mean():.4f}±{acc_on.std():.4f} | Macro-F1={f1_on.mean():.4f}±{f1_on.std():.4f} | Links={links_on.mean():.1f} | Bridges={br_on.mean():.1f}")
    print(f"No Bridges (Bridges OFF): Acc={acc_off.mean():.4f}±{acc_off.std():.4f} | Macro-F1={f1_off.mean():.4f}±{f1_off.std():.4f} | Links={links_off.mean():.1f} | Bridges={br_off.mean():.1f}")
    print(f"W/T/L = {wins}/{ties}/{losses} | Paired t-test p = {p_t:.4e} | Wilcoxon p = {p_w:.4e}")


if __name__ == "__main__":
    run_bridge_ablation(max_axes=56, n_splits=15)