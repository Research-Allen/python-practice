"""
Growth-Policy Ablation для OptimalCrossTowerAGNN.

Сравнивает политики выбора точки роста:
  regional : полная формула AGNN (base_score * (1 + 0.6 * regional_error_mass)
             + 0.35 * neighbor_support)
  classic  : жадный рост по локальной ошибке кандидата (только base_score,
             без региональной массы ошибки и без поддержки соседей)
  random   : случайный выбор среди тех же допустимых кандидатов

"""
import time
import numpy as np
from scipy.stats import wilcoxon
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from AGNN import OptimalCrossTowerAGNN, make_two_spirals

N_SPLITS = 15


class PolicyAGNN(OptimalCrossTowerAGNN):
  """AGNN с переключаемой политикой роста."""

  def __init__(self, *args, growth_policy="regional", **kwargs):
    super().__init__(*args, **kwargs)
    assert growth_policy in ("regional", "classic", "random")
    self.growth_policy = growth_policy

  def build_global_candidates(self, X_sub, y_idx, tabu_actions, **kw):
    pool = super().build_global_candidates(X_sub, y_idx, tabu_actions, **kw)
    if not pool or self.growth_policy == "regional":
      return pool

    if self.growth_policy == "classic":
      for c in pool:
        c["score"] = c["base_score"]
      pool.sort(key=lambda c: c["score"], reverse=True)
    else:  # random
      # отдельный RNG, чтобы не сдвигать поток self.rng (инициализация,
      # перестановки в train_epoch) и сравнение оставалось честным
      if not hasattr(self, "_policy_rng"):
        self._policy_rng = np.random.RandomState(self.seed + 10_007)
      order = self._policy_rng.permutation(len(pool))
      pool = [pool[i] for i in order]
    return pool


def run_ablation(n_splits=N_SPLITS, verbose=True):
  datasets = {
      "Two Spirals (32D)": (*make_two_spirals(seed=42), 20, 0.04),
      "UCI Digits (56 axes)": (
          load_digits().data, load_digits().target, 56, 0.05
      ),
  }
  policies = ["regional", "classic", "random"]
  results = {}

  for ds_name, (X, y, max_ax, lr) in datasets.items():
    out_dim = len(np.unique(y))
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=0.2, random_state=42
    )
    res = {
        p: {"acc": [], "f1": [], "links": [], "units": [], "train_s": []}
        for p in policies
    }

    for seed, (tr, te) in enumerate(sss.split(X, y)):
      sc = StandardScaler()
      X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
      y_tr, y_te = y[tr], y[te]

      for p in policies:
        t0 = time.perf_counter()
        m = PolicyAGNN(
            output_dim=out_dim,
            max_axes=max_ax,
            max_units=45,
            learning_rate=lr,
            max_bridges_per_unit=4,
            seed=seed,
            growth_policy=p,
        )
        m.fit(X_tr, y_tr)
        dt = time.perf_counter() - t0
        pred = m.predict(X_te)
        st = m.get_topology_stats()
        res[p]["acc"].append(accuracy_score(y_te, pred))
        res[p]["f1"].append(f1_score(y_te, pred, average="macro"))
        res[p]["links"].append(st["active_links"])
        res[p]["units"].append(st["units"])
        res[p]["train_s"].append(dt)

      if verbose:
        print(
            f"[{ds_name}] split {seed + 1}/{n_splits}  "
            + "  ".join(f"{p}={res[p]['acc'][-1]:.4f}" for p in policies),
            flush=True,
        )
    results[ds_name] = res

  print_report(results, policies)
  return results


def _wilcoxon(a, b):
  d = np.asarray(a) - np.asarray(b)
  if np.allclose(d, 0):
    return 1.0
  return float(wilcoxon(d)[1])


def print_report(results, policies):
  names = {
      "regional": "Regional AGNN",
      "classic": "Classic growth",
      "random": "Random growth",
  }
  for ds_name, res in results.items():
    print("\n" + "=" * 86)
    print(ds_name)
    print("=" * 86)
    print(f"{'Policy':<16}{'Acc (mean±std)':<20}{'F1':<20}"
          f"{'Links':<12}{'Units':<10}{'Train,s':<8}")
    for p in policies:
      r = res[p]
      print(
          f"{names[p]:<16}"
          f"{np.mean(r['acc']):.4f}±{np.std(r['acc']):.4f}    "
          f"{np.mean(r['f1']):.4f}±{np.std(r['f1']):.4f}    "
          f"{np.mean(r['links']):<12.1f}{np.mean(r['units']):<10.1f}"
          f"{np.mean(r['train_s']):<8.1f}"
      )
    print("\nPaired Wilcoxon (accuracy), regional vs:")
    for p in ("classic", "random"):
      diff = np.array(res["regional"]["acc"]) - np.array(res[p]["acc"])
      w, t, l = (
          int(np.sum(diff > 1e-12)),
          int(np.sum(np.abs(diff) <= 1e-12)),
          int(np.sum(diff < -1e-12)),
      )
      print(
          f"  vs {names[p]:<15} W/T/L = {w}/{t}/{l}   "
          f"p = {_wilcoxon(res['regional']['acc'], res[p]['acc']):.4f}   "
          f"Δacc = {np.mean(diff):+.4f}"
      )


if __name__ == "__main__":
  run_ablation()
