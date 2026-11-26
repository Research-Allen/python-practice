import time
import warnings
import numpy as np
from scipy.stats import wilcoxon
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

N_SPLITS = 15


# ==============================================================================
# 1. CORE PRIMITIVES & SYNTHETIC BENCHMARK GENERATOR
# ==============================================================================
def relu(x):
  return np.maximum(0.0, x)


def relu_grad(x):
  return (x > 0.0).astype(float)


def softmax(z):
  z = np.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
  z_shift = z - np.max(z, axis=1, keepdims=True)
  exp_z = np.exp(np.clip(z_shift, -50.0, 50.0))
  return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)


def cross_entropy(y_oh, probs):
  p = np.clip(probs, 1e-8, 1.0)
  return -float(np.mean(np.sum(y_oh * np.log(p), axis=1)))


def make_two_spirals(
    samples=1000, turns=2.0, noise=0.12, noise_dims=30, seed=42
):
  rng = np.random.RandomState(seed)
  n = samples // 2
  t = np.linspace(0.3, turns * 2 * np.pi, n)

  x0 = t * np.cos(t) + rng.randn(n) * noise
  y0 = t * np.sin(t) + rng.randn(n) * noise
  x1 = t * np.cos(t + np.pi) + rng.randn(n) * noise
  y1 = t * np.sin(t + np.pi) + rng.randn(n) * noise

  X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
  if noise_dims > 0:
    X = np.hstack([X, rng.randn(samples, noise_dims)])

  y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
  noise_idx = rng.choice(samples, int(0.03 * samples), replace=False)
  y[noise_idx] = 1 - y[noise_idx]

  perm = rng.permutation(samples)
  return X[perm], y[perm]


# ==============================================================================
# 2. SUBSPACE SELECTION & SPATIAL HASHING
# ==============================================================================
def detect_informative_axes(X, y, max_axes=22):
  n, d = X.shape
  if d <= max_axes:
    return np.arange(d)

  classes = np.unique(y)
  sub = min(n, 350)
  Xs, ys = X[:sub], y[:sub]

  single_nn = np.zeros(d)
  for i in range(d):
    dist1 = (Xs[:, i : i + 1] - Xs[:, i : i + 1].T) ** 2
    np.fill_diagonal(dist1, np.inf)
    single_nn[i] = np.mean(ys == ys[np.argmin(dist1, axis=1)])

  selected = set()
  if len(classes) == 2:
    for i in range(d):
      for j in range(i + 1, d):
        dist2 = (Xs[:, i : i + 1] - Xs[:, i : i + 1].T) ** 2 + (
            Xs[:, j : j + 1] - Xs[:, j : j + 1].T
        ) ** 2
        np.fill_diagonal(dist2, np.inf)
        pair_acc = np.mean(ys == ys[np.argmin(dist2, axis=1)])
        if pair_acc >= 0.68 and (pair_acc - max(single_nn[i], single_nn[j])) >= 0.08:
          selected.add(i)
          selected.add(j)

  if len(selected) >= 2 and d <= 40:
    return np.array(sorted(list(selected))[:max_axes])

  overall_mean = np.mean(X, axis=0)
  between_var = np.zeros(d)
  within_var = np.zeros(d)
  for c in classes:
    Xc = X[y == c]
    if len(Xc) > 0:
      w = len(Xc) / n
      between_var += w * ((np.mean(Xc, axis=0) - overall_mean) ** 2)
      within_var += w * np.var(Xc, axis=0)

  fisher = between_var / (within_var + 1e-6)
  top_fisher = set(np.argsort(fisher)[::-1][:max_axes].tolist())
  top_nn = set(np.argsort(single_nn)[::-1][: max(4, max_axes // 3)].tolist())
  combined = sorted(list(selected | top_fisher | top_nn))

  if len(combined) > max_axes:
    arr = np.array(combined)
    combined = sorted(arr[np.argsort(fisher[arr])[::-1][:max_axes]].tolist())
  return np.array(combined)


class BaseSpatialHasher:

  def __init__(self, base_dim, n_planes=10, seed=42):
    rng = np.random.RandomState(seed)
    self.planes = rng.randn(base_dim, n_planes)
    self.offsets = rng.uniform(-1.0, 1.0, size=(1, n_planes))
    self.n_planes = n_planes

  def hash_points(self, X):
    bits = ((X @ self.planes + self.offsets) > 0).astype(int)
    powers = 1 << np.arange(self.n_planes)
    return bits @ powers


def golden_section_discrete_search(eval_fn, low, high):
  low, high = int(low), int(high)
  if high <= low:
    return low
  inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
  a, b = low, high
  x1 = int(round(b - inv_phi * (b - a)))
  x2 = int(round(a + inv_phi * (b - a)))
  f1, f2 = eval_fn(x1), eval_fn(x2)

  while (b - a) > 1:
    if f1 < f2:
      b, x2, f2 = x2, x1, f1
      x1 = int(round(b - inv_phi * (b - a)))
      f1 = eval_fn(x1)
    else:
      a, x1, f1 = x1, x2, f2
      x2 = int(round(a + inv_phi * (b - a)))
      f2 = eval_fn(x2)

  return int(np.clip(x1 if f1 <= f2 else x2, low, high))


# ==============================================================================
# 3. OptimalCrossTowerAGNN (FINAL v2.3)
# ==============================================================================
class OptimalCrossTowerAGNN:

  def __init__(
      self,
      output_dim=2,
      max_axes=22,
      max_units=45,
      learning_rate=0.04,
      sparsity_penalty=0.025,
      max_bridges_per_unit=4,
      conn_selector="hybrid_golden",
      residual_mode="prob_gap",
      seed=42,
  ):
    self.output_dim = output_dim
    self.max_axes = max_axes
    self.max_units = max_units
    self.base_lr = learning_rate
    self.lr = learning_rate
    self.sparsity_penalty = sparsity_penalty
    self.max_bridges_per_unit = max_bridges_per_unit
    self.conn_selector = conn_selector
    self.residual_mode = residual_mode
    self.seed = seed

  def _init_graph(self, base_dim):
    self.rng = np.random.RandomState(self.seed)
    self.base_dim = base_dim
    self.units = []
    self.seeds = {}
    self.next_seed_id = 1

    self.W_out = self.rng.randn(base_dim, self.output_dim) * np.sqrt(
        2.0 / base_dim
    )
    self.out_mask = np.ones((base_dim, self.output_dim))
    self.b_out = np.zeros((1, self.output_dim))
    self.hasher = BaseSpatialHasher(base_dim, n_planes=10, seed=self.seed)

  @property
  def total_dim(self):
    return self.base_dim + len(self.units)

  def _compute_residuals(self, probs, y_idx):
    p_true = probs[np.arange(len(y_idx)), y_idx]
    if self.residual_mode == "margin" and probs.shape[1] > 1:
      p_other = probs.copy()
      p_other[np.arange(len(y_idx)), y_idx] = -1.0
      p_second = np.max(p_other, axis=1)
      return 0.5 * (1.0 + p_second - p_true)
    return 1.0 - p_true

  def forward_sub(self, X_sub):
    m = len(X_sub)
    self.coords = np.zeros((m, self.total_dim))
    self.coords[:, : self.base_dim] = X_sub
    self.cache = []

    for idx, u in enumerate(self.units):
      if u["type"] == "local_dome":
        sub_axes = u["sub_axes"]
        diff = X_sub[:, sub_axes] - u["center"]
        sq_dist = np.sum(diff**2, axis=1, keepdims=True)
        sigma2 = max(u["sigma"] ** 2, 1e-6)
        dome_act = np.exp(-0.5 * sq_dist / sigma2)

        if len(u["bridge_indices"]) > 0:
          br_in = self.coords[:, u["bridge_indices"]]
          br_w = u["bridge_weights"] * u["bridge_mask"]
          lateral = np.tanh(br_in @ br_w)
          act = dome_act * (1.0 + 0.4 * lateral)
          self.cache.append((diff, dome_act, lateral))
        else:
          act = dome_act
          self.cache.append((diff, dome_act, None))

        self.coords[:, self.base_dim + idx] = act.flatten()
      else:
        in_idx = u["input_indices"]
        sub_in = self.coords[:, in_idx]
        w_eff = u["weights"] * u["mask"]
        z = sub_in @ w_eff + u["bias"]
        self.cache.append((z, None, None))
        self.coords[:, self.base_dim + idx] = relu(z).flatten()

    w_out_eff = self.W_out * self.out_mask
    self.z_out = self.coords @ w_out_eff + self.b_out
    self.output = softmax(self.z_out)
    return self.output

  def backward(
      self, y_oh, frozen_before_idx=0, l2_reg=1e-4, return_unit_grads=False
  ):
    m = len(y_oh)
    dz_out = self.output - y_oh
    w_out_eff = self.W_out * self.out_mask

    dW_out = (self.coords.T @ dz_out) / m + l2_reg * w_out_eff
    db_out = np.mean(dz_out, axis=0, keepdims=True)
    d_coords = dz_out @ w_out_eff.T

    if not return_unit_grads:
      self.W_out -= self.lr * (dW_out * self.out_mask)
      self.b_out -= self.lr * db_out

    unit_dz = {}
    for idx in range(len(self.units) - 1, -1, -1):
      u = self.units[idx]
      col = self.base_dim + idx
      d_act = d_coords[:, col : col + 1]

      if u["type"] == "local_dome":
        diff, dome_act, lateral = self.cache[idx]
        if lateral is not None and len(u["bridge_indices"]) > 0:
          d_lat = d_act * dome_act * 0.4 * (1.0 - lateral**2)
          unit_dz[idx] = d_lat
          br_in = self.coords[:, u["bridge_indices"]]
          br_w = u["bridge_weights"] * u["bridge_mask"]
          d_coords[:, u["bridge_indices"]] += d_lat @ br_w.T
          if not return_unit_grads and idx >= frozen_before_idx:
            dW_br = (br_in.T @ d_lat) / m + l2_reg * br_w
            u["bridge_weights"] -= self.lr * (dW_br * u["bridge_mask"])
          d_dome = d_act * (1.0 + 0.4 * lateral)
        else:
          unit_dz[idx] = d_act
          d_dome = d_act

        if not return_unit_grads and idx >= frozen_before_idx:
          sigma2 = max(u["sigma"] ** 2, 1e-6)
          sigma3 = max(u["sigma"] ** 3, 1e-9)
          d_center = np.mean(d_dome * dome_act * (diff / sigma2), axis=0)
          sq_dist = np.sum(diff**2, axis=1, keepdims=True)
          d_sigma = np.mean(d_dome * dome_act * (sq_dist / sigma3))
          u["center"] -= self.lr * 0.5 * d_center
          u["sigma"] = np.clip(u["sigma"] - self.lr * 0.5 * d_sigma, 0.15, 2.0)
      else:
        z, _, _ = self.cache[idx]
        dz = d_act * relu_grad(z)
        unit_dz[idx] = dz
        in_idx = u["input_indices"]
        sub_in = self.coords[:, in_idx]
        w_eff = u["weights"] * u["mask"]
        d_coords[:, in_idx] += dz @ w_eff.T

        if not return_unit_grads and idx >= frozen_before_idx:
          dW = (sub_in.T @ dz) / m + l2_reg * w_eff
          db = np.mean(dz, axis=0, keepdims=True)
          u["weights"] -= self.lr * (dW * u["mask"])
          u["bias"] -= self.lr * db

    if return_unit_grads:
      return unit_dz

  def train_epoch(self, X_sub, y_oh, frozen_before_idx=0, batch_size=32):
    idx = self.rng.permutation(len(X_sub))
    for start in range(0, len(X_sub), batch_size):
      batch = idx[start : start + batch_size]
      self.forward_sub(X_sub[batch])
      self.backward(y_oh[batch], frozen_before_idx=frozen_before_idx)

  def build_global_candidates(
      self, X_sub, y_idx, tabu_actions, cluster_size=18, min_seed_dist=0.42
  ):
    probs = self.forward_sub(X_sub)
    gaps = self._compute_residuals(probs, y_idx)
    hashes = self.hasher.hash_points(X_sub)
    global_pool = []

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
      err_count = np.sum(local_gaps > 0.35)

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
          rel_compact = np.std(X_sub, axis=0) / (
              np.std(X_sub[b_idx], axis=0) + 1e-8
          )
          sub_axes = sorted(np.argsort(rel_compact)[::-1][:dome_dim].tolist())

        epicenter_sub = X_sub[b_idx[np.argmax(gaps[b_idx])]][sub_axes]
        too_close = any(
            sdata["sub_axes"] == sub_axes
            and np.linalg.norm(epicenter_sub - sdata["center_sub"])
            < min_seed_dist
            for sdata in self.seeds.values()
        )
        if too_close:
          continue

        dists = np.linalg.norm(X_sub[:, sub_axes] - epicenter_sub, axis=1)
        local_idx = np.argsort(dists)[:cluster_size]
        local_gaps = gaps[local_idx]
        err_count = np.sum(local_gaps > 0.35)

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
      pair_dists = np.linalg.norm(
          centers[:, None, :] - centers[None, :, :], axis=2
      )
      nonzero = pair_dists[pair_dists > 0]
      sigma_zone = (
          float(np.clip(np.percentile(nonzero, 30), 0.5, 5.0))
          if len(nonzero) > 0
          else 1.0
      )
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

  def _select_connections_omp(self, candidate_cols, local_idx, local_gaps):
    M = len(candidate_cols)
    max_k = min(M, max(2, M // 2))
    r = local_gaps - np.mean(local_gaps)
    selected = []
    basis = []
    for _ in range(max_k):
      best_c, best_score = None, -1.0
      for col in candidate_cols:
        if col in selected:
          continue
        v = self.coords[local_idx, col] - np.mean(self.coords[local_idx, col])
        for b in basis:
          v = v - np.dot(v, b) * b
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-6:
          continue
        v_unit = v / norm_v
        sc = abs(float(np.dot(v_unit, r)))
        if sc > best_score:
          best_score = sc
          best_c = (col, v_unit)
      if best_c is None:
        break
      selected.append(best_c[0])
      basis.append(best_c[1])
      r = r - np.dot(r, best_c[1]) * best_c[1]
    return selected if len(selected) >= 2 else candidate_cols[:2]

  def select_connections_golden(
      self, candidate_cols, local_idx, local_gaps, y_idx
  ):
    M = len(candidate_cols)
    if M <= 2:
      return candidate_cols

    if self.conn_selector == "omp":
      return self._select_connections_omp(candidate_cols, local_idx, local_gaps)

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

    y_loc = y_idx[local_idx]
    dom_cls = int(np.bincount(y_loc).argmax())
    target_bin = (y_loc == dom_cls).astype(float)
    base_err = float(np.mean((0.5 - target_bin) ** 2))

    def eval_cost(k):
      k = int(np.clip(k, 2, M))
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

    if M <= 20:
      losses = [eval_cost(k) for k in range(2, M + 1)]
      k_opt = int(np.argmin(losses) + 2)
    else:
      k_opt = golden_section_discrete_search(eval_cost, low=2, high=M)

    return ordered_cols[:k_opt]

  def apply_growth(
      self, X_sub, y_idx, action, seed_id, local_idx, local_gaps, sub_axes
  ):
    self.forward_sub(X_sub)
    center_sub = np.mean(X_sub[local_idx][:, sub_axes], axis=0)
    spread_sub = np.mean(np.std(X_sub[local_idx][:, sub_axes], axis=0))

    if action == "NEW_SEED":
      sid = self.next_seed_id
      self.next_seed_id += 1
      level = 3
      sigma = float(np.clip(spread_sub * 1.4, 0.25, 1.0))

      peer_indices = []
      same_3d = [
          self.base_dim + i
          for i, u in enumerate(self.units)
          if u["seed_id"] != sid and u["level"] == 3
      ]
      if len(same_3d) > 0:
        peer_indices = self.select_connections_golden(
            same_3d, local_idx, local_gaps, y_idx
        )[:2]

      peer_w = (
          self.rng.randn(len(peer_indices), 1) * 0.12
          if peer_indices
          else np.zeros((0, 1))
      )
      peer_m = (
          np.ones((len(peer_indices), 1)) if peer_indices else np.zeros((0, 1))
      )

      unit = {
          "type": "local_dome",
          "seed_id": sid,
          "level": level,
          "sub_axes": list(sub_axes),
          "center": center_sub.copy(),
          "sigma": sigma,
          "bridge_indices": peer_indices,
          "bridge_weights": peer_w,
          "bridge_mask": peer_m,
      }
      new_idx = len(self.units)
      self.units.append(unit)
      self.seeds[sid] = {
          "center_sub": center_sub.copy(),
          "sub_axes": list(sub_axes),
          "unit_indices": [new_idx],
          "level": level,
      }
    else:
      sid = seed_id
      seed = self.seeds[sid]
      own_cols = [self.base_dim + u_idx for u_idx in seed["unit_indices"]]
      new_level = seed["level"] + 1
      same_lvl_peers = [
          self.base_dim + idx
          for idx, u in enumerate(self.units)
          if u["seed_id"] != sid and u["level"] == new_level
      ]
      all_candidate_cols = (
          list(range(self.base_dim)) + own_cols + same_lvl_peers
      )
      input_indices = self.select_connections_golden(
          all_candidate_cols, local_idx, local_gaps, y_idx
      )

      sub_in = self.coords[local_idx][:, input_indices]
      center_in = np.mean(sub_in, axis=0)
      spread_in = np.std(sub_in, axis=0)

      in_dim = len(input_indices)
      new_w = (
          center_in + 0.08 * spread_in * self.rng.randn(in_dim)
      ).reshape(in_dim, 1)
      new_w /= np.linalg.norm(new_w) + 1e-8
      new_b = np.array([[-float(center_in @ new_w)]])

      new_idx = len(self.units)
      self.units.append({
          "type": "deep_axis",
          "seed_id": sid,
          "level": new_level,
          "sub_axes": list(seed["sub_axes"]),
          "center": center_sub.copy(),
          "input_indices": input_indices,
          "weights": new_w,
          "mask": np.ones((in_dim, 1)),
          "bias": new_b,
      })
      seed["level"] = new_level
      seed["unit_indices"].append(new_idx)

    new_out_w = self.rng.randn(1, self.output_dim) * 0.25
    dom_class = int(np.bincount(y_idx[local_idx]).argmax())
    new_out_w[0, dom_class] += 0.55

    self.W_out = np.vstack([self.W_out, new_out_w])
    self.out_mask = np.vstack([self.out_mask, np.ones((1, self.output_dim))])

  def optimize_inter_tower_bridges(
      self, X_tr_sub, y_tr_oh, X_val_sub, y_val, y_val_oh
  ):
    if len(self.seeds) < 2:
      return 0

    self.forward_sub(X_tr_sub)
    unit_dz = self.backward(y_tr_oh, return_unit_grads=True)

    bridge_candidates = []
    for dst_idx, u_dst in enumerate(self.units):
      if u_dst["type"] == "local_dome":
        if len(u_dst["bridge_indices"]) >= self.max_bridges_per_unit:
          continue
      else:
        own_sid = u_dst["seed_id"]
        own_cols = {
            self.base_dim + idx for idx in self.seeds[own_sid]["unit_indices"]
        }
        ext_count = sum(
            1
            for col in u_dst["input_indices"]
            if col >= self.base_dim and col not in own_cols
        )
        if ext_count >= self.max_bridges_per_unit:
          continue

      dz_dst = unit_dz[dst_idx]
      existing_inputs = set(
          u_dst["bridge_indices"]
          if u_dst["type"] == "local_dome"
          else u_dst["input_indices"]
      )

      for src_idx in range(dst_idx):
        u_src = self.units[src_idx]
        if u_src["seed_id"] == u_dst["seed_id"]:
          continue
        src_col = self.base_dim + src_idx
        if src_col in existing_inputs:
          continue

        a_src = self.coords[:, src_col : src_col + 1]
        std_src = np.std(a_src)
        std_dst = np.std(dz_dst)
        if std_src < 1e-6 or std_dst < 1e-6:
          continue

        num = np.mean(a_src * dz_dst)
        grad_score = min(abs(float(num / (std_src * std_dst))), 10.0)

        level_bonus = 1.25 if u_src["level"] == u_dst["level"] else 1.0
        bridge_candidates.append(
            (grad_score * level_bonus, grad_score, src_col, dst_idx)
        )

    bridge_candidates = [b for b in bridge_candidates if np.isfinite(b[0])]
    if not bridge_candidates:
      return 0

    bridge_candidates.sort(key=lambda x: x[0], reverse=True)
    top_bridges = bridge_candidates[: min(12, len(bridge_candidates))]
    M = len(top_bridges)

    base_probs = self.forward_sub(X_val_sub)
    base_acc = np.mean(np.argmax(base_probs, axis=1) == y_val)
    base_loss = cross_entropy(y_val_oh, base_probs)
    base_state = self.save_state()
    cache_eval = {}

    def get_eval(k):
      k = int(np.clip(k, 1, M))
      if k not in cache_eval:
        self.restore_state(base_state)
        self.forward_sub(X_tr_sub)
        dz_now = self.backward(y_tr_oh, return_unit_grads=True)
        for _, _, src_col, dst_idx in top_bridges[:k]:
          u_dst = self.units[dst_idx]
          if u_dst["type"] == "local_dome":
            if len(u_dst["bridge_indices"]) >= self.max_bridges_per_unit:
              continue
          else:
            own_sid = u_dst["seed_id"]
            own_cols = {
                self.base_dim + idx
                for idx in self.seeds[own_sid]["unit_indices"]
            }
            ext_count = sum(
                1
                for col in u_dst["input_indices"]
                if col >= self.base_dim and col not in own_cols
            )
            if ext_count >= self.max_bridges_per_unit:
              continue

          a_src = self.coords[:, src_col : src_col + 1]
          raw_grad = float(np.mean(a_src * dz_now[dst_idx]))
          if abs(raw_grad) < 1e-5:
            init_w = float(self.rng.uniform(-0.05, 0.05))
          else:
            init_w = float(
                np.clip(
                    -0.35
                    * np.sign(raw_grad)
                    * min(1.0, abs(raw_grad) * 10.0),
                    -0.4,
                    0.4,
                )
            )

          if u_dst["type"] == "local_dome":
            u_dst["bridge_indices"].append(src_col)
            u_dst["bridge_weights"] = np.vstack(
                [u_dst["bridge_weights"], [[init_w]]]
            )
            u_dst["bridge_mask"] = np.vstack([u_dst["bridge_mask"], [[1.0]]])
          else:
            u_dst["input_indices"].append(src_col)
            u_dst["weights"] = np.vstack([u_dst["weights"], [[init_w]]])
            u_dst["mask"] = np.vstack([u_dst["mask"], [[1.0]]])

        old_lr = self.lr
        self.lr = self.base_lr * 0.35
        for _ in range(35):
          self.train_epoch(X_tr_sub, y_tr_oh, frozen_before_idx=0)
        self.lr = old_lr

        p_val = self.forward_sub(X_val_sub)
        acc_v = np.mean(np.argmax(p_val, axis=1) == y_val)
        loss_v = cross_entropy(y_val_oh, p_val)
        cost_v = loss_v + 0.015 * (k / M)
        cache_eval[k] = (cost_v, acc_v, loss_v, self.save_state())
      return cache_eval[k]

    best_k = (
        1
        if M == 1
        else golden_section_discrete_search(
            lambda k: get_eval(k)[0], low=1, high=M
        )
    )
    _, best_acc_k, best_loss_k, best_st_k = get_eval(best_k)

    if (best_acc_k > base_acc + 0.001) or (
        best_loss_k < base_loss - 0.003 and best_acc_k >= base_acc
    ):
      self.restore_state(best_st_k)
      return best_k

    self.restore_state(base_state)
    return 0

  def prune_weak_connections(self, X_sub, rel_threshold=0.16):
    self.forward_sub(X_sub)
    for u in self.units:
      if u["type"] == "local_dome":
        if len(u["bridge_indices"]) > 0:
          br_in = self.coords[:, u["bridge_indices"]]
          mean_act = np.mean(np.abs(br_in), axis=0, keepdims=True).T
          imp = np.abs(u["bridge_weights"] * u["bridge_mask"]) * mean_act
          active = imp[u["bridge_mask"] > 0]
          if len(active) > 0:
            cutoff = np.max(active) * 0.22
            u["bridge_mask"] = (imp >= cutoff).astype(float)
            u["bridge_weights"] *= u["bridge_mask"]
        continue

      in_idx = u["input_indices"]
      sub_in = self.coords[:, in_idx]
      mean_act = np.mean(np.abs(sub_in), axis=0, keepdims=True).T
      imp = np.abs(u["weights"] * u["mask"]) * mean_act
      active = imp[u["mask"] > 0]
      if len(active) <= 2:
        continue
      cutoff = np.max(active) * rel_threshold
      new_mask = (imp >= cutoff).astype(float)
      if np.sum(new_mask) < 2:
        top2 = np.argsort(imp.flatten())[::-1][:2]
        new_mask = np.zeros_like(u["mask"])
        new_mask[top2, 0] = 1.0
      u["mask"] = new_mask
      u["weights"] *= new_mask

  def prune_redundant_outputs(
      self, X_tr_sub, X_val_sub, y_val, y_val_oh, loss_tol=0.002
  ):
    self.forward_sub(X_tr_sub)
    std_coords = np.std(self.coords, axis=0, keepdims=True).T
    importance = np.abs(self.W_out * self.out_mask) * std_coords

    probs_val = self.forward_sub(X_val_sub)
    base_preds = np.argmax(probs_val, axis=1)
    base_acc = accuracy_score(y_val, base_preds)
    base_f1 = f1_score(y_val, base_preds, average="macro")
    base_score = 0.5 * base_acc + 0.5 * base_f1
    base_loss = cross_entropy(y_val_oh, probs_val)

    for flat_idx in np.argsort(importance.ravel()):
      r, c = divmod(flat_idx, self.output_dim)
      if self.out_mask[r, c] == 0:
        continue
      if r >= self.base_dim and np.sum(self.out_mask[r, :]) <= 1:
        continue

      self.out_mask[r, c] = 0.0
      probs_new = self.forward_sub(X_val_sub)
      preds = np.argmax(probs_new, axis=1)
      new_acc = accuracy_score(y_val, preds)
      new_f1 = f1_score(y_val, preds, average="macro")
      new_score = 0.5 * new_acc + 0.5 * new_f1
      new_loss = cross_entropy(y_val_oh, probs_new)

      if new_score >= base_score and new_loss <= base_loss + loss_tol:
        self.W_out[r, c] = 0.0
      else:
        self.out_mask[r, c] = 1.0

  def save_state(self):
    u_copy = []
    for u in self.units:
      d = {
          "type": u["type"],
          "seed_id": u["seed_id"],
          "level": u["level"],
          "sub_axes": list(u["sub_axes"]),
          "center": u["center"].copy(),
      }
      if u["type"] == "local_dome":
        d["sigma"] = u["sigma"]
        d["bridge_indices"] = list(u["bridge_indices"])
        d["bridge_weights"] = u["bridge_weights"].copy()
        d["bridge_mask"] = u["bridge_mask"].copy()
      else:
        d["input_indices"] = list(u["input_indices"])
        d["weights"] = u["weights"].copy()
        d["mask"] = u["mask"].copy()
        d["bias"] = u["bias"].copy()
      u_copy.append(d)
    s_copy = {
        k: {
            "center_sub": v["center_sub"].copy(),
            "sub_axes": list(v["sub_axes"]),
            "unit_indices": list(v["unit_indices"]),
            "level": v["level"],
        }
        for k, v in self.seeds.items()
    }
    return (
        u_copy,
        s_copy,
        self.next_seed_id,
        self.W_out.copy(),
        self.out_mask.copy(),
        self.b_out.copy(),
    )

  def restore_state(self, state):
    u_copy, s_copy, next_sid, w_out, o_mask, b_out = state
    self.units = []
    for u in u_copy:
      d = {
          "type": u["type"],
          "seed_id": u["seed_id"],
          "level": u["level"],
          "sub_axes": list(u["sub_axes"]),
          "center": u["center"].copy(),
      }
      if u["type"] == "local_dome":
        d["sigma"] = u["sigma"]
        d["bridge_indices"] = list(u["bridge_indices"])
        d["bridge_weights"] = u["bridge_weights"].copy()
        d["bridge_mask"] = u["bridge_mask"].copy()
      else:
        d["input_indices"] = list(u["input_indices"])
        d["weights"] = u["weights"].copy()
        d["mask"] = u["mask"].copy()
        d["bias"] = u["bias"].copy()
      self.units.append(d)
    self.seeds = {
        k: {
            "center_sub": v["center_sub"].copy(),
            "sub_axes": list(v["sub_axes"]),
            "unit_indices": list(v["unit_indices"]),
            "level": v["level"],
        }
        for k, v in s_copy.items()
    }
    self.next_seed_id = next_sid
    self.W_out = w_out.copy()
    self.out_mask = o_mask.copy()
    self.b_out = b_out.copy()

  def fit(self, X, y, epochs_per_stage=95, patience=8):
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.18, random_state=self.seed
    )
    tr_idx, val_idx = next(sss.split(X, y))
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    self.active_axes = detect_informative_axes(
        X_tr, y_tr, max_axes=self.max_axes
    )
    X_tr_sub = X_tr[:, self.active_axes]
    X_val_sub = X_val[:, self.active_axes]
    y_tr_oh = np.eye(self.output_dim)[y_tr]
    y_val_oh = np.eye(self.output_dim)[y_val]

    self._init_graph(len(self.active_axes))

    for _ in range(epochs_per_stage):
      self.train_epoch(X_tr_sub, y_tr_oh)

    probs_val = self.forward_sub(X_val_sub)
    best_acc = np.mean(np.argmax(probs_val, axis=1) == y_val)
    best_loss = cross_entropy(y_val_oh, probs_val)
    best_state = self.save_state()

    tabu_actions = set()
    stall = 0

    while len(self.units) < self.max_units:
      if best_acc >= 0.990 or stall >= patience:
        break

      global_pool = self.build_global_candidates(X_tr_sub, y_tr, tabu_actions)
      if not global_pool:
        break

      best_cand = global_pool[0]
      checkpoint = self.save_state()
      new_unit_idx = len(self.units)

      self.apply_growth(
          X_tr_sub,
          y_tr,
          best_cand["action"],
          best_cand["seed_id"],
          best_cand["local_idx"],
          best_cand["local_gaps"],
          best_cand["sub_axes"],
      )

      self.lr = self.base_lr
      for _ in range(epochs_per_stage):
        self.train_epoch(X_tr_sub, y_tr_oh, frozen_before_idx=new_unit_idx)

      self.prune_weak_connections(X_tr_sub, rel_threshold=0.16)
      self.lr = self.base_lr * 0.3
      for _ in range(epochs_per_stage // 3):
        self.train_epoch(X_tr_sub, y_tr_oh, frozen_before_idx=0)

      probs_val = self.forward_sub(X_val_sub)
      acc = np.mean(np.argmax(probs_val, axis=1) == y_val)
      loss = cross_entropy(y_val_oh, probs_val)

      improved = (acc > best_acc + 0.002) or (
          loss < best_loss - 0.004 and acc >= best_acc - 0.003
      )
      if improved:
        best_acc = max(best_acc, acc)
        best_loss = min(best_loss, loss)
        best_state = self.save_state()
        stall = 0
      else:
        self.restore_state(checkpoint)
        tabu_actions.add(best_cand["action_key"])
        stall += 1

    self.restore_state(best_state)
    self.optimize_inter_tower_bridges(
        X_tr_sub, y_tr_oh, X_val_sub, y_val, y_val_oh
    )

    self.pre_prune_stats = self.get_topology_stats()
    self.prune_weak_connections(X_tr_sub, rel_threshold=0.16)
    self.prune_redundant_outputs(
        X_tr_sub, X_val_sub, y_val, y_val_oh, loss_tol=0.002
    )
    return self

  def predict(self, X):
    probs = self.forward_sub(X[:, self.active_axes])
    return np.argmax(probs, axis=1)

  def get_topology_stats(self):
    internal = 0
    bridges = 0
    for u in self.units:
      if u["type"] == "local_dome":
        br = int(np.sum(u["bridge_mask"]))
        internal += len(u["sub_axes"]) + br
        bridges += br
      else:
        internal += int(np.sum(u["mask"]))
        own_sid = u["seed_id"]
        own_cols = {
            self.base_dim + idx for idx in self.seeds[own_sid]["unit_indices"]
        }
        for pos, inp_col in enumerate(u["input_indices"]):
          if (
              inp_col >= self.base_dim
              and inp_col not in own_cols
              and u["mask"][pos, 0] > 0
          ):
            bridges += 1

    total_links = internal + int(np.sum(self.out_mask))
    total_biases = len(self.units) + self.output_dim
    param_count = total_links + total_biases
    return {
        "active_links": total_links,
        "cross_tower_bridges": bridges,
        "towers": len(self.seeds),
        "units": len(self.units),
        "model_bytes": param_count * 8,
    }


# ==============================================================================
# 4. REFERENCE BASELINES
# ==============================================================================
class CascadeCorrelationBaseline:

  def __init__(
      self, input_dim, output_dim, max_hidden=18, lr=0.04, pool_size=6, seed=42
  ):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.max_hidden = max_hidden
    self.lr = lr
    self.pool_size = pool_size
    self.rng = np.random.RandomState(seed)

  def _train_output(self, H, y_oh, epochs=110):
    m, dim = H.shape
    if not hasattr(self, "W_out") or self.W_out.shape[0] != dim:
      W_new = self.rng.randn(dim, self.output_dim) * 0.05
      if hasattr(self, "W_out"):
        W_new[: self.W_out.shape[0], :] = self.W_out
      self.W_out = W_new
      if not hasattr(self, "b_out"):
        self.b_out = np.zeros((1, self.output_dim))

    for _ in range(epochs):
      probs = softmax(H @ self.W_out + self.b_out)
      dz = (probs - y_oh) / m
      self.W_out -= self.lr * (H.T @ dz + 1e-4 * self.W_out)
      self.b_out -= self.lr * np.mean(dz, axis=0, keepdims=True)

  def fit(self, X, y):
    y_oh = np.eye(self.output_dim)[y]
    self.hidden_weights = []
    self.hidden_biases = []
    H = X.copy()
    self._train_output(H, y_oh, epochs=140)

    for _ in range(self.max_hidden):
      probs = softmax(H @ self.W_out + self.b_out)
      err = y_oh - probs
      err_centered = err - np.mean(err, axis=0, keepdims=True)
      if np.mean(np.abs(err)) < 0.045:
        break

      best_corr = -1.0
      best_w, best_b = None, None
      dim = H.shape[1]

      for _ in range(self.pool_size):
        w = self.rng.randn(dim, 1) * np.sqrt(2.0 / dim)
        b = np.zeros((1, 1))
        for _ in range(55):
          v = np.tanh(H @ w + b)
          v_cent = v - np.mean(v)
          cov = np.sum(v_cent * err_centered, axis=0, keepdims=True)
          signs = np.sign(cov)
          dv = np.sum(err_centered * signs, axis=1, keepdims=True)
          dz = dv * (1.0 - v**2)
          w += (self.lr * 0.5 / len(H)) * (H.T @ dz)
          b += (self.lr * 0.5 / len(H)) * np.sum(dz)

        v = np.tanh(H @ w + b)
        v_cent = v - np.mean(v)
        score = float(np.sum(np.abs(np.sum(v_cent * err_centered, axis=0))))
        if score > best_corr:
          best_corr = score
          best_w, best_b = w, b

      self.hidden_weights.append(best_w)
      self.hidden_biases.append(best_b)
      v_new = np.tanh(H @ best_w + best_b)
      H = np.hstack([H, v_new])
      self._train_output(H, y_oh, epochs=90)
    return self

  def predict(self, X):
    H = X.copy()
    for w, b in zip(self.hidden_weights, self.hidden_biases):
      H = np.hstack([H, np.tanh(H @ w + b)])
    return np.argmax(softmax(H @ self.W_out + self.b_out), axis=1)

  def get_topology_stats(self):
    active = sum(w.size for w in self.hidden_weights) + self.W_out.size
    biases = len(self.hidden_biases) + self.output_dim
    param_count = active + biases
    return {
        "active_links": active,
        "cross_tower_bridges": 0,
        "model_bytes": param_count * 8,
    }


class BaselineMLP32:

  def __init__(self, input_dim, output_dim, hidden_dim=32, lr=0.04, seed=42):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.lr = lr
    self.rng = np.random.RandomState(seed)

    self.W1 = self.rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    self.b1 = np.zeros((1, hidden_dim))
    self.m1 = np.ones_like(self.W1)

    self.W2 = self.rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    self.b2 = np.zeros((1, output_dim))
    self.m2 = np.ones_like(self.W2)

  def _forward(self, X):
    self.z1 = X @ (self.W1 * self.m1) + self.b1
    self.h1 = relu(self.z1)
    self.z2 = self.h1 @ (self.W2 * self.m2) + self.b2
    self.probs = softmax(self.z2)
    return self.probs

  def fit(self, X, y, epochs=240, batch_size=32, l2_reg=1e-4):
    y_oh = np.eye(self.output_dim)[y]
    n = len(X)
    for _ in range(epochs):
      idx = self.rng.permutation(n)
      for s in range(0, n, batch_size):
        b = idx[s : s + batch_size]
        Xb, yb = X[b], y_oh[b]
        m = len(Xb)
        probs = self._forward(Xb)
        dz2 = (probs - yb) / m
        dW2 = self.h1.T @ dz2 + l2_reg * (self.W2 * self.m2)
        db2 = np.mean(dz2, axis=0, keepdims=True)
        dh1 = dz2 @ (self.W2 * self.m2).T
        dz1 = dh1 * relu_grad(self.z1)
        dW1 = Xb.T @ dz1 + l2_reg * (self.W1 * self.m1)
        db1 = np.mean(dz1, axis=0, keepdims=True)

        self.W2 -= self.lr * (dW2 * self.m2)
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * (dW1 * self.m1)
        self.b1 -= self.lr * db1
    return self

  def prune_to_target_links(self, X, y, target_links, finetune_epochs=60):
    all_w = np.concatenate([np.abs(self.W1).ravel(), np.abs(self.W2).ravel()])
    k = int(np.clip(target_links, 2, len(all_w)))
    thresh = np.partition(all_w, -k)[-k]
    self.m1 = (np.abs(self.W1) >= thresh).astype(float)
    self.m2 = (np.abs(self.W2) >= thresh).astype(float)
    total_now = int(np.sum(self.m1) + np.sum(self.m2))
    if total_now > k:
      ones_1 = np.argwhere(self.m1 > 0)
      for r, c in ones_1[: total_now - k]:
        self.m1[r, c] = 0.0
    self.W1 *= self.m1
    self.W2 *= self.m2
    self.fit(X, y, epochs=finetune_epochs)

  def predict(self, X):
    return np.argmax(self._forward(X), axis=1)

  def get_topology_stats(self):
    active = int(np.sum(self.m1) + np.sum(self.m2))
    biases = self.hidden_dim + self.output_dim
    param_count = active + biases
    return {
        "active_links": active,
        "cross_tower_bridges": 0,
        "model_bytes": param_count * 8,
    }


# ==============================================================================
# 5. EXPERIMENT 1: MAX_AXES SENSITIVITY (INCLUDING 48, 56, 64)
# ==============================================================================
def run_max_axes_sensitivity():
  print("\n" + "=" * 90)
  print("MAX_AXES SENSITIVITY (UCI DIGITS)")
  print("=" * 90)

  X = load_digits().data
  y = load_digits().target

  axes_grid = [8, 16, 24, 32, 40, 48, 56, 64]

  for max_axes in axes_grid:
    acc_all = []
    f1_all = []
    links_all = []

    splitter = StratifiedShuffleSplit(
        n_splits=5, test_size=0.20, random_state=42
    )

    for seed, (tr, te) in enumerate(splitter.split(X, y)):
      scaler = StandardScaler()
      X_tr = scaler.fit_transform(X[tr])
      X_te = scaler.transform(X[te])
      y_tr = y[tr]
      y_te = y[te]

      model = OptimalCrossTowerAGNN(
          output_dim=len(np.unique(y)),
          max_axes=max_axes,
          max_units=45,
          learning_rate=0.05,
          conn_selector="hybrid_golden",
          residual_mode="prob_gap",
          seed=seed,
      )
      model.fit(X_tr, y_tr)
      preds = model.predict(X_te)
      stats = model.get_topology_stats()

      acc_all.append(accuracy_score(y_te, preds))
      f1_all.append(f1_score(y_te, preds, average="macro"))
      links_all.append(stats["active_links"])

    print(
        f"max_axes={max_axes:2d} | "
        f"Acc={np.mean(acc_all):.4f} ± {np.std(acc_all):.4f} | "
        f"F1={np.mean(f1_all):.4f} | "
        f"Links={np.mean(links_all):.1f}",
        flush=True,
    )

  print("=" * 90)


# ==============================================================================
# 6. FULL Q2 BENCHMARK (UCI DIGITS UPDATED TO max_axes = 48)
# ==============================================================================
def format_stat_sig(agnn_arr, base_arr):
  diff = agnn_arr - base_arr
  wins = int(np.sum(diff > 1e-12))
  ties = int(np.sum(np.abs(diff) <= 1e-12))
  losses = int(np.sum(diff < -1e-12))
  if np.allclose(diff, 0):
    p_w = 1.0
  else:
    try:
      _, p_w = wilcoxon(diff)
    except ValueError:
      p_w = 1.0
  sig_mark = (
      "$^{***}$"
      if p_w < 0.001
      else ("$^{**}$" if p_w < 0.01 else ("$^{*}$" if p_w < 0.05 else ""))
  )
  return f"{wins}/{ties}/{losses}", f"{p_w:.4f}{sig_mark}"


def run_full_q2_benchmark(only_digits=False):
  all_datasets = {
      "Two Spirals (32D)": (*make_two_spirals(seed=42), 20, 0.04),
      "Iris (4D)": (load_iris().data, load_iris().target, 4, 0.04),
      "Breast Cancer (30D)": (
          load_breast_cancer().data,
          load_breast_cancer().target,
          18,
          0.04,
      ),
      "Wine (13D)": (load_wine().data, load_wine().target, 13, 0.04),
      "UCI Digits (64D)": (
          load_digits().data,
          load_digits().target,
          48,  # Updated from 24 to 48
          0.05,
      ),
  }

  datasets = (
      {"UCI Digits (64D)": all_datasets["UCI Digits (64D)"]}
      if only_digits
      else all_datasets
  )

  models_order = ["AGNN", "CasCor", "MLP32", "MagPrune"]
  models_latex = {
      "AGNN": r"\textbf{OptimalCrossTowerAGNN (Ours)}",
      "CasCor": r"Cascade-Correlation \cite{fahlman1990}",
      "MLP32": r"Baseline MLP-32",
      "MagPrune": r"MagnitudePruningMLP (Iso-Links)",
  }

  all_results = {}
  pre_post_prune = {}
  ablation_results = {}

  for ds_name, (X, y, max_ax, lr) in datasets.items():
    out_dim = len(np.unique(y))
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=0.2, random_state=42
    )

    ds_res = {
        m: {
            "acc": [],
            "f1": [],
            "links": [],
            "bridges": [],
            "bytes": [],
            "train_s": [],
            "inf_ms": [],
        }
        for m in models_order
    }
    pre_post_prune[ds_name] = {
        "pre_links": [],
        "post_links": [],
        "test_acc": [],
    }

    run_ablation = ds_name in ("Two Spirals (32D)", "UCI Digits (64D)")
    if run_ablation:
      ablation_results[ds_name] = {
          "OMP": {"acc": [], "f1": [], "links": []},
          "Margin": {"acc": [], "f1": [], "links": []},
      }

    for seed, (tr, te) in enumerate(sss.split(X, y)):
      sc = StandardScaler()
      X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
      y_tr, y_te = y[tr], y[te]

      # 1. OptimalCrossTowerAGNN (v2.3)
      t0 = time.perf_counter()
      agnn = OptimalCrossTowerAGNN(
          output_dim=out_dim,
          max_axes=max_ax,
          max_units=45,
          learning_rate=lr,
          max_bridges_per_unit=4,
          conn_selector="hybrid_golden",
          residual_mode="prob_gap",
          seed=seed,
      )
      agnn.fit(X_tr, y_tr)
      t_train = time.perf_counter() - t0

      t_i0 = time.perf_counter()
      p_agnn = agnn.predict(X_te)
      t_inf = (time.perf_counter() - t_i0) * 1000.0
      st_agnn = agnn.get_topology_stats()

      ds_res["AGNN"]["acc"].append(accuracy_score(y_te, p_agnn))
      ds_res["AGNN"]["f1"].append(f1_score(y_te, p_agnn, average="macro"))
      ds_res["AGNN"]["links"].append(st_agnn["active_links"])
      ds_res["AGNN"]["bridges"].append(st_agnn["cross_tower_bridges"])
      ds_res["AGNN"]["bytes"].append(st_agnn["model_bytes"])
      ds_res["AGNN"]["train_s"].append(t_train)
      ds_res["AGNN"]["inf_ms"].append(t_inf)

      pre_post_prune[ds_name]["pre_links"].append(
          agnn.pre_prune_stats["active_links"]
      )
      pre_post_prune[ds_name]["post_links"].append(st_agnn["active_links"])
      pre_post_prune[ds_name]["test_acc"].append(accuracy_score(y_te, p_agnn))

      if run_ablation:
        agnn_omp = OptimalCrossTowerAGNN(
            output_dim=out_dim,
            max_axes=max_ax,
            max_units=45,
            learning_rate=lr,
            max_bridges_per_unit=4,
            conn_selector="omp",
            residual_mode="prob_gap",
            seed=seed,
        ).fit(X_tr, y_tr)
        p_omp = agnn_omp.predict(X_te)
        ablation_results[ds_name]["OMP"]["acc"].append(
            accuracy_score(y_te, p_omp)
        )
        ablation_results[ds_name]["OMP"]["f1"].append(
            f1_score(y_te, p_omp, average="macro")
        )
        ablation_results[ds_name]["OMP"]["links"].append(
            agnn_omp.get_topology_stats()["active_links"]
        )

        if ds_name == "UCI Digits (64D)":
          agnn_mrg = OptimalCrossTowerAGNN(
              output_dim=out_dim,
              max_axes=max_ax,
              max_units=45,
              learning_rate=lr,
              max_bridges_per_unit=4,
              conn_selector="hybrid_golden",
              residual_mode="margin",
              seed=seed,
          ).fit(X_tr, y_tr)
          p_mrg = agnn_mrg.predict(X_te)
          ablation_results[ds_name]["Margin"]["acc"].append(
              accuracy_score(y_te, p_mrg)
          )
          ablation_results[ds_name]["Margin"]["f1"].append(
              f1_score(y_te, p_mrg, average="macro")
          )
          ablation_results[ds_name]["Margin"]["links"].append(
              agnn_mrg.get_topology_stats()["active_links"]
          )

      # 2. Cascade-Correlation
      t0 = time.perf_counter()
      cascor = CascadeCorrelationBaseline(
          input_dim=X_tr.shape[1],
          output_dim=out_dim,
          max_hidden=18,
          lr=lr,
          seed=seed,
      )
      cascor.fit(X_tr, y_tr)
      t_train = time.perf_counter() - t0

      t_i0 = time.perf_counter()
      p_cc = cascor.predict(X_te)
      t_inf = (time.perf_counter() - t_i0) * 1000.0
      st_cc = cascor.get_topology_stats()

      ds_res["CasCor"]["acc"].append(accuracy_score(y_te, p_cc))
      ds_res["CasCor"]["f1"].append(f1_score(y_te, p_cc, average="macro"))
      ds_res["CasCor"]["links"].append(st_cc["active_links"])
      ds_res["CasCor"]["bridges"].append(0)
      ds_res["CasCor"]["bytes"].append(st_cc["model_bytes"])
      ds_res["CasCor"]["train_s"].append(t_train)
      ds_res["CasCor"]["inf_ms"].append(t_inf)

      # 3. Baseline MLP-32
      t0 = time.perf_counter()
      mlp = BaselineMLP32(
          input_dim=X_tr.shape[1],
          output_dim=out_dim,
          hidden_dim=32,
          lr=lr,
          seed=seed,
      )
      mlp.fit(X_tr, y_tr)
      t_train = time.perf_counter() - t0

      t_i0 = time.perf_counter()
      p_mlp = mlp.predict(X_te)
      t_inf = (time.perf_counter() - t_i0) * 1000.0
      st_mlp = mlp.get_topology_stats()

      ds_res["MLP32"]["acc"].append(accuracy_score(y_te, p_mlp))
      ds_res["MLP32"]["f1"].append(f1_score(y_te, p_mlp, average="macro"))
      ds_res["MLP32"]["links"].append(st_mlp["active_links"])
      ds_res["MLP32"]["bridges"].append(0)
      ds_res["MLP32"]["bytes"].append(st_mlp["model_bytes"])
      ds_res["MLP32"]["train_s"].append(t_train)
      ds_res["MLP32"]["inf_ms"].append(t_inf)

      # 4. MagnitudePruningMLP
      t0 = time.perf_counter()
      mlp_pr = BaselineMLP32(
          input_dim=X_tr.shape[1],
          output_dim=out_dim,
          hidden_dim=32,
          lr=lr,
          seed=seed,
      )
      mlp_pr.fit(X_tr, y_tr)
      mlp_pr.prune_to_target_links(
          X_tr, y_tr, target_links=st_agnn["active_links"]
      )
      t_train = time.perf_counter() - t0

      t_i0 = time.perf_counter()
      p_pr = mlp_pr.predict(X_te)
      t_inf = (time.perf_counter() - t_i0) * 1000.0
      st_pr = mlp_pr.get_topology_stats()

      ds_res["MagPrune"]["acc"].append(accuracy_score(y_te, p_pr))
      ds_res["MagPrune"]["f1"].append(f1_score(y_te, p_pr, average="macro"))
      ds_res["MagPrune"]["links"].append(st_pr["active_links"])
      ds_res["MagPrune"]["bridges"].append(0)
      ds_res["MagPrune"]["bytes"].append(st_pr["model_bytes"])
      ds_res["MagPrune"]["train_s"].append(t_train)
      ds_res["MagPrune"]["inf_ms"].append(t_inf)

    all_results[ds_name] = ds_res
    print(f"[OK] Completed dataset: {ds_name} ({N_SPLITS} splits)", flush=True)

  # ==========================================================================
  # PRINT LATEX ROWS FOR TABLES 1-4
  # ==========================================================================
  print("\n" + "%" * 90)
  print("% TABLE 1 ROWS (PREDICTIVE PERFORMANCE & WILCOXON SIGNIFICANCE)")
  print("%" * 90)
  for ds_name, ds_res in all_results.items():
    agnn_acc = np.array(ds_res["AGNN"]["acc"])
    for idx_m, m in enumerate(models_order):
      ds_col = f"\\multirow{{4}}{{*}}{{\\textbf{{{ds_name}}}}}" if idx_m == 0 else ""
      acc_str = (
          f"${np.mean(ds_res[m]['acc']):.4f} \\pm {np.std(ds_res[m]['acc']):.4f}$"
      )
      f1_str = (
          f"${np.mean(ds_res[m]['f1']):.4f} \\pm {np.std(ds_res[m]['f1']):.4f}$"
      )
      if m == "AGNN":
        acc_str = r"\textbf{" + acc_str + "}"
        wtl_str, p_str = "---", "---"
      else:
        wtl_str, p_str = format_stat_sig(agnn_acc, np.array(ds_res[m]["acc"]))
      print(
          f"{ds_col} & {models_latex[m]} & {acc_str} & {f1_str} & {wtl_str} &"
          f" {p_str} \\\\"
      )
    print(r"\midrule")

  print("\n" + "%" * 90)
  print("% TABLE 2 ROWS (ACTIVE CONNECTIONS, BRIDGES, BYTES & LATENCY)")
  print("%" * 90)
  for ds_name, ds_res in all_results.items():
    for idx_m, m in enumerate(models_order):
      ds_col = f"\\multirow{{4}}{{*}}{{\\textbf{{{ds_name}}}}}" if idx_m == 0 else ""
      links = np.mean(ds_res[m]["links"])
      br = np.mean(ds_res[m]["bridges"])
      b_sz = np.mean(ds_res[m]["bytes"])
      tr_s = np.mean(ds_res[m]["train_s"])
      inf_m = np.mean(ds_res[m]["inf_ms"])
      print(
          f"{ds_col} & {models_latex[m]} & {links:.1f} & {br:.1f} & {b_sz:.0f}"
          f" & {tr_s:.2f} & {inf_m:.3f} \\\\"
      )
    print(r"\midrule")

  print("\n" + "%" * 90)
  print("% TABLE 3 ROWS (PRE-PRUNING VS POST-PRUNING)")
  print("%" * 90)
  for ds_name, ds_res in all_results.items():
    dense_l = np.mean(ds_res["MLP32"]["links"])
    pre_l = np.mean(pre_post_prune[ds_name]["pre_links"])
    post_l = np.mean(pre_post_prune[ds_name]["post_links"])
    comp = dense_l / max(1.0, post_l)
    print(
        f"\\textbf{{{ds_name}}} & {dense_l:.0f} & {pre_l:.1f} & {post_l:.1f} &"
        f" $\\mathbf{{{comp:.1f}\\times}}$ \\\\"
    )

  print("\n" + "%" * 90)
  print("% TABLE 4 ROWS (ABLATION STUDY)")
  print("%" * 90)
  for ds_name in all_results.keys():
    if ds_name not in ablation_results:
      continue
    base_acc = np.array(all_results[ds_name]["AGNN"]["acc"])
    base_f1 = np.array(all_results[ds_name]["AGNN"]["f1"])
    base_links = np.mean(all_results[ds_name]["AGNN"]["links"])
    n_rows = 3 if ds_name == "UCI Digits (64D)" else 2
    ds_col = f"\\multirow{{{n_rows}}}{{*}}{{\\textbf{{{ds_name}}}}}"

    print(
        f"{ds_col} & \\textbf{{Hybrid Golden + Prob. Gap (Ours)}} &"
        f" \\textbf{{${np.mean(base_acc):.4f} \\pm {np.std(base_acc):.4f}$}} &"
        f" ${np.mean(base_f1):.4f} \\pm {np.std(base_f1):.4f}$ &"
        f" {base_links:.1f} & --- & --- \\\\"
    )

    omp_acc = np.array(ablation_results[ds_name]["OMP"]["acc"])
    omp_f1 = np.array(ablation_results[ds_name]["OMP"]["f1"])
    omp_links = np.mean(ablation_results[ds_name]["OMP"]["links"])
    wtl_omp, p_omp = format_stat_sig(base_acc, omp_acc)
    print(
        f" & Orthogonal Matching Pursuit (OMP) & ${np.mean(omp_acc):.4f}"
        f" \\pm {np.std(omp_acc):.4f}$ & ${np.mean(omp_f1):.4f} \\pm"
        f" {np.std(omp_f1):.4f}$ & {omp_links:.1f} & {wtl_omp} & {p_omp} \\\\"
    )

    if ds_name == "UCI Digits (64D)":
      mrg_acc = np.array(ablation_results[ds_name]["Margin"]["acc"])
      mrg_f1 = np.array(ablation_results[ds_name]["Margin"]["f1"])
      mrg_links = np.mean(ablation_results[ds_name]["Margin"]["links"])
      wtl_mrg, p_mrg = format_stat_sig(base_acc, mrg_acc)
      print(
          f" & Hybrid Golden + Margin Residual & ${np.mean(mrg_acc):.4f} \\pm"
          f" {np.std(mrg_acc):.4f}$ & ${np.mean(mrg_f1):.4f} \\pm"
          f" {np.std(mrg_f1):.4f}$ & {mrg_links:.1f} & {wtl_mrg} & {p_mrg} \\\\"
      )
    print(r"\midrule")


if __name__ == "__main__":
  # 1. Чувствительность по сетке [8, 16, 24, 32, 40, 48, 56, 64]
  run_max_axes_sensitivity()

  # 2. Быстрый перезапуск основного бенчмарка только для UCI Digits с max_axes=48
  # (установите only_digits=False, если нужно пересчитать все 5 датасетов сразу)
  run_full_q2_benchmark(only_digits=True)
