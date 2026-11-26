import time
import warnings
import numpy as np
from scipy.stats import wilcoxon
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

N_SPLITS = 15


# ==============================================================================
# 1. CORE PRIMITIVES & DATA GENERATORS
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
# 2. OptimalCrossTowerAGNN WITH BRIDGE ABLATION MODES ("gradient" | "none" | "random")
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
      bridge_mode="gradient",  # "gradient" (Ours) | "none" (Isolated) | "random"
      seed=42,
  ):
    self.output_dim = output_dim
    self.max_axes = max_axes
    self.max_units = max_units
    self.base_lr = learning_rate
    self.lr = learning_rate
    self.sparsity_penalty = sparsity_penalty
    self.max_bridges_per_unit = max_bridges_per_unit
    self.bridge_mode = bridge_mode
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
    gaps = 1.0 - probs[np.arange(len(y_idx)), y_idx]
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

  def select_connections_golden(
      self, candidate_cols, local_idx, local_gaps, y_idx
  ):
    M = len(candidate_cols)
    if M <= 2:
      return candidate_cols

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
      if self.bridge_mode != "none":
        same_3d = [
            self.base_dim + i
            for i, u in enumerate(self.units)
            if u["seed_id"] != sid and u["level"] == 3
        ]
        if len(same_3d) > 0:
          if self.bridge_mode == "random":
            n_pick = min(2, len(same_3d))
            peer_indices = self.rng.choice(
                same_3d, size=n_pick, replace=False
            ).tolist()
          else:
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
      if self.bridge_mode == "none":
        same_lvl_peers = []
      else:
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
    if self.bridge_mode == "none" or len(self.seeds) < 2:
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

    if self.bridge_mode == "random":
      perm = self.rng.permutation(len(bridge_candidates))
      bridge_candidates = [bridge_candidates[i] for i in perm]
    else:
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

          if self.bridge_mode == "random":
            init_w = float(self.rng.uniform(-0.25, 0.25))
          else:
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
    return {
        "active_links": total_links,
        "cross_tower_bridges": bridges,
        "towers": len(self.seeds),
        "units": len(self.units),
        "model_bytes": (total_links + total_biases) * 8,
    }


# ==============================================================================
# 3. BRIDGE ABLATION RUNNER
# ==============================================================================
def format_stat_sig(base_arr, var_arr):
  diff = base_arr - var_arr
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


def run_bridge_ablation():
  datasets = {
      "Two Spirals (32D)": (*make_two_spirals(seed=42), 20, 0.04),
      "UCI Digits (64D)": (load_digits().data, load_digits().target, 48, 0.05),
  }

  modes = [
      ("gradient", r"\textbf{Gradient-Guided Bridges (Ours)}"),
      ("none", r"No Inter-Tower Bridges (Isolated Towers)"),
      ("random", r"Random Acyclic Bridges"),
  ]

  print("=" * 90)
  print("INTER-TOWER BRIDGE ABLATION STUDY (15 STRATIFIED SPLITS)")
  print("=" * 90)

  results = {}
  for ds_name, (X, y, max_ax, lr) in datasets.items():
    out_dim = len(np.unique(y))
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=0.20, random_state=42
    )
    results[ds_name] = {
        m[0]: {"acc": [], "f1": [], "links": [], "bridges": [], "bytes": []}
        for m in modes
    }

    for seed, (tr, te) in enumerate(sss.split(X, y)):
      sc = StandardScaler()
      X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
      y_tr, y_te = y[tr], y[te]

      for mode_key, _ in modes:
        model = OptimalCrossTowerAGNN(
            output_dim=out_dim,
            max_axes=max_ax,
            max_units=45,
            learning_rate=lr,
            max_bridges_per_unit=4,
            bridge_mode=mode_key,
            seed=seed,
        ).fit(X_tr, y_tr)

        preds = model.predict(X_te)
        st = model.get_topology_stats()
        results[ds_name][mode_key]["acc"].append(accuracy_score(y_te, preds))
        results[ds_name][mode_key]["f1"].append(
            f1_score(y_te, preds, average="macro")
        )
        results[ds_name][mode_key]["links"].append(st["active_links"])
        results[ds_name][mode_key]["bridges"].append(st["cross_tower_bridges"])
        results[ds_name][mode_key]["bytes"].append(st["model_bytes"])

    print(f"\n--- {ds_name} ---")
    for mode_key, _ in modes:
      r = results[ds_name][mode_key]
      print(
          f"{mode_key:10s} | Acc={np.mean(r['acc']):.4f} ±"
          f" {np.std(r['acc']):.4f} | F1={np.mean(r['f1']):.4f} |"
          f" Links={np.mean(r['links']):5.1f} |"
          f" Bridges={np.mean(r['bridges']):4.1f} | Bytes={np.mean(r['bytes']):.0f}",
          flush=True,
      )

  # Print LaTeX Table
  print("\n" + "%" * 90)
  print("% LATEX TABLE: INTER-TOWER BRIDGE ABLATION")
  print("%" * 90)
  print(r"\begin{table*}[t]")
  print(r"\centering")
  print(
      r"\caption{Ablation study isolating the contribution of gradient-guided"
      r" inter-tower bridge synthesis across 15 stratified splits"
      r" ($\text{Mean} \pm \text{Std}$).}"
  )
  print(r"\label{tab:bridge_ablation}")
  print(r"\resizebox{\textwidth}{!}{%")
  print(r"\begin{tabular}{llcccccc}")
  print(r"\toprule")
  print(
      r"\textbf{Dataset} & \textbf{Bridge Mechanism} & \textbf{Test Accuracy}"
      r" & \textbf{Macro-F1} & \textbf{Active Links} & \textbf{Bridges} &"
      r" \textbf{W / T / L} & \textbf{Wilcoxon $p$-val} \\"
  )
  print(r"\midrule")

  for ds_name in datasets.keys():
    base_acc = np.array(results[ds_name]["gradient"]["acc"])
    for idx_m, (mode_key, mode_latex) in enumerate(modes):
      ds_col = f"\\multirow{{3}}{{*}}{{\\textbf{{{ds_name}}}}}" if idx_m == 0 else ""
      r = results[ds_name][mode_key]
      acc_s = f"${np.mean(r['acc']):.4f} \\pm {np.std(r['acc']):.4f}$"
      f1_s = f"${np.mean(r['f1']):.4f} \\pm {np.std(r['f1']):.4f}$"
      if mode_key == "gradient":
        acc_s = r"\textbf{" + acc_s + "}"
        wtl_s, p_s = "---", "---"
      else:
        wtl_s, p_s = format_stat_sig(base_acc, np.array(r["acc"]))
      print(
          f"{ds_col} & {mode_latex} & {acc_s} & {f1_s} &"
          f" {np.mean(r['links']):.1f} & {np.mean(r['bridges']):.1f} &"
          f" {wtl_s} & {p_s} \\\\"
      )
    print(r"\midrule")

  print(r"\bottomrule")
  print(r"\end{tabular}}")
  print(r"\end{table*}")


if __name__ == "__main__":
  run_bridge_ablation()
