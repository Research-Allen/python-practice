import time
import warnings
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ГЕНЕРАТОРЫ ДАННЫХ
# ==============================================================================
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_oh, probs):
    p = np.clip(probs, 1e-8, 1.0)
    return -np.mean(np.sum(y_oh * np.log(p), axis=1))

def make_two_spirals(samples=1000, turns=2.0, noise=0.12, noise_dims=30, seed=42):
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

def detect_axes_universal(X, y, max_axes=22):
    n, d = X.shape
    if d <= max_axes:
        return np.arange(d)

    classes = np.unique(y)
    sub = min(n, 350)
    Xs, ys = X[:sub], y[:sub]

    single_nn = np.zeros(d)
    for i in range(d):
        dist1 = (Xs[:, i:i+1] - Xs[:, i:i+1].T) ** 2
        np.fill_diagonal(dist1, np.inf)
        single_nn[i] = np.mean(ys == ys[np.argmin(dist1, axis=1)])

    selected = set()
    if len(classes) == 2:
        for i in range(d):
            for j in range(i + 1, d):
                dist2 = (Xs[:, i:i+1] - Xs[:, i:i+1].T)**2 + (Xs[:, j:j+1] - Xs[:, j:j+1].T)**2
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
    top_nn = set(np.argsort(single_nn)[::-1][:max(4, max_axes // 3)].tolist())
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

# ==============================================================================
# 2. МЕТОДЫ ОПТИМИЗАЦИИ: GOLDEN SECTION VS MULTI-BASIN GBLR + OMP
# ==============================================================================
def golden_section_discrete_search(eval_cost_and_work_fn, low, high):
    """Классический дискретный поиск методом золотого сечения (phi = 1.618)."""
    low, high = int(low), int(high)
    if high <= low:
        return low
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    x1 = int(round(b - inv_phi * (b - a)))
    x2 = int(round(a + inv_phi * (b - a)))
    f1 = eval_cost_and_work_fn(x1)[0]
    f2 = eval_cost_and_work_fn(x2)[0]

    while (b - a) > 1:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = int(round(b - inv_phi * (b - a)))
            f1 = eval_cost_and_work_fn(x1)[0]
        else:
            a, x1, f1 = x1, x2, f2
            x2 = int(round(a + inv_phi * (b - a)))
            f2 = eval_cost_and_work_fn(x2)[0]

    return int(np.clip(x1 if f1 <= f2 else x2, low, high))

def multi_basin_global_local_search(eval_cost_and_work_fn, low, high, max_evals=8):
    """
    Многобассейновый глобально-локальный поиск (GBLR):
    1) Глобальное стратифицированное зондирование по всей сетке;
    2) Детекция всех локальных минимумов и оценка потенциала «полезной работы» каждого бассейна;
    3) Параболическое и локальное уточнение в топ-2 бассейнах и выбор глобального минимума.
    """
    low, high = int(low), int(high)
    n_pts = high - low + 1
    if n_pts <= 1:
        return low
    if n_pts <= max_evals:
        vals = [eval_cost_and_work_fn(k)[0] for k in range(low, high + 1)]
        return low + int(np.argmin(vals))

    cache = {}
    def query(k):
        k = int(np.clip(k, low, high))
        if k not in cache:
            cache[k] = eval_cost_and_work_fn(k)
        return cache[k]

    # Фаза 1: Неравномерное глобальное зондирование по всей сетке [low, high]
    n_coarse = min(5, max(4, max_evals - 3))
    u = np.linspace(0.0, 1.0, n_coarse)
    coarse_grid = np.unique(np.round(low + (high - low) * (0.65 * (u ** 1.35) + 0.35 * u)).astype(int))
    coarse_grid[0] = low
    coarse_grid[-1] = high

    J_vals, W_vals = [], []
    for k in coarse_grid:
        j_v, w_v = query(k)
        J_vals.append(j_v)
        W_vals.append(w_v)
    J_vals = np.array(J_vals)
    W_vals = np.array(W_vals)

    # Фаза 2: Поиск локальных минимумов и бассейнов с наибольшим приростом полезной работы
    basins = []
    m_c = len(coarse_grid)
    for i in range(m_c):
        left_idx = max(0, i - 1)
        right_idx = min(m_c - 1, i + 1)

        is_local_min = (J_vals[i] <= J_vals[left_idx]) and (J_vals[i] <= J_vals[right_idx])
        work_gain = W_vals[i] - (W_vals[left_idx] if i > 0 else 0.0)
        is_high_work_slope = work_gain > 0.25 * max(1e-8, np.max(W_vals))

        if is_local_min or is_high_work_slope:
            slope_bonus = max(0.0, W_vals[right_idx] - W_vals[left_idx])
            basin_priority = -J_vals[i] + 0.5 * W_vals[i] + 0.35 * slope_bonus
            basins.append((basin_priority, int(coarse_grid[left_idx]), int(coarse_grid[i]), int(coarse_grid[right_idx])))

    if not basins:
        best_i = int(np.argmin(J_vals))
        basins.append((-J_vals[best_i], int(coarse_grid[max(0, best_i - 1)]),
                       int(coarse_grid[best_i]), int(coarse_grid[min(m_c - 1, best_i + 1)])))

    basins.sort(key=lambda x: x[0], reverse=True)
    top_basins = basins[:2]

    # Фаза 3: Параболическое и локальное уточнение в топ-2 бассейнах
    for _, b_left, b_mid, b_right in top_basins:
        if len(cache) >= max_evals:
            break
        if b_right - b_left <= 1:
            continue

        if b_left < b_mid < b_right:
            f_l, f_m, f_r = query(b_left)[0], query(b_mid)[0], query(b_right)[0]
            denom = 2.0 * ((b_mid - b_left) * (f_m - f_r) - (b_mid - b_right) * (f_m - f_l))
            if abs(denom) > 1e-10:
                num = ((b_mid - b_left) ** 2) * (f_m - f_r) - ((b_mid - b_right) ** 2) * (f_m - f_l)
                k_parab = int(round(b_mid - num / denom))
                if b_left < k_parab < b_right and k_parab not in cache:
                    query(k_parab)

        for mid_cand in [(b_left + b_mid) // 2, (b_mid + b_right) // 2]:
            if len(cache) >= max_evals:
                break
            if b_left < mid_cand < b_right and mid_cand not in cache:
                query(mid_cand)

    best_k = min(cache.keys(), key=lambda k: cache[k][0])
    return int(best_k)

# ==============================================================================
# 3. АРХИТЕКТУРА OptimalCrossTowerAGNN С ПЕРЕКЛЮЧАТЕЛЕМ МЕТОДА ПОИСКА
# ==============================================================================
class OptimalCrossTowerAGNN:
    """
    search_mode='golden' -> Ранжирование по корреляции + Золотое сечение.
    search_mode='gblr'   -> Ортогональное ранжирование (OMP) + Multi-Basin Global-Local Search (GBLR).
    """
    def __init__(self, output_dim=2, max_axes=22, max_units=45, learning_rate=0.04,
                 search_mode="gblr", sparsity_penalty=0.025, seed=42):
        self.output_dim = output_dim
        self.max_axes = max_axes
        self.max_units = max_units
        self.base_lr = learning_rate
        self.lr = learning_rate
        self.search_mode = search_mode
        self.sparsity_penalty = sparsity_penalty
        self.seed = seed

    def _init_graph(self, base_dim):
        self.rng = np.random.RandomState(self.seed)
        self.base_dim = base_dim
        self.units = []
        self.seeds = {}
        self.next_seed_id = 1

        self.W_out = self.rng.randn(base_dim, self.output_dim) * np.sqrt(2.0 / base_dim)
        self.out_mask = np.ones((base_dim, self.output_dim))
        self.b_out = np.zeros((1, self.output_dim))
        self.hasher = BaseSpatialHasher(base_dim, n_planes=10, seed=self.seed)

    @property
    def total_dim(self):
        return self.base_dim + len(self.units)

    def forward_sub(self, X_sub):
        m = len(X_sub)
        self.coords = np.zeros((m, self.total_dim))
        self.coords[:, :self.base_dim] = X_sub
        self.cache = []

        for idx, u in enumerate(self.units):
            if u['type'] == 'local_dome':
                sub_axes = u['sub_axes']
                diff = X_sub[:, sub_axes] - u['center']
                sq_dist = np.sum(diff ** 2, axis=1, keepdims=True)
                dome_act = np.exp(-0.5 * sq_dist / (u['sigma'] ** 2))

                if len(u['bridge_indices']) > 0:
                    br_in = self.coords[:, u['bridge_indices']]
                    br_w = u['bridge_weights'] * u['bridge_mask']
                    lateral = np.tanh(br_in @ br_w)
                    act = dome_act * (1.0 + 0.4 * lateral)
                    self.cache.append((diff, dome_act, lateral))
                else:
                    act = dome_act
                    self.cache.append((diff, dome_act, None))

                self.coords[:, self.base_dim + idx] = act.flatten()
            else:
                in_idx = u['input_indices']
                sub_in = self.coords[:, in_idx]
                w_eff = u['weights'] * u['mask']
                z = sub_in @ w_eff + u['bias']
                self.cache.append((z, None, None))
                self.coords[:, self.base_dim + idx] = relu(z).flatten()

        w_out_eff = self.W_out * self.out_mask
        self.z_out = self.coords @ w_out_eff + self.b_out
        self.output = softmax(self.z_out)
        return self.output

    def backward(self, y_oh, frozen_before_idx=0, l2_reg=1e-4, return_unit_grads=False):
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
            d_act = d_coords[:, col:col + 1]

            if u['type'] == 'local_dome':
                diff, dome_act, lateral = self.cache[idx]
                if lateral is not None and len(u['bridge_indices']) > 0:
                    d_lat = d_act * dome_act * 0.4 * (1.0 - lateral ** 2)
                    unit_dz[idx] = d_lat
                    br_in = self.coords[:, u['bridge_indices']]
                    br_w = u['bridge_weights'] * u['bridge_mask']
                    d_coords[:, u['bridge_indices']] += d_lat @ br_w.T
                    if not return_unit_grads and idx >= frozen_before_idx:
                        dW_br = (br_in.T @ d_lat) / m + l2_reg * br_w
                        u['bridge_weights'] -= self.lr * (dW_br * u['bridge_mask'])
                    d_dome = d_act * (1.0 + 0.4 * lateral)
                else:
                    unit_dz[idx] = d_act * dome_act * 0.4
                    d_dome = d_act

                if not return_unit_grads and idx >= frozen_before_idx:
                    d_center = np.mean(d_dome * dome_act * (diff / (u['sigma'] ** 2)), axis=0)
                    sq_dist = np.sum(diff ** 2, axis=1, keepdims=True)
                    d_sigma = np.mean(d_dome * dome_act * (sq_dist / (u['sigma'] ** 3)))
                    u['center'] -= self.lr * 0.5 * d_center
                    u['sigma'] = np.clip(u['sigma'] - self.lr * 0.5 * d_sigma, 0.15, 2.0)
            else:
                z, _, _ = self.cache[idx]
                dz = d_act * relu_grad(z)
                unit_dz[idx] = dz
                in_idx = u['input_indices']
                sub_in = self.coords[:, in_idx]
                w_eff = u['weights'] * u['mask']
                d_coords[:, in_idx] += dz @ w_eff.T

                if not return_unit_grads and idx >= frozen_before_idx:
                    dW = (sub_in.T @ dz) / m + l2_reg * w_eff
                    db = np.mean(dz, axis=0, keepdims=True)
                    u['weights'] -= self.lr * (dW * u['mask'])
                    u['bias'] -= self.lr * db

        if return_unit_grads:
            return unit_dz

    def train_epoch(self, X_sub, y_oh, frozen_before_idx=0, batch_size=32):
        idx = self.rng.permutation(len(X_sub))
        for start in range(0, len(X_sub), batch_size):
            batch = idx[start:start + batch_size]
            self.forward_sub(X_sub[batch])
            self.backward(y_oh[batch], frozen_before_idx=frozen_before_idx)

    def build_global_candidates(self, X_sub, y_idx, tabu_actions, cluster_size=18, min_seed_dist=0.42):
        probs = self.forward_sub(X_sub)
        gaps = 1.0 - probs[np.arange(len(y_idx)), y_idx]
        hashes = self.hasher.hash_points(X_sub)
        global_pool = []

        for sid, sdata in self.seeds.items():
            if sdata['level'] >= 5:
                continue
            next_lvl = sdata['level'] + 1
            action_key = ("DEEPEN", sid, next_lvl)
            if action_key in tabu_actions:
                continue

            sub_axes = sdata['sub_axes']
            dists = np.linalg.norm(X_sub[:, sub_axes] - sdata['center_sub'], axis=1)
            local_idx = np.argsort(dists)[:cluster_size]
            local_gaps = gaps[local_idx]
            err_count = np.sum(local_gaps > 0.35)

            if err_count >= 3:
                base_score = float(np.mean(local_gaps)) * np.sqrt(err_count)
                center_full = np.mean(X_sub[local_idx], axis=0)
                global_pool.append({
                    'base_score': base_score, 'score': base_score,
                    'action': "DEEPEN", 'seed_id': sid,
                    'local_idx': local_idx, 'local_gaps': local_gaps,
                    'sub_axes': sub_axes, 'center_full': center_full,
                    'action_key': action_key
                })

        work_mask = gaps > 0.38
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
                    sdata['sub_axes'] == sub_axes and np.linalg.norm(epicenter_sub - sdata['center_sub']) < min_seed_dist
                    for sdata in self.seeds.values()
                )
                if too_close:
                    continue

                dists = np.linalg.norm(X_sub[:, sub_axes] - epicenter_sub, axis=1)
                local_idx = np.argsort(dists)[:cluster_size]
                local_gaps = gaps[local_idx]
                err_count = np.sum(local_gaps > 0.35)

                if err_count >= 3:
                    base_score = float(np.mean(local_gaps)) * np.sqrt(err_count)
                    center_full = np.mean(X_sub[local_idx], axis=0)
                    global_pool.append({
                        'base_score': base_score, 'score': base_score,
                        'action': "NEW_SEED", 'seed_id': None,
                        'local_idx': local_idx, 'local_gaps': local_gaps,
                        'sub_axes': sub_axes, 'center_full': center_full,
                        'action_key': action_key
                    })

        if not global_pool:
            return []

        centers = np.array([c['center_full'] for c in global_pool])
        base_scores = np.array([c['base_score'] for c in global_pool])
        if len(centers) > 1:
            pair_dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            nonzero = pair_dists[pair_dists > 0]
            sigma_zone = max(0.5, float(np.percentile(nonzero, 30))) if len(nonzero) > 0 else 1.0
        else:
            sigma_zone = 1.0

        for i, cand in enumerate(global_pool):
            pt_dists = np.linalg.norm(X_sub - cand['center_full'], axis=1)
            kernel_pts = np.exp(-0.5 * (pt_dists / sigma_zone) ** 2)
            regional_error_mass = float(np.sum(gaps * kernel_pts) / (np.sum(kernel_pts) + 1e-8))

            if len(centers) > 1:
                c_dists = np.linalg.norm(centers - cand['center_full'], axis=1)
                kernel_cands = np.exp(-0.5 * (c_dists / sigma_zone) ** 2)
                kernel_cands[i] = 0.0
                neighbor_support = float(np.sum(base_scores * kernel_cands))
            else:
                neighbor_support = 0.0

            cand['score'] = cand['base_score'] * (1.0 + 0.6 * regional_error_mass) + 0.35 * neighbor_support

        global_pool.sort(key=lambda c: c['score'], reverse=True)
        return global_pool

    # --------------------------------------------------------------------------
    # УНИВЕРСАЛЬНЫЙ ОТБОР СВЯЗЕЙ ('golden' vs 'gblr' + OMP)
    # --------------------------------------------------------------------------
    def select_connections(self, candidate_cols, local_idx, local_gaps, y_idx):
        M = len(candidate_cols)
        if M <= 2:
            return candidate_cols

        if self.search_mode == "gblr":
            # 1. Ортогональное ранжирование OMP (устраняет дублирующие связи)
            residual = local_gaps - np.mean(local_gaps)
            remaining = list(candidate_cols)
            ordered_cols = []
            Q = []
            for _ in range(M):
                if np.linalg.norm(residual) < 1e-8:
                    ordered_cols.extend(remaining)
                    break
                best_sc, best_c, best_u = -1.0, remaining[0], None
                for col in remaining:
                    v = self.coords[local_idx, col] - np.mean(self.coords[local_idx, col])
                    for q in Q:
                        v = v - np.dot(v, q) * q
                    vn = np.linalg.norm(v)
                    if vn < 1e-8:
                        sc, u = 0.0, None
                    else:
                        u = v / vn
                        sc = abs(float(np.dot(u, residual))) * (0.25 + float(np.std(self.coords[local_idx, col])))
                    if sc > best_sc:
                        best_sc, best_c, best_u = sc, col, u
                ordered_cols.append(best_c)
                remaining.remove(best_c)
                if best_u is not None:
                    Q.append(best_u)
                    residual = residual - np.dot(residual, best_u) * best_u
        else:
            # Статическое ранжирование по корреляции для Golden Section
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

        def eval_cost_and_work(k):
            k = int(np.clip(k, 2, M))
            sub_cols = ordered_cols[:k]
            sub_in = self.coords[local_idx][:, sub_cols]
            c_in = np.mean(sub_in[target_bin == 1], axis=0) if np.any(target_bin == 1) else np.mean(sub_in, axis=0)
            w = c_in.reshape(-1, 1)
            b = -0.5 * float(np.dot(c_in, c_in))
            act_proto = relu(sub_in @ w + b).flatten()
            if np.std(act_proto) < 1e-8:
                loss_k = base_err
            else:
                p_norm = (act_proto - np.min(act_proto)) / (np.max(act_proto) - np.min(act_proto) + 1e-8)
                loss_k = float(np.mean((p_norm - target_bin) ** 2))
            work_k = max(0.0, base_err - loss_k)
            j_k = loss_k + self.sparsity_penalty * (k / M)
            return j_k, work_k

        if self.search_mode == "gblr":
            k_opt = multi_basin_global_local_search(eval_cost_and_work, low=2, high=M, max_evals=8)
        else:
            k_opt = golden_section_discrete_search(eval_cost_and_work, low=2, high=M)

        return ordered_cols[:k_opt]

    def apply_growth(self, X_sub, y_idx, action, seed_id, local_idx, local_gaps, sub_axes):
        self.forward_sub(X_sub)
        center_sub = np.mean(X_sub[local_idx][:, sub_axes], axis=0)
        spread_sub = np.mean(np.std(X_sub[local_idx][:, sub_axes], axis=0))

        if action == "NEW_SEED":
            sid = self.next_seed_id
            self.next_seed_id += 1
            level = 3
            sigma = float(np.clip(spread_sub * 1.4, 0.25, 1.0))

            peer_indices = []
            same_3d = [self.base_dim + i for i, u in enumerate(self.units) if u['seed_id'] != sid and u['level'] == 3]
            if len(same_3d) > 0:
                peer_indices = self.select_connections(same_3d, local_idx, local_gaps, y_idx)[:2]

            peer_w = self.rng.randn(len(peer_indices), 1) * 0.12 if peer_indices else np.zeros((0, 1))
            peer_m = np.ones((len(peer_indices), 1)) if peer_indices else np.zeros((0, 1))

            unit = {
                'type': 'local_dome', 'seed_id': sid, 'level': level,
                'sub_axes': list(sub_axes), 'center': center_sub.copy(), 'sigma': sigma,
                'bridge_indices': peer_indices, 'bridge_weights': peer_w, 'bridge_mask': peer_m
            }
            self.seeds[sid] = {
                'center_sub': center_sub.copy(), 'sub_axes': list(sub_axes),
                'unit_indices': [len(self.units)], 'level': level
            }
        else:
            sid = seed_id
            seed = self.seeds[sid]
            own_cols = [self.base_dim + u_idx for u_idx in seed['unit_indices']]
            new_level = seed['level'] + 1
            same_lvl_peers = [
                self.base_dim + idx for idx, u in enumerate(self.units)
                if u['seed_id'] != sid and u['level'] == new_level
            ]
            all_candidate_cols = list(range(self.base_dim)) + own_cols + same_lvl_peers
            input_indices = self.select_connections(all_candidate_cols, local_idx, local_gaps, y_idx)

            sub_in = self.coords[local_idx][:, input_indices]
            center_in = np.mean(sub_in, axis=0)
            spread_in = np.std(sub_in, axis=0)

            in_dim = len(input_indices)
            new_w = (center_in + 0.08 * spread_in * self.rng.randn(in_dim)).reshape(in_dim, 1)
            new_b = (-0.5 * np.dot(center_in, new_w)).reshape(1, 1)

            seed['level'] = new_level
            seed['unit_indices'].append(len(self.units))

            unit = {
                'type': 'deep_axis', 'seed_id': sid, 'level': new_level,
                'sub_axes': list(seed['sub_axes']), 'center': center_sub.copy(),
                'input_indices': input_indices, 'weights': new_w,
                'mask': np.ones((in_dim, 1)), 'bias': new_b
            }

        self.units.append(unit)
        new_out_w = self.rng.randn(1, self.output_dim) * 0.25
        dom_class = int(np.bincount(y_idx[local_idx]).argmax())
        new_out_w[0, dom_class] += 0.55

        self.W_out = np.vstack([self.W_out, new_out_w])
        self.out_mask = np.vstack([self.out_mask, np.ones((1, self.output_dim))])

    # --------------------------------------------------------------------------
    # ОПТИМИЗАЦИЯ МЕЖБАШЕННЫХ МОСТОВ ('golden' vs 'gblr')
    # --------------------------------------------------------------------------
    def optimize_inter_tower_bridges(self, X_tr_sub, y_tr_oh, X_val_sub, y_val, y_val_oh):
        if len(self.seeds) < 2:
            return 0

        self.forward_sub(X_tr_sub)
        unit_dz = self.backward(y_tr_oh, return_unit_grads=True)

        bridge_candidates = []
        for dst_idx, u_dst in enumerate(self.units):
            dz_dst = unit_dz[dst_idx]
            existing_inputs = set(u_dst['bridge_indices'] if u_dst['type'] == 'local_dome' else u_dst['input_indices'])

            for src_idx in range(dst_idx):
                u_src = self.units[src_idx]
                if u_src['seed_id'] == u_dst['seed_id']:
                    continue
                src_col = self.base_dim + src_idx
                if src_col in existing_inputs:
                    continue

                a_src = self.coords[:, src_col:src_col + 1]
                grad_score = abs(float(np.mean(a_src * dz_dst)))
                level_bonus = 1.25 if u_src['level'] == u_dst['level'] else 1.0
                bridge_candidates.append((grad_score * level_bonus, grad_score, src_col, dst_idx))

        if not bridge_candidates:
            return 0

        bridge_candidates.sort(key=lambda x: x[0], reverse=True)
        top_bridges = bridge_candidates[:min(12, len(bridge_candidates))]
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
                    a_src = self.coords[:, src_col:src_col + 1]
                    raw_grad = float(np.mean(a_src * dz_now[dst_idx]))
                    init_w = float(np.clip(-0.35 * np.sign(raw_grad) * min(1.0, abs(raw_grad) * 10.0), -0.4, 0.4))

                    if u_dst['type'] == 'local_dome':
                        u_dst['bridge_indices'].append(src_col)
                        u_dst['bridge_weights'] = np.vstack([u_dst['bridge_weights'], [[init_w]]])
                        u_dst['bridge_mask'] = np.vstack([u_dst['bridge_mask'], [[1.0]]])
                    else:
                        u_dst['input_indices'].append(src_col)
                        u_dst['weights'] = np.vstack([u_dst['weights'], [[init_w]]])
                        u_dst['mask'] = np.vstack([u_dst['mask'], [[1.0]]])

                old_lr = self.lr
                self.lr = self.base_lr * 0.35
                for _ in range(35):
                    self.train_epoch(X_tr_sub, y_tr_oh, frozen_before_idx=0)
                self.lr = old_lr

                p_val = self.forward_sub(X_val_sub)
                acc_v = np.mean(np.argmax(p_val, axis=1) == y_val)
                loss_v = cross_entropy(y_val_oh, p_val)
                work_v = max(0.0, base_loss - loss_v)
                cost_v = loss_v + 0.015 * (k / M)
                cache_eval[k] = (cost_v, work_v, acc_v, loss_v, self.save_state())
            return cache_eval[k]

        if M == 1:
            best_k = 1
        elif self.search_mode == "gblr":
            best_k = multi_basin_global_local_search(
                lambda k: (get_eval(k)[0], get_eval(k)[1]), low=1, high=M, max_evals=7
            )
        else:
            best_k = golden_section_discrete_search(
                lambda k: (get_eval(k)[0], get_eval(k)[1]), low=1, high=M
            )

        _, _, best_acc_k, best_loss_k, best_st_k = get_eval(best_k)
        if (best_acc_k > base_acc + 0.001) or (best_loss_k < base_loss - 0.003 and best_acc_k >= base_acc):
            self.restore_state(best_st_k)
            return best_k
        else:
            self.restore_state(base_state)
            return 0

    def prune_weak_connections(self, X_sub, rel_threshold=0.16):
        self.forward_sub(X_sub)
        for u in self.units:
            if u['type'] == 'local_dome':
                if len(u['bridge_indices']) > 0:
                    br_in = self.coords[:, u['bridge_indices']]
                    mean_act = np.mean(np.abs(br_in), axis=0, keepdims=True).T
                    imp = np.abs(u['bridge_weights'] * u['bridge_mask']) * mean_act
                    active = imp[u['bridge_mask'] > 0]
                    if len(active) > 0:
                        cutoff = np.max(active) * 0.22
                        u['bridge_mask'] = (imp >= cutoff).astype(float)
                        u['bridge_weights'] *= u['bridge_mask']
                continue

            in_idx = u['input_indices']
            sub_in = self.coords[:, in_idx]
            mean_act = np.mean(np.abs(sub_in), axis=0, keepdims=True).T
            imp = np.abs(u['weights'] * u['mask']) * mean_act
            active = imp[u['mask'] > 0]
            if len(active) <= 2:
                continue
            cutoff = np.max(active) * rel_threshold
            new_mask = (imp >= cutoff).astype(float)
            if np.sum(new_mask) < 2:
                top2 = np.argsort(imp.flatten())[::-1][:2]
                new_mask = np.zeros_like(u['mask'])
                new_mask[top2, 0] = 1.0
            u['mask'] = new_mask
            u['weights'] *= new_mask

    def prune_redundant_outputs(self, X_tr_sub, X_val_sub, y_val):
        self.forward_sub(X_tr_sub)
        base_acc = np.mean(np.argmax(self.forward_sub(X_val_sub), axis=1) == y_val)
        std_coords = np.std(self.coords, axis=0, keepdims=True).T
        importance = np.abs(self.W_out * self.out_mask) * std_coords

        for flat_idx in np.argsort(importance.ravel()):
            r, c = divmod(flat_idx, self.output_dim)
            if self.out_mask[r, c] == 0:
                continue
            self.out_mask[r, c] = 0.0
            if np.mean(np.argmax(self.forward_sub(X_val_sub), axis=1) == y_val) >= base_acc:
                self.W_out[r, c] = 0.0
            else:
                self.out_mask[r, c] = 1.0

    def save_state(self):
        u_copy = []
        for u in self.units:
            d = {'type': u['type'], 'seed_id': u['seed_id'], 'level': u['level'],
                 'sub_axes': list(u['sub_axes']), 'center': u['center'].copy()}
            if u['type'] == 'local_dome':
                d['sigma'] = u['sigma']
                d['bridge_indices'] = list(u['bridge_indices'])
                d['bridge_weights'] = u['bridge_weights'].copy()
                d['bridge_mask'] = u['bridge_mask'].copy()
            else:
                d['input_indices'] = list(u['input_indices'])
                d['weights'] = u['weights'].copy()
                d['mask'] = u['mask'].copy()
                d['bias'] = u['bias'].copy()
            u_copy.append(d)
        s_copy = {
            k: {'center_sub': v['center_sub'].copy(), 'sub_axes': list(v['sub_axes']),
                'unit_indices': list(v['unit_indices']), 'level': v['level']}
            for k, v in self.seeds.items()
        }
        return (u_copy, s_copy, self.next_seed_id, self.W_out.copy(), self.out_mask.copy(), self.b_out.copy())

    def restore_state(self, state):
        u_copy, s_copy, next_sid, w_out, o_mask, b_out = state
        self.units = []
        for u in u_copy:
            d = {'type': u['type'], 'seed_id': u['seed_id'], 'level': u['level'],
                 'sub_axes': list(u['sub_axes']), 'center': u['center'].copy()}
            if u['type'] == 'local_dome':
                d['sigma'] = u['sigma']
                d['bridge_indices'] = list(u['bridge_indices'])
                d['bridge_weights'] = u['bridge_weights'].copy()
                d['bridge_mask'] = u['bridge_mask'].copy()
            else:
                d['input_indices'] = list(u['input_indices'])
                d['weights'] = u['weights'].copy()
                d['mask'] = u['mask'].copy()
                d['bias'] = u['bias'].copy()
            self.units.append(d)
        self.seeds = {
            k: {'center_sub': v['center_sub'].copy(), 'sub_axes': list(v['sub_axes']),
                'unit_indices': list(v['unit_indices']), 'level': v['level']}
            for k, v in s_copy.items()
        }
        self.next_seed_id = next_sid
        self.W_out = w_out.copy()
        self.out_mask = o_mask.copy()
        self.b_out = b_out.copy()

    def fit(self, X, y, epochs_per_stage=95, patience=8):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.18, random_state=self.seed)
        tr_idx, val_idx = next(sss.split(X, y))
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        self.active_axes = detect_axes_universal(X_tr, y_tr, max_axes=self.max_axes)
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
            if best_acc >= 0.985 or stall >= patience:
                break

            global_pool = self.build_global_candidates(X_tr_sub, y_tr, tabu_actions)
            if not global_pool:
                break

            best_cand = global_pool[0]
            checkpoint = self.save_state()
            new_unit_idx = len(self.units)

            self.apply_growth(X_tr_sub, y_tr, best_cand['action'], best_cand['seed_id'],
                              best_cand['local_idx'], best_cand['local_gaps'], best_cand['sub_axes'])

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

            improved = (acc > best_acc + 0.002) or (loss < best_loss - 0.004 and acc >= best_acc - 0.003)
            if improved:
                best_acc = max(best_acc, acc)
                best_loss = min(best_loss, loss)
                best_state = self.save_state()
                stall = 0
            else:
                self.restore_state(checkpoint)
                tabu_actions.add(best_cand['action_key'])
                stall += 1

        self.restore_state(best_state)
        self.optimize_inter_tower_bridges(X_tr_sub, y_tr_oh, X_val_sub, y_val, y_val_oh)
        self.prune_weak_connections(X_tr_sub, rel_threshold=0.16)
        self.prune_redundant_outputs(X_tr_sub, X_val_sub, y_val)

    def predict(self, X):
        probs = self.forward_sub(X[:, self.active_axes])
        return np.argmax(probs, axis=1)

    def active_connections(self):
        internal = 0
        for u in self.units:
            if u['type'] == 'local_dome':
                internal += len(u['sub_axes']) + int(np.sum(u['bridge_mask']))
            else:
                internal += int(np.sum(u['mask']))
        return internal + int(np.sum(self.out_mask))

    def inter_tower_connections(self):
        cross = 0
        for u in self.units:
            if u['type'] == 'local_dome':
                cross += int(np.sum(u['bridge_mask']))
            else:
                own_sid = u['seed_id']
                own_cols = {self.base_dim + idx for idx in self.seeds[own_sid]['unit_indices']}
                for pos, inp_col in enumerate(u['input_indices']):
                    if inp_col >= self.base_dim and inp_col not in own_cols and u['mask'][pos, 0] > 0:
                        cross += 1
        return cross

# ==============================================================================
# 4. СРАВНИТЕЛЬНЫЙ ЗАПУСК: GOLDEN SECTION VS MULTI-BASIN GBLR + OMP (15 РАЗБИЕНИЙ)
# ==============================================================================
def run_search_comparison(n_splits=15):
    benchmarks = {}
    X1, y1 = make_two_spirals(samples=1000, turns=2.0, noise=0.12, noise_dims=30, seed=42)
    benchmarks["1. Two Spirals + 30 Noise Dims (32D, 2 класса)"] = (X1, y1, 20, 0.04)

    digits = load_digits()
    benchmarks["2. UCI Digits (64D, 10 классов, Unified Softmax)"] = (digits.data, digits.target, 24, 0.05)

    for title, (X, y, max_ax, lr) in benchmarks.items():
        output_dim = len(np.unique(y))
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

        res = {
            "GOLDEN": {"acc": [], "f1": [], "conn": [], "bridges": [], "time": []},
            "GBLR":   {"acc": [], "f1": [], "conn": [], "bridges": [], "time": []},
        }

        for seed, (tr, te) in enumerate(sss.split(X, y)):
            sc = StandardScaler()
            X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
            y_tr, y_te = y[tr], y[te]

            for mode_key, search_mode in [("GOLDEN", "golden"), ("GBLR", "gblr")]:
                model = OptimalCrossTowerAGNN(
                    output_dim=output_dim, max_axes=max_ax, max_units=45,
                    learning_rate=lr, search_mode=search_mode, seed=seed
                )
                t0 = time.perf_counter()
                model.fit(X_tr, y_tr, epochs_per_stage=95, patience=8)
                res[mode_key]["time"].append(time.perf_counter() - t0)

                preds = model.predict(X_te)
                res[mode_key]["acc"].append(accuracy_score(y_te, preds))
                res[mode_key]["f1"].append(f1_score(y_te, preds, average="macro"))
                res[mode_key]["conn"].append(model.active_connections())
                res[mode_key]["bridges"].append(model.inter_tower_connections())

            if (seed + 1) % 5 == 0:
                print(f"[{title[:22]}...] Завершено разбиений: {seed + 1}/{n_splits}", flush=True)

        print("\n" + "=" * 108)
        print(f"СРАВНЕНИЕ ПОИСКОВЫХ ОПТИМИЗАТОРОВ ({n_splits} РАЗБИЕНИЙ): {title}")
        print("=" * 108)
        print(f"{'Алгоритм поиска связей и мостов':<36} | {'Test Acc':<15} | {'Macro-F1':<15} | {'Связей':<7} | {'Мосты':<6} | {'Train(с)'}")
        print("-" * 108)
        for key, label in [("GOLDEN", "Golden Section Search (phi=1.618)"),
                           ("GBLR",   "Multi-Basin GBLR + OMP (новый)")]:
            r = res[key]
            print(f"{label:<36} | {np.mean(r['acc']):.4f}±{np.std(r['acc']):.4f} | "
                  f"{np.mean(r['f1']):.4f}±{np.std(r['f1']):.4f} | "
                  f"{np.mean(r['conn']):<7.1f} | {np.mean(r['bridges']):<6.1f} | {np.mean(r['time']):.2f}")

        acc_gld, acc_gblr = np.array(res["GOLDEN"]["acc"]), np.array(res["GBLR"]["acc"])
        diff = acc_gblr - acc_gld
        wins = int(np.sum(diff > 1e-12))
        ties = int(np.sum(np.abs(diff) <= 1e-12))
        losses = int(np.sum(diff < -1e-12))
        t_stat, p_t = ttest_rel(acc_gblr, acc_gld)
        w_stat, p_w = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(diff)

        print("-" * 108)
        print(f"Разница точности (GBLR - GOLDEN): {np.mean(diff):+.4f} ± {np.std(diff):.4f} | "
              f"W/T/L = {wins}/{ties}/{losses} | t = {t_stat:+.4f}, p = {p_t:.4f} (Wilcoxon p = {p_w:.4f})")
        print("=" * 108, flush=True)

if __name__ == "__main__":
    run_search_comparison(n_splits=15)
