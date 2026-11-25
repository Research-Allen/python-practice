import time
import warnings
import numpy as np
from scipy.stats import ttest_rel
from sklearn.datasets import load_digits, make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ГИБРИДНЫЙ ОТБОР ОСЕЙ (2D + МНОГОМЕРНЫЙ ВКЛАД)
# ==============================================================================
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def weighted_bce(y, p, pos_weight=1.0):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -np.mean(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p))

def balanced_accuracy(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    pos = (y_true == 1)
    neg = (y_true == 0)
    tpr = np.mean(y_pred[pos] == 1) if np.any(pos) else 0.0
    tnr = np.mean(y_pred[neg] == 0) if np.any(neg) else 0.0
    return 0.5 * (tpr + tnr)

def detect_informative_axes(X, y, synergy_threshold=0.08, min_pair_acc=0.68, max_axes=20):
    n, d = X.shape
    if d <= 2:
        return np.arange(d)

    sub = min(n, 400)
    Xs, ys = X[:sub], y[:sub]

    pos_ratio = np.mean(ys == 1)
    use_bal = (pos_ratio < 0.25) or (pos_ratio > 0.75)
    score_fn = balanced_accuracy if use_bal else (lambda yt, yp: np.mean(yt == yp))

    single_scores = np.zeros(d)
    for i in range(d):
        dist1 = (Xs[:, i:i+1] - Xs[:, i:i+1].T) ** 2
        np.fill_diagonal(dist1, np.inf)
        single_scores[i] = score_fn(ys, ys[np.argmin(dist1, axis=1)])

    selected = set()
    best_pair = (0, 1)
    best_pair_score = -1.0

    for i in range(d):
        for j in range(i + 1, d):
            dist2 = (Xs[:, i:i+1] - Xs[:, i:i+1].T)**2 + (Xs[:, j:j+1] - Xs[:, j:j+1].T)**2
            np.fill_diagonal(dist2, np.inf)
            pair_acc = score_fn(ys, ys[np.argmin(dist2, axis=1)])

            if pair_acc > best_pair_score:
                best_pair_score = pair_acc
                best_pair = (i, j)

            synergy = pair_acc - max(single_scores[i], single_scores[j])
            if pair_acc >= min_pair_acc and synergy >= synergy_threshold:
                selected.add(i)
                selected.add(j)

    for i in range(d):
        if single_scores[i] >= 0.72:
            selected.add(i)

    # Многомерный отбор для распределенных пространственных данных (Digits и др.)
    if len(selected) < 4 and d >= 10:
        full_dist = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(full_dist, np.inf)
        nn_idx = np.argsort(full_dist, axis=1)[:, :3]
        base_knn = score_fn(ys, (np.mean(ys[nn_idx], axis=1) >= 0.5).astype(int))

        if base_knn > max(best_pair_score + 0.05, 0.75):
            mu1 = np.mean(Xs[ys == 1], axis=0) if np.any(ys == 1) else np.zeros(d)
            mu0 = np.mean(Xs[ys == 0], axis=0) if np.any(ys == 0) else np.zeros(d)
            v1 = np.var(Xs[ys == 1], axis=0) if np.any(ys == 1) else np.ones(d)
            v0 = np.var(Xs[ys == 0], axis=0) if np.any(ys == 0) else np.ones(d)
            fisher_score = ((mu1 - mu0) ** 2) / (v1 + v0 + 1e-6)
            top_k = min(max_axes, max(10, d // 3))
            selected.update(np.argsort(fisher_score)[::-1][:top_k].tolist())

    if len(selected) < 2:
        selected.update(best_pair)

    axes = np.array(sorted(list(selected)))
    if len(axes) > max_axes:
        top_idx = np.argsort(single_scores[axes])[::-1][:max_axes]
        axes = np.sort(axes[top_idx])
    return axes

# ==============================================================================
# 2. МОДИФИЦИРОВАННАЯ GlobalCompetitionAGNN (ВЗВЕШИВАНИЕ + РАСШИРЕННЫЕ КУПОЛА)
# ==============================================================================
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

class GlobalCompetitionAGNN:
    def __init__(self, base_dim=2, max_units=45, learning_rate=0.02, seed=42):
        np.random.seed(seed)
        self.base_dim = base_dim
        self.max_units = max_units
        self.base_lr = learning_rate
        self.lr = learning_rate

        self.units = []
        self.seeds = {}
        self.next_seed_id = 1

        self.W_out = np.random.randn(base_dim, 1) * np.sqrt(2.0 / base_dim)
        self.out_mask = np.ones((base_dim, 1))
        self.b_out = np.zeros((1, 1))
        self.hasher = BaseSpatialHasher(base_dim, n_planes=10, seed=seed)

    @property
    def total_dim(self):
        return self.base_dim + len(self.units)

    def forward(self, X):
        m = len(X)
        self.coords = np.zeros((m, self.total_dim))
        self.coords[:, :self.base_dim] = X
        self.cache = []

        for idx, u in enumerate(self.units):
            if u['type'] == 'local_dome':
                sub_axes = u['sub_axes']
                diff = X[:, sub_axes] - u['center']
                sq_dist = np.sum(diff ** 2, axis=1, keepdims=True)
                act = np.exp(-0.5 * sq_dist / (u['sigma'] ** 2))
                self.cache.append((diff, act))
                self.coords[:, self.base_dim + idx] = act.flatten()
            else:
                in_idx = u['input_indices']
                sub_in = self.coords[:, in_idx]
                w_eff = u['weights'] * u['mask']
                z = sub_in @ w_eff + u['bias']
                self.cache.append((z, None))
                self.coords[:, self.base_dim + idx] = relu(z).flatten()

        w_out_eff = self.W_out * self.out_mask
        self.z_out = self.coords @ w_out_eff + self.b_out
        self.output = sigmoid(self.z_out)
        return self.output

    def backward(self, X, y, frozen_before_idx=0, l2_reg=1e-4, pos_weight=1.0):
        y = y.reshape(-1, 1)
        m = len(y)

        sample_w = np.where(y == 1, pos_weight, 1.0)
        dz_out = (self.output - y) * sample_w

        w_out_eff = self.W_out * self.out_mask
        dW_out = (self.coords.T @ dz_out) / m + l2_reg * w_out_eff
        db_out = np.mean(dz_out, axis=0, keepdims=True)
        d_coords = dz_out @ w_out_eff.T

        self.W_out -= self.lr * (dW_out * self.out_mask)
        self.b_out -= self.lr * db_out

        for idx in range(len(self.units) - 1, -1, -1):
            u = self.units[idx]
            col = self.base_dim + idx
            d_act = d_coords[:, col:col + 1]

            if u['type'] == 'local_dome':
                if idx >= frozen_before_idx:
                    diff, act = self.cache[idx]
                    d_center = np.mean(d_act * act * (diff / (u['sigma'] ** 2)), axis=0)
                    sq_dist = np.sum(diff ** 2, axis=1, keepdims=True)
                    d_sigma = np.mean(d_act * act * (sq_dist / (u['sigma'] ** 3)))

                    u['center'] -= self.lr * 0.5 * d_center
                    u['sigma'] = np.clip(u['sigma'] - self.lr * 0.5 * d_sigma, 0.12, 1.8)
            else:
                z, _ = self.cache[idx]
                dz = d_act * relu_grad(z)
                in_idx = u['input_indices']
                sub_in = self.coords[:, in_idx]
                w_eff = u['weights'] * u['mask']

                d_coords[:, in_idx] += dz @ w_eff.T

                if idx >= frozen_before_idx:
                    dW = (sub_in.T @ dz) / m + l2_reg * w_eff
                    db = np.mean(dz, axis=0, keepdims=True)
                    u['weights'] -= self.lr * (dW * u['mask'])
                    u['bias'] -= self.lr * db

    def train_epoch(self, X, y, frozen_before_idx=0, batch_size=32, pos_weight=1.0):
        idx = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch = idx[start:start + batch_size]
            self.forward(X[batch])
            self.backward(X[batch], y[batch], frozen_before_idx=frozen_before_idx, pos_weight=pos_weight)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int).flatten()

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def prune_weak_connections(self, X, rel_threshold=0.14):
        self.forward(X)
        pruned = 0
        for u in self.units:
            if u['type'] == 'local_dome':
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

            pruned += int(np.sum(u['mask']) - np.sum(new_mask))
            u['mask'] = new_mask
            u['weights'] *= new_mask
        return pruned

    def prune_redundant_outputs(self, X_train, X_val, y_val, metric_fn):
        self.forward(X_train)
        base_score = metric_fn(y_val, self.predict(X_val))
        std_coords = np.std(self.coords, axis=0, keepdims=True).T
        importance = np.abs(self.W_out * self.out_mask) * std_coords

        order = np.argsort(importance[self.base_dim:].flatten()) + self.base_dim
        pruned_dims = 0
        for col in order:
            if self.out_mask[col, 0] == 0:
                continue
            self.out_mask[col, 0] = 0.0
            if metric_fn(y_val, self.predict(X_val)) >= base_score:
                self.W_out[col, 0] = 0.0
                pruned_dims += 1
            else:
                self.out_mask[col, 0] = 1.0
        return pruned_dims

    def count_connections(self):
        internal = sum(len(u['sub_axes']) if u['type'] == 'local_dome' else int(np.sum(u['mask'])) for u in self.units)
        return internal + int(np.sum(self.out_mask))

    def build_global_candidates(self, X, y, tabu_actions, cluster_size=16, min_seed_dist=0.38):
        output = self.forward(X)
        gaps = np.abs(y.reshape(-1, 1) - output).flatten()
        hashes = self.hasher.hash_points(X)
        global_pool = []

        for sid, sdata in self.seeds.items():
            if sdata['depth'] >= 5:
                continue
            action_key = ("DEEPEN", sid, sdata['depth'] + 1)
            if action_key in tabu_actions:
                continue

            sub_axes = sdata['sub_axes']
            dists = np.linalg.norm(X[:, sub_axes] - sdata['center_sub'], axis=1)
            local_idx = np.argsort(dists)[:cluster_size]
            local_gaps = gaps[local_idx]
            err_count = np.sum(local_gaps > 0.30)

            if err_count >= 3:
                avg_gap = float(np.mean(local_gaps))
                score = avg_gap * np.sqrt(err_count)
                global_pool.append({
                    'score': score, 'action': "DEEPEN", 'seed_id': sid,
                    'local_idx': local_idx, 'sub_axes': sub_axes, 'action_key': action_key
                })

        work_mask = gaps > 0.32
        if np.any(work_mask):
            dome_dim = min(self.base_dim, 6)
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
                    rel_compact = np.std(X, axis=0) / (np.std(X[b_idx], axis=0) + 1e-8)
                    sub_axes = sorted(np.argsort(rel_compact)[::-1][:dome_dim].tolist())

                epicenter_sub = X[b_idx[np.argmax(gaps[b_idx])]][sub_axes]
                too_close = any(
                    sdata['sub_axes'] == sub_axes and np.linalg.norm(epicenter_sub - sdata['center_sub']) < min_seed_dist
                    for sdata in self.seeds.values()
                )
                if too_close:
                    continue

                dists = np.linalg.norm(X[:, sub_axes] - epicenter_sub, axis=1)
                local_idx = np.argsort(dists)[:cluster_size]
                local_gaps = gaps[local_idx]
                err_count = np.sum(local_gaps > 0.30)

                if err_count >= 3:
                    avg_gap = float(np.mean(local_gaps))
                    score = avg_gap * np.sqrt(err_count)
                    global_pool.append({
                        'score': score, 'action': "NEW_SEED", 'seed_id': None,
                        'local_idx': local_idx, 'sub_axes': sub_axes, 'action_key': action_key
                    })

        global_pool.sort(key=lambda c: c['score'], reverse=True)
        return global_pool

    def apply_growth(self, X, action, seed_id, local_idx, sub_axes):
        self.forward(X)
        center_sub = np.mean(X[local_idx][:, sub_axes], axis=0)
        spread_sub = np.mean(np.std(X[local_idx][:, sub_axes], axis=0))

        if action == "NEW_SEED":
            sid = self.next_seed_id
            self.next_seed_id += 1
            sigma = float(np.clip(spread_sub * 1.4, 0.25, 0.85))
            unit = {
                'type': 'local_dome', 'seed_id': sid, 'local_dim': len(sub_axes),
                'sub_axes': list(sub_axes), 'center': center_sub.copy(), 'sigma': sigma
            }
            self.seeds[sid] = {
                'center_sub': center_sub.copy(), 'sub_axes': list(sub_axes),
                'unit_indices': [len(self.units)], 'depth': len(sub_axes)
            }
        else:
            sid = seed_id
            seed = self.seeds[sid]
            own_cols = [self.base_dim + u_idx for u_idx in seed['unit_indices']]
            neighbor_cols = []
            for other_sid, other_seed in self.seeds.items():
                if other_sid != sid and other_seed['sub_axes'] == seed['sub_axes']:
                    if np.linalg.norm(seed['center_sub'] - other_seed['center_sub']) < 0.9:
                        neighbor_cols.append(self.base_dim + other_seed['unit_indices'][-1])
                        break

            input_indices = list(range(self.base_dim)) + own_cols + neighbor_cols
            sub_in = self.coords[local_idx][:, input_indices]
            center_in = np.mean(sub_in, axis=0)
            spread_in = np.std(sub_in, axis=0)

            in_dim = len(input_indices)
            new_w = (center_in + 0.08 * spread_in * np.random.randn(in_dim)).reshape(in_dim, 1)
            new_b = (-0.5 * np.dot(center_in, new_w)).reshape(1, 1)

            seed['depth'] += 1
            seed['unit_indices'].append(len(self.units))

            unit = {
                'type': 'deep_axis', 'seed_id': sid, 'local_dim': seed['depth'],
                'sub_axes': list(seed['sub_axes']), 'center': center_sub.copy(),
                'input_indices': input_indices, 'weights': new_w,
                'mask': np.ones((in_dim, 1)), 'bias': new_b
            }

        self.units.append(unit)
        self.W_out = np.vstack([self.W_out, np.random.randn(1, 1) * 0.6])
        self.out_mask = np.vstack([self.out_mask, np.ones((1, 1))])

    def save_state(self):
        u_copy = []
        for u in self.units:
            d = {'type': u['type'], 'seed_id': u['seed_id'], 'local_dim': u['local_dim'],
                 'sub_axes': list(u['sub_axes']), 'center': u['center'].copy()}
            if u['type'] == 'local_dome':
                d['sigma'] = u['sigma']
            else:
                d['input_indices'] = list(u['input_indices'])
                d['weights'] = u['weights'].copy()
                d['mask'] = u['mask'].copy()
                d['bias'] = u['bias'].copy()
            u_copy.append(d)
        s_copy = {
            k: {'center_sub': v['center_sub'].copy(), 'sub_axes': list(v['sub_axes']),
                'unit_indices': list(v['unit_indices']), 'depth': v['depth']}
            for k, v in self.seeds.items()
        }
        return (u_copy, s_copy, self.next_seed_id, self.W_out.copy(), self.out_mask.copy(), self.b_out.copy())

    def restore_state(self, state):
        u_copy, s_copy, next_sid, w_out, o_mask, b_out = state
        self.units = []
        for u in u_copy:
            d = {'type': u['type'], 'seed_id': u['seed_id'], 'local_dim': u['local_dim'],
                 'sub_axes': list(u['sub_axes']), 'center': u['center'].copy()}
            if u['type'] == 'local_dome':
                d['sigma'] = u['sigma']
            else:
                d['input_indices'] = list(u['input_indices'])
                d['weights'] = u['weights'].copy()
                d['mask'] = u['mask'].copy()
                d['bias'] = u['bias'].copy()
            self.units.append(d)
        self.seeds = {
            k: {'center_sub': v['center_sub'].copy(), 'sub_axes': list(v['sub_axes']),
                'unit_indices': list(v['unit_indices']), 'depth': v['depth']}
            for k, v in s_copy.items()
        }
        self.next_seed_id = next_sid
        self.W_out = w_out.copy()
        self.out_mask = o_mask.copy()
        self.b_out = b_out.copy()

    def fit(self, X_train, y_train, X_val, y_val, target_accuracy=0.975,
            epochs_per_stage=120, cluster_size=16, min_loss_delta=0.003,
            prune_threshold=0.14, patience=7):
        n_pos = max(1, int(np.sum(y_train == 1)))
        n_neg = max(1, int(np.sum(y_train == 0)))
        pos_weight = float(np.clip(n_neg / n_pos, 1.0, 4.5))
        is_imbalanced = (pos_weight > 1.8)
        metric_fn = balanced_accuracy if is_imbalanced else (lambda yt, yp: np.mean(yt == yp))

        for _ in range(epochs_per_stage):
            self.train_epoch(X_train, y_train, pos_weight=pos_weight)

        best_acc = metric_fn(y_val, self.predict(X_val))
        best_loss = weighted_bce(y_val.reshape(-1, 1), self.forward(X_val), pos_weight)
        best_state = self.save_state()

        tabu_actions = set()
        stall = 0

        while len(self.units) < self.max_units:
            if best_acc >= target_accuracy or stall >= patience:
                break

            global_pool = self.build_global_candidates(X_train, y_train, tabu_actions, cluster_size)
            if not global_pool:
                break

            best_cand = global_pool[0]
            checkpoint = self.save_state()
            new_unit_idx = len(self.units)

            self.apply_growth(X_train, best_cand['action'], best_cand['seed_id'],
                              best_cand['local_idx'], best_cand['sub_axes'])

            self.lr = self.base_lr
            for _ in range(epochs_per_stage):
                self.train_epoch(X_train, y_train, frozen_before_idx=new_unit_idx, pos_weight=pos_weight)

            self.prune_weak_connections(X_train, rel_threshold=prune_threshold)
            self.lr = self.base_lr * 0.25
            for _ in range(epochs_per_stage // 3):
                self.train_epoch(X_train, y_train, frozen_before_idx=0, pos_weight=pos_weight)

            out = self.forward(X_val)
            acc = metric_fn(y_val, (out > 0.5).astype(int))
            loss = weighted_bce(y_val.reshape(-1, 1), out, pos_weight)

            adaptive_delta = min_loss_delta * (0.97 ** len(self.units))
            improved = (acc > best_acc + 0.002) or (loss < best_loss - adaptive_delta and acc >= best_acc - 0.005)

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
        self.prune_redundant_outputs(X_train, X_val, y_val, metric_fn)

class HybridGlobalCompetitionAGNN:
    def __init__(self, output_dim=2, max_units=35, lr=0.02, seed=42):
        self.output_dim = output_dim
        self.max_units = max_units
        self.lr = lr
        self.seed = seed
        self.models = []

    def fit(self, X, y):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
        tr_idx, val_idx = next(sss.split(X, y))
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        self.models = []
        if self.output_dim == 2:
            axes = detect_informative_axes(X_tr, y_tr, max_axes=15)
            m = GlobalCompetitionAGNN(base_dim=len(axes), max_units=self.max_units,
                                      learning_rate=self.lr, seed=self.seed)
            m.fit(X_tr[:, axes], y_tr, X_val[:, axes], y_val)
            self.models.append((axes, m))
        else:
            sub_units = max(14, self.max_units // 2)
            for c in range(self.output_dim):
                y_tr_bin = (y_tr == c).astype(int)
                y_val_bin = (y_val == c).astype(int)
                axes = detect_informative_axes(X_tr, y_tr_bin, max_axes=14)
                m = GlobalCompetitionAGNN(base_dim=len(axes), max_units=sub_units,
                                          learning_rate=self.lr, seed=self.seed + c)
                m.fit(X_tr[:, axes], y_tr_bin, X_val[:, axes], y_val_bin, epochs_per_stage=90, patience=5)
                self.models.append((axes, m))

    def predict(self, X):
        if self.output_dim == 2:
            axes, m = self.models[0]
            return m.predict(X[:, axes])
        else:
            probs = np.column_stack([m.forward(X[:, axes]).flatten() for axes, m in self.models])
            return np.argmax(probs, axis=1)

    def active_connections(self):
        return sum(m.count_connections() for _, m in self.models)

# ==============================================================================
# 3. ЭТАЛОННЫЙ MLP-32 И ЗАПУСК ПОВТОРНОЙ ПРОВЕРКИ НА ПРОБЛЕМНЫХ ТЕСТАХ (3 И 4)
# ==============================================================================
class BaselineMLP32:
    def __init__(self, input_dim, output_dim, hidden_dim=32, lr=0.1, seed=42):
        self.rng = np.random.default_rng(seed)
        self.output_dim = output_dim
        self.lr = lr
        self.W1 = self.rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        exp_z = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, epochs=300):
        y_oh = np.eye(self.output_dim)[y]
        n = X.shape[0]
        for _ in range(epochs):
            probs = self.forward(X)
            dz2 = (probs - y_oh) / n
            dW2 = self.a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (self.z1 > 0)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def active_connections(self):
        return self.W1.size + self.W2.size

def run_patched_verification(n_splits=10):
    benchmarks = {}
    X3, y3 = make_classification(n_samples=1000, n_features=25, n_informative=6,
                                 n_redundant=6, weights=[0.93, 0.07], flip_y=0.02, random_state=42)
    benchmarks["3. Severe Imbalance (25D, 93:7 ratio) [PATCHED]"] = (X3, y3)

    digits = load_digits()
    benchmarks["4. Multi-Class Digits (64D, 10 classes) [PATCHED]"] = (digits.data, digits.target)

    for title, (X, y) in benchmarks.items():
        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        res = {m: {"acc": [], "f1": [], "conn": [], "inf": []} for m in ["AGNN", "MLP"]}

        for seed, (tr, te) in enumerate(sss.split(X, y)):
            sc = StandardScaler()
            X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
            y_tr, y_te = y[tr], y[te]

            agnn = HybridGlobalCompetitionAGNN(output_dim=output_dim, max_units=35, lr=0.02, seed=seed)
            agnn.fit(X_tr, y_tr)
            t0 = time.perf_counter()
            p_agnn = agnn.predict(X_te)
            res["AGNN"]["inf"].append((time.perf_counter() - t0) * 1000)
            res["AGNN"]["acc"].append(accuracy_score(y_te, p_agnn))
            res["AGNN"]["f1"].append(f1_score(y_te, p_agnn, average="macro"))
            res["AGNN"]["conn"].append(agnn.active_connections())

            mlp = BaselineMLP32(input_dim, output_dim, hidden_dim=32, lr=0.1, seed=seed)
            mlp.fit(X_tr, y_tr, epochs=300)
            t0 = time.perf_counter()
            p_mlp = mlp.predict(X_te)
            res["MLP"]["inf"].append((time.perf_counter() - t0) * 1000)
            res["MLP"]["acc"].append(accuracy_score(y_te, p_mlp))
            res["MLP"]["f1"].append(f1_score(y_te, p_mlp, average="macro"))
            res["MLP"]["conn"].append(mlp.active_connections())

        print("\n" + "=" * 92)
        print(f"СТРЕСС-ТЕСТ: {title} ({n_splits} разбиений)")
        print("=" * 92)
        print(f"{'Модель':<28} | {'Test Acc':<15} | {'Macro-F1':<15} | {'Связей':<8} | {'Inf(мс)'}")
        print("-" * 92)
        for m, label in [("AGNN", "HybridGlobalCompetitionAGNN"), ("MLP", "Baseline MLP-32")]:
            r = res[m]
            print(f"{label:<28} | {np.mean(r['acc']):.4f}±{np.std(r['acc']):.4f} | "
                  f"{np.mean(r['f1']):.4f}±{np.std(r['f1']):.4f} | "
                  f"{np.mean(r['conn']):<8.1f} | {np.mean(r['inf']):.3f}")
        ratio = np.mean(res["MLP"]["conn"]) / max(1e-9, np.mean(res["AGNN"]["conn"]))
        diff_f1 = np.mean(np.array(res["AGNN"]["f1"]) - np.array(res["MLP"]["f1"]))
        _, p_val = ttest_rel(res["AGNN"]["f1"], res["MLP"]["f1"])
        print("-" * 92)
        print(f"Сжатие vs MLP: {ratio:.1f}x | dF1(AGNN-MLP): {diff_f1:+.4f} (p={p_val:.4f})")
        print("=" * 92, flush=True)

if __name__ == "__main__":
    run_patched_verification(n_splits=10)