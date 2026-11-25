import numpy as np

np.random.seed(42)

# =====================================================
# 1. ГЕНЕРАТОР ДАННЫХ И СИНЕРГЕТИЧЕСКИЙ ОТБОР ОСЕЙ
# =====================================================
def make_two_spirals(samples=2000, turns=2.0, noise=0.12, noise_dims=15, seed=42):
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

    y = np.concatenate([np.zeros(n), np.ones(n)])
    noise_idx = rng.choice(samples, int(0.03 * samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    perm = rng.permutation(samples)
    return X[perm], y[perm]

def split_data(X, y):
    n = len(X)
    train_end, val_end = int(0.60 * n), int(0.80 * n)
    return (
        X[:train_end], X[train_end:val_end], X[val_end:],
        y[:train_end], y[train_end:val_end], y[val_end:]
    )

def standardize(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std

def detect_informative_axes(X, y, synergy_threshold=0.08, min_pair_acc=0.68):
    """
    Отбирает оси через нелинейную синергию 1-NN:
    пара (i, j) считается значимой, только если вместе они разделяют классы
    существенно лучше, чем каждая ось i или j по отдельности.
    """
    n, d = X.shape
    if d <= 2:
        return np.arange(d)

    sub = min(n, 450)
    Xs, ys = X[:sub], y[:sub]

    # 1. Одиночная 1-NN согласованность каждой оси
    single_scores = np.zeros(d)
    for i in range(d):
        dist1 = (Xs[:, i:i+1] - Xs[:, i:i+1].T) ** 2
        np.fill_diagonal(dist1, np.inf)
        nn1 = np.argmin(dist1, axis=1)
        single_scores[i] = np.mean(ys == ys[nn1])

    # 2. Парная синергия
    selected = set()
    best_pair = (0, 1)
    best_pair_score = -1.0

    for i in range(d):
        for j in range(i + 1, d):
            diff0 = Xs[:, i:i+1] - Xs[:, i:i+1].T
            diff1 = Xs[:, j:j+1] - Xs[:, j:j+1].T
            dist2 = diff0**2 + diff1**2
            np.fill_diagonal(dist2, np.inf)
            nn2 = np.argmin(dist2, axis=1)
            pair_acc = np.mean(ys == ys[nn2])

            if pair_acc > best_pair_score:
                best_pair_score = pair_acc
                best_pair = (i, j)

            synergy = pair_acc - max(single_scores[i], single_scores[j])
            if pair_acc >= min_pair_acc and synergy >= synergy_threshold:
                selected.add(i)
                selected.add(j)

    # Также оставляем одиночные оси с сильным линейным/одномерным сигналом
    for i in range(d):
        if single_scores[i] >= 0.72:
            selected.add(i)

    if len(selected) < 2:
        selected.update(best_pair)

    return np.array(sorted(list(selected)))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def bce(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# =====================================================
# 2. ПРОСТРАНСТВЕННЫЙ ХЕШЕР
# =====================================================
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

# =====================================================
# 3. GLOBAL COMPETITION AGNN (С ПОДДЕРЖКОЙ ЛОКАЛЬНЫХ ПОДПРОСТРАНСТВ)
# =====================================================
class GlobalCompetitionAGNN:
    def __init__(self, base_dim=2, max_units=45, learning_rate=0.02):
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

        self.hasher = BaseSpatialHasher(base_dim, n_planes=10)

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
                # Купол работает в компактном подпространстве (2-3 наиболее плотные оси очага)
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

    def backward(self, X, y, frozen_before_idx=0, l2_reg=1e-4):
        y = y.reshape(-1, 1)
        m = len(y)

        dz_out = self.output - y
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
                    u['sigma'] = np.clip(u['sigma'] - self.lr * 0.5 * d_sigma, 0.12, 1.5)
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

    def train_epoch(self, X, y, frozen_before_idx=0, batch_size=32):
        idx = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch = idx[start:start + batch_size]
            self.forward(X[batch])
            self.backward(X[batch], y[batch], frozen_before_idx=frozen_before_idx)

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

    def prune_redundant_outputs(self, X_train, X_val, y_val):
        self.forward(X_train)
        base_acc = self.accuracy(X_val, y_val)
        std_coords = np.std(self.coords, axis=0, keepdims=True).T
        importance = np.abs(self.W_out * self.out_mask) * std_coords

        order = np.argsort(importance[self.base_dim:].flatten()) + self.base_dim
        pruned_dims = 0
        for col in order:
            if self.out_mask[col, 0] == 0:
                continue
            self.out_mask[col, 0] = 0.0
            if self.accuracy(X_val, y_val) >= base_acc:
                self.W_out[col, 0] = 0.0
                pruned_dims += 1
            else:
                self.out_mask[col, 0] = 1.0
        return pruned_dims

    def count_connections(self):
        internal = sum(len(u['sub_axes']) if u['type'] == 'local_dome' else int(np.sum(u['mask'])) for u in self.units)
        return internal + int(np.sum(self.out_mask))

    # -------------------------------------------------
    # ГЛОБАЛЬНОЕ СРАВНЕНИЕ ВСЕЙ СЕТИ: СТАРЫЕ БАШНИ VS НОВЫЕ ОЧАГИ
    # -------------------------------------------------
    def build_global_candidates(self, X, y, tabu_actions, cluster_size=16, min_seed_dist=0.38):
        output = self.forward(X)
        gaps = np.abs(y.reshape(-1, 1) - output).flatten()
        hashes = self.hasher.hash_points(X)

        global_pool = []

        # А. Оцениваем ВСЕ существующие башни на фоне всей текущей сети
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
                    'score': score,
                    'action': "DEEPEN",
                    'seed_id': sid,
                    'local_idx': local_idx,
                    'sub_axes': sub_axes,
                    'action_key': action_key,
                    'avg_gap': avg_gap,
                    'center': sdata['center_sub']
                })

        # Б. Оцениваем ВСЕ непосещенные хеш-зоны в остальном пространстве
        work_mask = gaps > 0.32
        if np.any(work_mask):
            unique_hashes = np.unique(hashes[work_mask])
            for h in unique_hashes:
                h_int = int(h)
                action_key = ("NEW_SEED", h_int)
                if action_key in tabu_actions:
                    continue

                b_idx = np.where((hashes == h_int) & work_mask)[0]
                if len(b_idx) < 3:
                    continue

                # Выбираем до 3 наиболее плотных осей для этого локального купола
                if self.base_dim <= 3:
                    sub_axes = list(range(self.base_dim))
                else:
                    rel_compact = np.std(X, axis=0) / (np.std(X[b_idx], axis=0) + 1e-8)
                    sub_axes = sorted(np.argsort(rel_compact)[::-1][:3].tolist())

                epicenter_sub = X[b_idx[np.argmax(gaps[b_idx])]][sub_axes]

                too_close = False
                for sdata in self.seeds.values():
                    if sdata['sub_axes'] == sub_axes:
                        if np.linalg.norm(epicenter_sub - sdata['center_sub']) < min_seed_dist:
                            too_close = True
                            break
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
                        'score': score,
                        'action': "NEW_SEED",
                        'seed_id': None,
                        'local_idx': local_idx,
                        'sub_axes': sub_axes,
                        'action_key': action_key,
                        'avg_gap': avg_gap,
                        'center': epicenter_sub
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
            sigma = float(np.clip(spread_sub * 1.4, 0.22, 0.55))

            unit = {
                'type': 'local_dome',
                'seed_id': sid,
                'local_dim': 3,
                'sub_axes': list(sub_axes),
                'center': center_sub.copy(),
                'sigma': sigma
            }
            self.seeds[sid] = {
                'center_sub': center_sub.copy(),
                'sub_axes': list(sub_axes),
                'unit_indices': [len(self.units)],
                'depth': 3
            }
            local_dim = 3
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
            local_dim = seed['depth']
            seed['unit_indices'].append(len(self.units))

            unit = {
                'type': 'deep_axis',
                'seed_id': sid,
                'local_dim': local_dim,
                'sub_axes': list(seed['sub_axes']),
                'center': center_sub.copy(),
                'input_indices': input_indices,
                'weights': new_w,
                'mask': np.ones((in_dim, 1)),
                'bias': new_b
            }

        self.units.append(unit)
        self.W_out = np.vstack([self.W_out, np.random.randn(1, 1) * 0.6])
        self.out_mask = np.vstack([self.out_mask, np.ones((1, 1))])
        return sid, local_dim, center_sub

    def save_state(self):
        u_copy = []
        for u in self.units:
            d = {
                'type': u['type'], 'seed_id': u['seed_id'], 'local_dim': u['local_dim'],
                'sub_axes': list(u['sub_axes']), 'center': u['center'].copy()
            }
            if u['type'] == 'local_dome':
                d['sigma'] = u['sigma']
            else:
                d['input_indices'] = list(u['input_indices'])
                d['weights'] = u['weights'].copy()
                d['mask'] = u['mask'].copy()
                d['bias'] = u['bias'].copy()
            u_copy.append(d)
        s_copy = {
            k: {
                'center_sub': v['center_sub'].copy(), 'sub_axes': list(v['sub_axes']),
                'unit_indices': list(v['unit_indices']), 'depth': v['depth']
            }
            for k, v in self.seeds.items()
        }
        return (u_copy, s_copy, self.next_seed_id, self.W_out.copy(), self.out_mask.copy(), self.b_out.copy())

    def restore_state(self, state):
        u_copy, s_copy, next_sid, w_out, o_mask, b_out = state
        self.units = []
        for u in u_copy:
            d = {
                'type': u['type'], 'seed_id': u['seed_id'], 'local_dim': u['local_dim'],
                'sub_axes': list(u['sub_axes']), 'center': u['center'].copy()
            }
            if u['type'] == 'local_dome':
                d['sigma'] = u['sigma']
            else:
                d['input_indices'] = list(u['input_indices'])
                d['weights'] = u['weights'].copy()
                d['mask'] = u['mask'].copy()
                d['bias'] = u['bias'].copy()
            self.units.append(d)
        self.seeds = {
            k: {
                'center_sub': v['center_sub'].copy(), 'sub_axes': list(v['sub_axes']),
                'unit_indices': list(v['unit_indices']), 'depth': v['depth']
            }
            for k, v in s_copy.items()
        }
        self.next_seed_id = next_sid
        self.W_out = w_out.copy()
        self.out_mask = o_mask.copy()
        self.b_out = b_out.copy()

    def fit(
        self,
        X_train, y_train,
        X_val, y_val,
        target_accuracy=0.955,
        epochs_per_stage=180,
        cluster_size=16,
        min_loss_delta=0.003,
        prune_threshold=0.14,
        patience=10
    ):
        for _ in range(epochs_per_stage):
            self.train_epoch(X_train, y_train)

        best_acc = self.accuracy(X_val, y_val)
        best_loss = bce(y_val.reshape(-1, 1), self.forward(X_val))
        best_state = self.save_state()

        print(f"[База {self.base_dim}D] Acc={best_acc:.4f} Loss={best_loss:.4f}")

        tabu_actions = set()
        stall = 0

        while len(self.units) < self.max_units:
            if best_acc >= target_accuracy:
                print("\nЦелевая точность достигнута!")
                break
            if stall >= patience:
                print("\nГлобальный оптимум всей сети достигнут.")
                break

            global_pool = self.build_global_candidates(
                X_train, y_train, tabu_actions, cluster_size
            )
            if not global_pool:
                print("\nВсе кандидаты во всей сети исчерпаны.")
                break

            best_cand = global_pool[0]
            runner_up_info = ""
            if len(global_pool) > 1:
                sec = global_pool[1]
                sec_name = f"Очаг #{sec['seed_id']}" if sec['action'] == "DEEPEN" else "Новый очаг"
                runner_up_info = f" (Конкурент: {sec_name}, Score={sec['score']:.2f})"

            checkpoint = self.save_state()
            new_unit_idx = len(self.units)

            sid, local_dim, center_sub = self.apply_growth(
                X_train, best_cand['action'], best_cand['seed_id'],
                best_cand['local_idx'], best_cand['sub_axes']
            )
            tag = (
                f"НОВЫЙ Очаг #{sid} (2D->3D)"
                if best_cand['action'] == "NEW_SEED"
                else f"Прыжок в Очаг #{sid} (-> {local_dim}D)"
            )
            print(
                f"-> {tag} | Score={best_cand['score']:.2f} | "
                f"Коорд: [{center_sub[0]:+.2f}, {center_sub[1]:+.2f}]{runner_up_info}"
            )

            self.lr = self.base_lr
            for _ in range(epochs_per_stage):
                self.train_epoch(X_train, y_train, frozen_before_idx=new_unit_idx)

            pruned = self.prune_weak_connections(X_train, rel_threshold=prune_threshold)
            self.lr = self.base_lr * 0.25
            for _ in range(epochs_per_stage // 3):
                self.train_epoch(X_train, y_train, frozen_before_idx=0)

            out = self.forward(X_val)
            acc = self.accuracy(X_val, y_val)
            loss = bce(y_val.reshape(-1, 1), out)

            adaptive_delta = min_loss_delta * (0.97 ** len(self.units))
            improved = (acc > best_acc + 0.002) or (loss < best_loss - adaptive_delta and acc >= best_acc - 0.005)

            if improved:
                print(
                    f"   [+] Успех! Acc={acc:.4f} Loss={loss:.4f} | "
                    f"Прунинг: -{pruned} связей (Живых: {self.count_connections()})"
                )
                best_acc = max(best_acc, acc)
                best_loss = min(best_loss, loss)
                best_state = self.save_state()
                stall = 0
            else:
                print(f"   [-] Без значимого улучшения (Acc={acc:.4f}, Loss={loss:.4f}). Откат и прыжок в другое место.")
                self.restore_state(checkpoint)
                tabu_actions.add(best_cand['action_key'])
                stall += 1

        self.restore_state(best_state)
        pruned_out = self.prune_redundant_outputs(X_train, X_val, y_val)
        if pruned_out > 0:
            print(f"[Финальный прунинг] Отключено лишних выходных измерений: {pruned_out}")

# =====================================================
# ЗАПУСК НА 17D ДАННЫХ (2 СИГНАЛА + 15 ШУМОВЫХ ОСЕЙ)
# =====================================================
X_raw, y_raw = make_two_spirals(samples=2000, turns=2.0, noise=0.12, noise_dims=15, seed=42)
X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X_raw, y_raw)
X_tr, X_v, X_te = standardize(X_tr, X_v, X_te)

active_axes = detect_informative_axes(X_tr, y_tr)
print(f"Обнаружены информативные оси из {X_raw.shape[1]}D: {active_axes.tolist()}")

model = GlobalCompetitionAGNN(base_dim=len(active_axes), max_units=45, learning_rate=0.02)
model.fit(
    X_tr[:, active_axes], y_tr,
    X_v[:, active_axes], y_v,
    target_accuracy=0.955,
    epochs_per_stage=180,
    cluster_size=16,
    min_loss_delta=0.003,
    prune_threshold=0.14,
    patience=10
)

print("\n" + "=" * 65)
print(f"Итоговое число очагов: {len(model.seeds)} | Нейронов-измерений: {len(model.units)} | Живых связей: {model.count_connections()}")
print("Validation accuracy:", round(model.accuracy(X_v[:, active_axes], y_v), 4))
print("Test accuracy (17D):", round(model.accuracy(X_te[:, active_axes], y_te), 4))
