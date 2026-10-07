import numpy as np
import torch
from sklearn.datasets import load_iris

EPS = 1e-6


def load_split():
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target
    rng = np.random.default_rng(0)
    tr, te = [], []
    for c in (0, 1, 2):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        tr.append(idx[:35])
        te.append(idx[35:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0, ddof=0)
    return (X[tr] - mean) / std, y[tr], (X[te] - mean) / std, y[te]


def init_params():
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, np.sqrt(2 / 4), (4, 8))
    W2 = rng.normal(0, np.sqrt(2 / (8 + 3)), (8, 3))
    return {"W1": W1, "b1": np.zeros(8), "W2": W2, "b2": np.zeros(3)}


def log_softmax(Z):
    s = Z - Z.max(axis=1, keepdims=True)
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


class MLP:
    def __init__(self):
        self.p = init_params()

    def forward(self, X):
        Z1 = X @ self.p["W1"] + self.p["b1"]
        A1 = np.maximum(Z1, 0)
        Z2 = A1 @ self.p["W2"] + self.p["b2"]
        self.cache = (X, Z1, A1, Z2)
        return Z2

    def loss(self, X, y):
        logp = log_softmax(self.forward(X))
        return float(-logp[np.arange(len(y)), y].mean())

    def backward(self, y, buggy=False):
        X, Z1, A1, Z2 = self.cache
        N = len(y)
        dZ2 = np.exp(log_softmax(Z2))
        dZ2[np.arange(N), y] -= 1
        if not buggy:
            dZ2 /= N
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dZ1 = (dZ2 @ self.p["W2"].T) * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def torch_reference(p, X, y):
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(p["W1"].T.copy()))
        net[0].bias.copy_(torch.from_numpy(p["b1"].copy()))
        net[2].weight.copy_(torch.from_numpy(p["W2"].T.copy()))
        net[2].bias.copy_(torch.from_numpy(p["b2"].copy()))
    loss = torch.nn.functional.cross_entropy(net(torch.from_numpy(X)), torch.from_numpy(y))
    loss.backward()
    grads = {
        "W1": net[0].weight.grad.T.numpy(),
        "b1": net[0].bias.grad.numpy(),
        "W2": net[2].weight.grad.T.numpy(),
        "b2": net[2].bias.grad.numpy(),
    }
    return loss.item(), grads


def yes(ok):
    return "так" if ok else "ні"


def print_table(header, rows):
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    line = lambda cells: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells)) + " |"
    print(line(header))
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print(line(r))


def check_torch(X, y, buggy):
    net = MLP()
    l_np = net.loss(X, y)
    g_np = net.backward(y, buggy)
    l_t, g_t = torch_reference(net.p, X, y)
    print(f"Втрата NumPy   = {l_np!r}")
    print(f"Втрата PyTorch = {l_t!r}\n")
    d = abs(l_np - l_t)
    rows = [["Втрата", f"{d:.3e}", yes(np.isfinite(d) and d <= 1e-12)]]
    for k in ("W1", "b1", "W2", "b2"):
        d = np.max(np.abs(g_np[k] - g_t[k]))
        rows.append([f"Градієнт {k}", f"{d:.3e}", yes(np.isfinite(d) and d <= 1e-12)])
    print_table(["Величина", "Макс. абсолютна різниця", "Перевірка"], rows)


def check_numeric(X, y, buggy):
    net = MLP()
    net.loss(X, y)
    g = net.backward(y, buggy)
    rows = []
    for name, idx in [("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,))]:
        a = net.p[name]
        old = a[idx]
        a[idx] = old + EPS
        lp = net.loss(X, y)
        a[idx] = old - EPS
        lm = net.loss(X, y)
        a[idx] = old
        g_num = (lp - lm) / (2 * EPS)
        d = abs(g_num - g[name][idx])
        label = f"{name}[{', '.join(map(str, idx))}]"
        rows.append([label, f"{g[name][idx]:.12e}", f"{g_num:.12e}", f"{d:.3e}", yes(d <= 1e-7)])
    print_table(["Параметр", "backward()", "Чисельна похідна", "Абс. різниця", "Перевірка"], rows)


def run(buggy):
    X, y, _, _ = load_split()
    print("дослід: без ділення на N" if buggy else "правильна реалізація")
    print("\nЗвірка з PyTorch\n")
    check_torch(X, y, buggy)
    print("\nЧисельна перевірка\n")
    check_numeric(X, y, buggy)
    if buggy:
        net = MLP()
        net.loss(X, y)
        good, bad = net.backward(y), net.backward(y, buggy=True)
        print("\nВідношення градієнтів (з помилкою / правильний):")
        for k in good:
            nz = good[k] != 0
            r = bad[k][nz] / good[k][nz]
            print(f"  {k}: {r.min():.6f} .. {r.max():.6f}")
    print()


if __name__ == "__main__":
    run(buggy=False)
    run(buggy=True)