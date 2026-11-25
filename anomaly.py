"""Compare z-score, IQR and Isolation Forest outlier detection on synthetic data with known injected anomalies."""
import numpy as np
from sklearn.ensemble import IsolationForest


def make_data(seed=0, n=500, n_out=15):
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 5, n)
    idx = rng.choice(n, n_out, replace=False)
    x[idx] = rng.choice([-1, 1], n_out) * rng.uniform(25, 45, n_out) + 50
    truth = np.zeros(n, bool)
    truth[idx] = True
    return x, truth


def zscore(x, t=3.0):
    return np.abs((x - x.mean()) / x.std()) > t


def iqr(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    return (x < q1 - k * (q3 - q1)) | (x > q3 + k * (q3 - q1))


def iforest(x, contamination=0.03):
    return IsolationForest(contamination=contamination, random_state=0).fit_predict(x.reshape(-1, 1)) == -1


def prf(pred, truth):
    tp = int((pred & truth).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / int(truth.sum())
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


if __name__ == "__main__":
    x, truth = make_data()
    print(f"{'method':<18}{'precision':>10}{'recall':>8}{'f1':>7}")
    for n, fn in [("zscore(3)", zscore), ("iqr(1.5)", iqr), ("isolation_forest", iforest)]:
        p, r, f = prf(fn(x), truth)
        print(f"{n:<18}{p:>10.2f}{r:>8.2f}{f:>7.2f}")
