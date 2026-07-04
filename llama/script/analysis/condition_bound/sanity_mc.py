"""CPU Monte Carlo sanity checks for every inequality used in the condition proofs.

Checks (all must PASS):
  1. TV lemma:            sum_i |1/n - softmax(x)_i| <= 2 tanh(delta/2)
  2. Bennett lemma:       (1/n) sum e^{x_i} <= G(sigma, delta), G(d,d) = cosh(d)
  3. All-approx bound:    |o_hat - o| <= condition (cosh version)
  4. Hybrid certificate:  |o_hyb - o| <= 2B T/(1+T) + tanh part, random selections
"""

import math
import random


def G(sigma2, delta):
    if delta <= 0:
        return 1.0
    return (sigma2 * math.exp(delta) + delta * delta * math.exp(-sigma2 / delta)) / (sigma2 + delta * delta)


def mean_zero_sample(n, d):
    x = [random.uniform(-d, d) for _ in range(n)]
    mu = sum(x) / n
    return [xi - mu for xi in x]


def check_tv_lemma(trials=100000):
    worst = 0.0
    for _ in range(trials):
        x = mean_zero_sample(random.randint(2, 12), random.uniform(0.01, 5.0))
        dmax = max(abs(v) for v in x)
        if dmax < 1e-9:
            continue
        ex = [math.exp(v) for v in x]
        t = sum(ex)
        lhs = sum(abs(1.0 / len(x) - e / t) for e in ex)
        worst = max(worst, lhs / (2 * math.tanh(dmax / 2)))
    return worst


def check_bennett(trials=100000):
    worst = 0.0
    for _ in range(trials):
        x = mean_zero_sample(random.randint(2, 16), random.uniform(0.01, 6.0))
        dmax = max(abs(v) for v in x)
        if dmax < 1e-9:
            continue
        s2 = sum(v * v for v in x) / len(x)
        lhs = sum(math.exp(v) for v in x) / len(x)
        worst = max(worst, lhs / G(s2, dmax))
    cosh_gap = max(abs(G(d * d, d) - math.cosh(d)) / math.cosh(d) for d in (0.1, 1.0, 5.0, 15.0))
    return worst, cosh_gap


def random_clusters(m_lo=2, m_hi=6):
    clusters = []
    for _ in range(random.randint(m_lo, m_hi)):
        n = random.randint(1, 8)
        s = [random.uniform(-3, 3) for _ in range(n)]
        v = [random.uniform(-2, 2) for _ in range(n)]
        clusters.append((s, v))
    return clusters


def cluster_stats(clusters):
    B = max(abs(vi) for s, v in clusters for vi in v)
    Z = sum(math.exp(x) for s, v in clusters for x in s)
    o = sum(math.exp(x) * vi for s, v in clusters for x, vi in zip(s, v)) / Z
    zs = [len(s) * math.exp(sum(s) / len(s)) for s, v in clusters]
    ph = [z / sum(zs) for z in zs]
    deltas = [max(abs(x - sum(s) / len(s)) for x in s) for s, v in clusters]
    Bc = [max(abs(vi) for vi in v) for s, v in clusters]
    return B, o, ph, deltas, Bc


def check_full_bound(trials=50000):
    worst = 0.0
    for _ in range(trials):
        clusters = random_clusters()
        B, o, ph, deltas, Bc = cluster_stats(clusters)
        o_hat = sum(p * (sum(v) / len(v)) for p, (s, v) in zip(ph, clusters))
        S = sum(p * math.cosh(d) for p, d in zip(ph, deltas))
        bound = sum(p * (2 * B * (math.cosh(d) - 1) / S + 2 * bc * math.tanh(d / 2))
                    for p, d, bc in zip(ph, deltas, Bc))
        if bound > 0:
            worst = max(worst, abs(o_hat - o) / bound)
    return worst


def check_hybrid(trials=30000):
    worst = 0.0
    for _ in range(trials):
        clusters = random_clusters(3, 8)
        B, o, ph, deltas, Bc = cluster_stats(clusters)
        sel = [random.random() < 0.4 for _ in clusters]
        num = den = 0.0
        for (s, v), is_sel in zip(clusters, sel):
            if is_sel:
                den += sum(math.exp(x) for x in s)
                num += sum(math.exp(x) * vi for x, vi in zip(s, v))
            else:
                zc = len(s) * math.exp(sum(s) / len(s))
                den += zc
                num += zc * (sum(v) / len(v))
        err = abs(num / den - o)
        T = sum(p * (math.cosh(d) - 1) for p, d, is_sel in zip(ph, deltas, sel) if not is_sel)
        cert = 2 * B * T / (1 + T) + sum(2 * p * bc * math.tanh(d / 2)
                                         for p, bc, d, is_sel in zip(ph, Bc, deltas, sel) if not is_sel)
        if cert > 0:
            worst = max(worst, err / cert)
    return worst


def main():
    random.seed(0)
    checks = []
    r = check_tv_lemma()
    checks.append(("TV lemma  (<=1)", r))
    r_b, cosh_gap = check_bennett()
    checks.append(("Bennett lemma (<=1)", r_b))
    checks.append(("G(d,d)=cosh(d) rel gap (~0)", cosh_gap))
    checks.append(("all-approx bound (<=1)", check_full_bound()))
    checks.append(("hybrid certificate (<=1)", check_hybrid()))
    ok = True
    for name, val in checks:
        passed = val <= 1.0 + 1e-9
        ok &= passed
        print(f"{name:32s} worst={val:.6f}  {'PASS' if passed else 'FAIL'}")
    print("ALL PASS" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
