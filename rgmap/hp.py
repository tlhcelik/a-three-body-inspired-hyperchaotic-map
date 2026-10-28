"""60-digit reference implementation of the N=3 map (mpmath), used for validity
horizons and numerical floors."""
import mpmath as mp

mp.mp.dps = 60


def step(q, a, eps=mp.mpf("0.1"), R2=mp.mpf(2), m=(1, 1, 1)):
    """q: list of 3 lists of 3 mpf."""
    a = mp.mpf(a); N = len(q)
    rt = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                rt[i][j] = mp.sqrt(sum((q[i][k] - q[j][k]) ** 2 for k in range(3)) + eps ** 2)
    out = []
    for i in range(N):
        Phi = -sum(mp.mpf(m[j]) / rt[i][j] for j in range(N) if j != i)
        B = 1 - sum(q[i][k] ** 2 for k in range(3)) / R2
        row = []
        for k in range(3):
            F = sum(mp.mpf(m[i]) * m[j] * (q[j][k] - q[i][k]) / rt[i][j] ** 3 for j in range(N) if j != i)
            D = a * q[i][k] * (1 + Phi) + B * q[i][k] + F
            row.append(mp.sin(1 / D))
        out.append(row)
    return out


def from_np(Q):
    return [[mp.mpf(float(v)) for v in row] for row in Q]


def to_float(q):
    return [[float(v) for v in row] for row in q]
