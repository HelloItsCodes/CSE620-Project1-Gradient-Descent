#!/usr/bin/env python3
"""
CSE 620 Fall 2026, Project 1
Aibinh Nguyen, Alwin Rajkumar
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

MAX_ITER = 2000
TOL = 1e-6
DIVERGE = 1e8

# f, grad, Hessian


def f_quadratic(x: np.ndarray) -> float:
    """f = x^2 + y^2. Strictly convex bowl. Unique min at (0, 0), f=0."""
    return float(x[0] ** 2 + x[1] ** 2)


def g_quadratic(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * x[0], 2.0 * x[1]], dtype=float)


def h_quadratic(x: np.ndarray) -> np.ndarray:
    return np.array([[2.0, 0.0], [0.0, 2.0]], dtype=float)


def f_rosenbrock(x: np.ndarray) -> float:
    """Classic banana: f = (1-x)^2 + 100(y-x^2)^2. Unique min (1, 1), f=0. Not convex."""
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def g_rosenbrock(x: np.ndarray) -> np.ndarray:
    u, v = x[0], x[1]
    dx = -2.0 * (1.0 - u) - 400.0 * u * (v - u ** 2)
    dy = 200.0 * (v - u ** 2)
    return np.array([dx, dy], dtype=float)


def h_rosenbrock(x: np.ndarray) -> np.ndarray:
    u, v = x[0], x[1]
    hxx = 2.0 - 400.0 * v + 1200.0 * u ** 2
    hxy = -400.0 * u
    hyy = 200.0
    return np.array([[hxx, hxy], [hxy, hyy]], dtype=float)


def f_cosine(x: np.ndarray) -> float:
    """f = x^2 + y^2 + 10 cos x + 10 cos y. Separable, multi-modal, non-convex."""
    return float(x[0] ** 2 + x[1] ** 2 + 10.0 * np.cos(x[0]) + 10.0 * np.cos(x[1]))


def g_cosine(x: np.ndarray) -> np.ndarray:
    return np.array(
        [2.0 * x[0] - 10.0 * np.sin(x[0]), 2.0 * x[1] - 10.0 * np.sin(x[1])],
        dtype=float,
    )


def h_cosine(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [2.0 - 10.0 * np.cos(x[0]), 0.0],
            [0.0, 2.0 - 10.0 * np.cos(x[1])],
        ],
        dtype=float,
    )


FUNCTIONS = {
    "quadratic": {
        "f": f_quadratic,
        "g": g_quadratic,
        "h": h_quadratic,
        "title": r"$f=x^2+y^2$",
        "minima": [(0.0, 0.0)],
        "xlim": (-3.5, 3.5),
        "ylim": (-3.5, 3.5),
        "log": True,
        "convex": True,
    },
    "rosenbrock": {
        "f": f_rosenbrock,
        "g": g_rosenbrock,
        "h": h_rosenbrock,
        "title": r"Rosenbrock $(1-x)^2+100(y-x^2)^2$",
        "minima": [(1.0, 1.0)],
        "xlim": (-2.5, 2.5),
        "ylim": (-1.5, 3.5),
        "log": True,
        "convex": False,
    },
    "cosine": {
        "f": f_cosine,
        "g": g_cosine,
        "h": h_cosine,
        "title": r"$x^2+y^2+10\cos x+10\cos y$",
        "minima": [],
        "xlim": (-6.5, 6.5),
        "ylim": (-6.5, 6.5),
        "log": False,
        "convex": False,
    },
}

STARTS = {
    1: np.array([-2.0, 2.0]),
    2: np.array([0.5, -1.5]),
    3: np.array([3.0, 3.0]),
}

PARAM_GRID = {
    "quadratic": {
        "GD": [0.001, 0.01, 0.1],
        "Newton": [0.1, 0.5, 1.0],
        "AdaGrad": [0.001, 0.01, 0.1],
        "Adam": [0.001, 0.01, 0.1],
    },
    "rosenbrock": {
        "GD": [0.001, 0.01, 0.1],
        "Newton": [0.1, 0.5, 1.0],
        "AdaGrad": [0.001, 0.01, 0.1],
        "Adam": [0.001, 0.01, 0.1],
    },
    "cosine": {
        "GD": [0.001, 0.01, 0.1],
        "Newton": [0.1, 0.5, 1.0],
        "AdaGrad": [0.001, 0.01, 0.1],
        "Adam": [0.001, 0.01, 0.1],
    },
}


def _finite(x: np.ndarray, val: float) -> bool:
    return np.all(np.isfinite(x)) and math.isfinite(val) and np.linalg.norm(x) < DIVERGE


def run_gd(f, g, h, x0, alpha, max_iter=MAX_ITER, tol=TOL):
    x = x0.astype(float).copy()
    path = [x.copy()]
    vals = [f(x)]
    for _ in range(max_iter):
        grad = g(x)
        x_new = x - alpha * grad
        val = f(x_new)
        if not _finite(x_new, val):
            return path, vals, "diverged"
        path.append(x_new.copy())
        vals.append(val)
        if np.linalg.norm(x_new - x) < tol:
            return path, vals, "converged"
        x = x_new
    return path, vals, "max_iter"


def run_newton(f, g, h, x0, alpha, max_iter=MAX_ITER, tol=TOL):
    """x <- x - alpha * H^{-1} g. Add lambda I if H is not PD."""
    x = x0.astype(float).copy()
    path = [x.copy()]
    vals = [f(x)]
    for _ in range(max_iter):
        grad = g(x)
        H = h(x)
        lam, _ = np.linalg.eigh(H)
        damp = 0.0
        if lam.min() <= 1e-8:
            damp = max(1e-6, 1e-4 - lam.min())
        try:
            step = np.linalg.solve(H + damp * np.eye(2), grad)
        except np.linalg.LinAlgError:
            step = grad
        x_new = x - alpha * step
        val = f(x_new)
        if not _finite(x_new, val):
            return path, vals, "diverged"
        path.append(x_new.copy())
        vals.append(val)
        if np.linalg.norm(x_new - x) < tol:
            return path, vals, "converged"
        x = x_new
    return path, vals, "max_iter"


def run_adagrad(f, g, h, x0, alpha, max_iter=MAX_ITER, tol=TOL, eps=1e-8):
    x = x0.astype(float).copy()
    s = np.zeros(2)
    path = [x.copy()]
    vals = [f(x)]
    for _ in range(max_iter):
        grad = g(x)
        s = s + grad * grad
        x_new = x - alpha * grad / (np.sqrt(s) + eps)
        val = f(x_new)
        if not _finite(x_new, val):
            return path, vals, "diverged"
        path.append(x_new.copy())
        vals.append(val)
        if np.linalg.norm(x_new - x) < tol:
            return path, vals, "converged"
        x = x_new
    return path, vals, "max_iter"


def run_adam(f, g, h, x0, alpha, max_iter=MAX_ITER, tol=TOL, b1=0.9, b2=0.999, eps=1e-8):
    x = x0.astype(float).copy()
    m = np.zeros(2)
    v = np.zeros(2)
    path = [x.copy()]
    vals = [f(x)]
    for t in range(1, max_iter + 1):
        grad = g(x)
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * (grad * grad)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        x_new = x - alpha * mhat / (np.sqrt(vhat) + eps)
        val = f(x_new)
        if not _finite(x_new, val):
            return path, vals, "diverged"
        path.append(x_new.copy())
        vals.append(val)
        if np.linalg.norm(x_new - x) < tol:
            return path, vals, "converged"
        x = x_new
    return path, vals, "max_iter"


OPTIMIZERS = {
    "GD": run_gd,
    "Newton": run_newton,
    "AdaGrad": run_adagrad,
    "Adam": run_adam,
}


@dataclass
class RunResult:
    role: str
    function: str
    start_id: int
    x0: float
    y0: float
    optimizer: str
    param: float
    status: str
    iterations: int
    runtime_ms: float
    x_final: float
    y_final: float
    f_final: float
    grad_norm: float
    path: list = field(default_factory=list, repr=False)
    vals: list = field(default_factory=list, repr=False)


def lecture_example():
    """min x^2 + xy + y^2 from (1,2), two steps (lecture slides)."""

    def f(x):
        return x[0] ** 2 + x[0] * x[1] + x[1] ** 2

    def g(x):
        return np.array([2 * x[0] + x[1], x[0] + 2 * x[1]], dtype=float)

    def h(x):
        return np.array([[2.0, 1.0], [1.0, 2.0]])

    x0 = np.array([1.0, 2.0])
    out = {}
    for name, fn, a in (
        ("GD", run_gd, 0.001),
        ("Newton", run_newton, 1.0),
        ("Adam", run_adam, 0.001),
    ):
        path, vals, status = fn(f, g, h, x0, a, max_iter=2, tol=0)
        out[name] = {
            "status": status,
            "x1": path[1].tolist() if len(path) > 1 else None,
            "x2": path[2].tolist() if len(path) > 2 else None,
            "f2": vals[-1],
        }
    return out


def run_suite() -> list[RunResult]:
    results: list[RunResult] = []
    for fname, spec in FUNCTIONS.items():
        for sid, x0 in STARTS.items():
            for opt_name, opt_fn in OPTIMIZERS.items():
                for alpha in PARAM_GRID[fname][opt_name]:
                    t0 = time.perf_counter()
                    path, vals, status = opt_fn(spec["f"], spec["g"], spec["h"], x0, alpha)
                    dt = (time.perf_counter() - t0) * 1000.0
                    xf = path[-1]
                    stride = max(1, len(path) // 400)
                    sp = path[::stride]
                    sv = vals[::stride]
                    if sp[-1] is not path[-1]:
                        sp.append(path[-1])
                        sv.append(vals[-1])
                    results.append(
                        RunResult(
                            role="core" if sid in (1, 2) else "extra",
                            function=fname,
                            start_id=sid,
                            x0=float(x0[0]),
                            y0=float(x0[1]),
                            optimizer=opt_name,
                            param=float(alpha),
                            status=status,
                            iterations=len(path) - 1,
                            runtime_ms=dt,
                            x_final=float(xf[0]),
                            y_final=float(xf[1]),
                            f_final=float(vals[-1]) if math.isfinite(vals[-1]) else float("nan"),
                            grad_norm=float(np.linalg.norm(spec["g"](xf))) if np.all(np.isfinite(xf)) else float("nan"),
                            path=[p.tolist() for p in sp],
                            vals=sv,
                        )
                    )
    return results


def write_csv(results: list[RunResult], path: Path):
    keys = [
        "role", "function", "start_id", "x0", "y0", "optimizer", "param",
        "status", "iterations", "runtime_ms", "x_final", "y_final", "f_final", "grad_norm",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in results:
            row = asdict(r)
            w.writerow({k: row[k] for k in keys})


def _mesh(spec, n=180):
    xs = np.linspace(*spec["xlim"], n)
    ys = np.linspace(*spec["ylim"], n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    f = spec["f"]
    for i in range(n):
        for j in range(n):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    return X, Y, Z


def _best(cands: list[RunResult]) -> RunResult:
    finite = [r for r in cands if r.status != "diverged" and math.isfinite(r.f_final)]
    pool = finite or cands
    conv = [r for r in pool if r.status == "converged"]
    pool = conv or pool
    return min(pool, key=lambda r: (abs(r.f_final) if math.isfinite(r.f_final) else 1e300, r.iterations))


CMAP = {"GD": "#c45c26", "Newton": "#2c4a6e", "AdaGrad": "#3d7a5a", "Adam": "#7a3d5c"}


def plot_function_fields():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for ax, fname in zip(axes, FUNCTIONS):
        spec = FUNCTIONS[fname]
        X, Y, Z = _mesh(spec)
        Zp = np.clip(Z, 1e-12, None) if spec["log"] else Z
        if spec["log"]:
            ax.contourf(X, Y, Zp, levels=16, cmap="cividis", norm=LogNorm())
        else:
            ax.contourf(X, Y, Zp, levels=16, cmap="cividis")
        ax.contour(X, Y, Zp, levels=10, colors="white", linewidths=0.3, alpha=0.4)
        for m in spec["minima"]:
            ax.plot(*m, marker="*", color="white", ms=11, markeredgecolor="black")
        for sid, p in STARTS.items():
            ax.plot(p[0], p[1], "o", color="#f4f0e8", ms=5, markeredgecolor="black")
            ax.annotate(f"s{sid}", (p[0], p[1]), textcoords="offset points", xytext=(4, 4), fontsize=7, color="white")
        ax.set_title(spec["title"])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(*spec["xlim"])
        ax.set_ylim(*spec["ylim"])
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Objective landscapes and the three required initializations", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "landscapes_starts.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(results: list[RunResult]):
    for fname, spec in FUNCTIONS.items():
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        X, Y, Z = _mesh(spec)
        Zp = np.clip(Z, 1e-12, None) if spec["log"] else Z
        if spec["log"]:
            ax.contourf(X, Y, Zp, levels=16, cmap="cividis", norm=LogNorm())
        else:
            ax.contourf(X, Y, Zp, levels=16, cmap="cividis")
        for m in spec["minima"]:
            ax.plot(*m, marker="*", color="white", ms=12, markeredgecolor="black", zorder=6, label="known min")
        for opt in OPTIMIZERS:
            cands = [r for r in results if r.function == fname and r.start_id == 1 and r.optimizer == opt]
            r = _best(cands)
            pts = np.array(r.path)
            ax.plot(pts[:, 0], pts[:, 1], color=CMAP[opt], lw=1.8, label=f"{opt} α={r.param:g} ({r.status})", zorder=4)
            ax.plot(pts[0, 0], pts[0, 1], "o", color=CMAP[opt], ms=5)
            ax.plot(pts[-1, 0], pts[-1, 1], "s", color=CMAP[opt], ms=5)
        ax.set_title(f"{spec['title']}  —  start (−2, 2), best α per method")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(*spec["xlim"])
        ax.set_ylim(*spec["ylim"])
        ax.legend(frameon=False, fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(FIG / f"traj_{fname}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    # combined strip for the report body
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.15))
    for ax, fname in zip(axes, FUNCTIONS):
        spec = FUNCTIONS[fname]
        X, Y, Z = _mesh(spec, n=160)
        Zp = np.clip(Z, 1e-12, None) if spec["log"] else Z
        if spec["log"]:
            ax.contourf(X, Y, Zp, levels=14, cmap="cividis", norm=LogNorm())
        else:
            ax.contourf(X, Y, Zp, levels=14, cmap="cividis")
        for opt in OPTIMIZERS:
            r = _best([u for u in results if u.function == fname and u.start_id == 1 and u.optimizer == opt])
            pts = np.array(r.path)
            ax.plot(pts[:, 0], pts[:, 1], color=CMAP[opt], lw=1.6, label=opt)
            ax.plot(pts[0, 0], pts[0, 1], "o", color=CMAP[opt], ms=3.5)
            ax.plot(pts[-1, 0], pts[-1, 1], "s", color=CMAP[opt], ms=3.5)
        ax.set_title(spec["title"], fontsize=10)
        ax.set_xlim(*spec["xlim"])
        ax.set_ylim(*spec["ylim"])
        ax.set_aspect("equal", adjustable="box")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Best-α trajectories from (−2, 2)", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "landscapes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(results: list[RunResult]):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    for ax, fname in zip(axes, FUNCTIONS):
        for opt in OPTIMIZERS:
            r = _best([u for u in results if u.function == fname and u.start_id == 1 and u.optimizer == opt])
            y = np.array(r.vals, dtype=float)
            y = np.clip(np.abs(y) + 1e-15, 1e-15, None)
            ax.plot(y, color=CMAP[opt], lw=1.6, label=opt)
        ax.set_title(FUNCTIONS[fname]["title"], fontsize=10)
        ax.set_xlabel("iteration (subsampled)")
        ax.set_ylabel("|f|")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_iters_heatmap(results: list[RunResult]):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    for ax, fname in zip(axes, FUNCTIONS):
        opts = list(OPTIMIZERS)
        means, stds = [], []
        for opt in opts:
            rs = [r.iterations for r in results if r.function == fname and r.optimizer == opt and r.role == "core"]
            means.append(float(np.mean(rs)))
            stds.append(float(np.std(rs)))
        ax.bar(opts, means, yerr=stds, color="#2c4a6e", width=0.62, ecolor="#c45c26", capsize=3)
        ax.set_title(fname)
        ax.set_ylabel("mean iters")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIG / "mean_iters.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize(results: list[RunResult], role: str = "core") -> list[dict]:
    rows = []
    for fname in FUNCTIONS:
        for opt in OPTIMIZERS:
            rs = [r for r in results if r.function == fname and r.optimizer == opt and r.role == role]
            if not rs:
                continue
            fs = np.array([r.f_final for r in rs], dtype=float)
            its = np.array([r.iterations for r in rs], dtype=float)
            ts = np.array([r.runtime_ms for r in rs], dtype=float)
            conv = sum(r.status == "converged" for r in rs)
            div = sum(r.status == "diverged" for r in rs)
            rows.append({
                "function": fname,
                "optimizer": opt,
                "n": len(rs),
                "converged": conv,
                "diverged": div,
                "mean_iters": float(np.nanmean(its)),
                "std_iters": float(np.nanstd(its)),
                "mean_f": float(np.nanmean(fs)),
                "std_f": float(np.nanstd(fs)),
                "mean_ms": float(np.nanmean(ts)),
                "std_ms": float(np.nanstd(ts)),
            })
    return rows


def main():
    print("lecture check, two steps from (1, 2):")
    lec = lecture_example()
    for k, v in lec.items():
        print(f"  {k}: x1={v['x1']}  x2={v['x2']}  f2={v['f2']:.6g}")

    print("\nrunning 72 core + 36 extra ...")
    results = run_suite()
    write_csv(results, OUT / "results.csv")
    plot_function_fields()
    plot_trajectories(results)
    plot_convergence(results)
    plot_iters_heatmap(results)
    stats = summarize(results, "core")

    print("\ncore 72 (starts 1-2):")
    for s in stats:
        print(
            f"  {s['function']:12} {s['optimizer']:8}  conv {s['converged']}/{s['n']}  "
            f"iters {s['mean_iters']:.0f}+/-{s['std_iters']:.0f}  "
            f"f {s['mean_f']:.3g}"
        )
    print("wrote", OUT / "results.csv", "and", FIG)


if __name__ == "__main__":
    main()
