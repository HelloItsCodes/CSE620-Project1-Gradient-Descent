"""Run one clip at a time for the video.

    python3 demo.py 0
    python3 demo.py 1
    ...
    python3 demo.py 7
"""
import sys
import numpy as np
from CSE620_Fall26_Project1_Nguyen_Rajkumar import (
    FUNCTIONS,
    lecture_example,
    run_gd,
    run_newton,
)


def clip0():
    print("loaded GD, Newton, AdaGrad, Adam")
    print("functions:", list(FUNCTIONS))


def clip1():
    out = lecture_example()
    for name, v in out.items():
        print(name, "x1=", v["x1"], "x2=", v["x2"], "f2=", v["f2"])


def clip2():
    spec = FUNCTIONS["quadratic"]
    path, vals, st = run_newton(spec["f"], spec["g"], spec["h"], np.array([-2.0, 2.0]), 1.0)
    print(st, "iters", len(path) - 1, "x", path[-1], "f", vals[-1])


def clip3():
    spec = FUNCTIONS["quadratic"]
    path, vals, st = run_gd(spec["f"], spec["g"], spec["h"], np.array([-2.0, 2.0]), 0.1)
    print(st, "iters", len(path) - 1, "x", path[-1], "f", vals[-1])


def clip4():
    spec = FUNCTIONS["rosenbrock"]
    path, vals, st = run_gd(spec["f"], spec["g"], spec["h"], np.array([-2.0, 2.0]), 0.1)
    print(st, "iters", len(path) - 1, "x", path[-1], "f", vals[-1])


def clip5():
    spec = FUNCTIONS["rosenbrock"]
    path, vals, st = run_newton(spec["f"], spec["g"], spec["h"], np.array([-2.0, 2.0]), 1.0)
    print(st, "iters", len(path) - 1, "x", path[-1], "f", vals[-1])


def clip6():
    spec = FUNCTIONS["cosine"]
    path, vals, st = run_gd(spec["f"], spec["g"], spec["h"], np.array([-2.0, 2.0]), 0.1)
    print(st, "iters", len(path) - 1, "x", path[-1], "f", vals[-1])


def clip7():
    spec = FUNCTIONS["cosine"]
    path, vals, st = run_newton(spec["f"], spec["g"], spec["h"], np.array([0.5, -1.5]), 1.0)
    print(st, "iters", len(path) - 1, "x", path[-1], "f", vals[-1])


CLIPS = {
    "0": ("setup", clip0),
    "1": ("lecture example", clip1),
    "2": ("Newton on the bowl", clip2),
    "3": ("GD on the bowl", clip3),
    "4": ("GD Rosenbrock blows up", clip4),
    "5": ("Newton Rosenbrock to (1,1)", clip5),
    "6": ("GD cosine local min", clip6),
    "7": ("Newton cosine diverges", clip7),
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CLIPS:
        print("python3 demo.py N")
        for k, (title, _) in CLIPS.items():
            print(" ", k, title)
        sys.exit(0)
    n = sys.argv[1]
    print("clip", n, "-", CLIPS[n][0])
    CLIPS[n][1]()
