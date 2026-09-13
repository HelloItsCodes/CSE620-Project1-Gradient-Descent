import sys
import numpy as np
from CSE620_Fall26_Project1_Nguyen_Rajkumar import (
    FUNCTIONS,
    lecture_example,
    run_gd,
    run_newton,
)


def clip0():
    print(list(FUNCTIONS))


def clip1():
    for name, v in lecture_example().items():
        print(name, v["x1"], v["x2"], v["f2"])


def clip2():
    spec = FUNCTIONS["quadratic"]
    path, vals, st = run_newton(spec["f"], spec["g"], spec["h"], np.array([-2.0, 2.0]), 1.0)
    print(st, len(path) - 1, path[-1], vals[-1])


def clip3():
    spec = FUNCTIONS["quadratic"]
    path, vals, st = run_gd(spec["f"], spec["g"], spec["h"], np.array([-2.0, 2.0]), 0.1)
    print(st, len(path) - 1, path[-1], vals[-1])


def clip4():
    spec = FUNCTIONS["rosenbrock"]
    path, vals, st = run_gd(spec["f"], spec["g"], spec["h"], np.array([-2.0, 2.0]), 0.1)
    print(st, len(path) - 1, path[-1], vals[-1])


def clip5():
    spec = FUNCTIONS["rosenbrock"]
    path, vals, st = run_newton(spec["f"], spec["g"], spec["h"], np.array([-2.0, 2.0]), 1.0)
    print(st, len(path) - 1, path[-1], vals[-1])


def clip6():
    spec = FUNCTIONS["cosine"]
    path, vals, st = run_gd(spec["f"], spec["g"], spec["h"], np.array([-2.0, 2.0]), 0.1)
    print(st, len(path) - 1, path[-1], vals[-1])


def clip7():
    spec = FUNCTIONS["cosine"]
    path, vals, st = run_newton(spec["f"], spec["g"], spec["h"], np.array([0.5, -1.5]), 1.0)
    print(st, len(path) - 1, path[-1], vals[-1])


CLIPS = {
    "0": clip0,
    "1": clip1,
    "2": clip2,
    "3": clip3,
    "4": clip4,
    "5": clip5,
    "6": clip6,
    "7": clip7,
}

if __name__ == "__main__":
    CLIPS[sys.argv[1]]()
