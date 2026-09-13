import numpy as np
import matplotlib.pyplot as plt
from CSE620_Fall26_Project1_Nguyen_Rajkumar import FUNCTIONS, OPTIMIZERS

name = input("function (quadratic / rosenbrock / cosine): ").strip()
method = input("method (GD / Newton / AdaGrad / Adam): ").strip()
x0 = float(input("x0: "))
y0 = float(input("y0: "))
alpha = float(input("alpha: "))

spec = FUNCTIONS[name]
opt = OPTIMIZERS[method]
x_start = np.array([x0, y0])
path, vals, status = opt(spec["f"], spec["g"], spec["h"], x_start, alpha)
pts = np.array(path)

print("status:", status)
print("iterations:", len(path) - 1)
print("final x,y:", pts[-1])
print("final f:", vals[-1])

xs = np.linspace(spec["xlim"][0], spec["xlim"][1], 200)
ys = np.linspace(spec["ylim"][0], spec["ylim"][1], 200)
X, Y = np.meshgrid(xs, ys)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = spec["f"](np.array([X[i, j], Y[i, j]]))

plt.figure(figsize=(6, 5))
plt.contour(X, Y, Z, levels=25)
plt.plot(pts[:, 0], pts[:, 1], "r.-", label=method)
plt.plot(pts[0, 0], pts[0, 1], "go", label="start")
plt.plot(pts[-1, 0], pts[-1, 1], "bs", label="end")
plt.xlim(*spec["xlim"])
plt.ylim(*spec["ylim"])
plt.title(name + "  " + method + "  a=" + str(alpha))
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
