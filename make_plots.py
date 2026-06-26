"""Generate the teaching graphs embedded in the README.

Reproduces the exact training in train.py (z-score normalization, batch
gradient descent, alpha=0.01, 1000 iterations) but also records the cost at
each step so we can visualize convergence. Saves three PNGs into assets/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA, ITERS = 0.01, 1000
os.makedirs("assets", exist_ok=True)

# --- data ---
data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
x_raw = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

mean_x, std_x = np.mean(x_raw), np.std(x_raw)
x_norm = (x_raw - mean_x) / std_x
X = np.hstack((x_norm, np.ones(x_norm.shape)))   # [x_norm, 1]


def cost(X, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2)


# --- gradient descent, tracking cost history ---
theta = np.zeros((2, 1))
history = []
m = len(y)
for _ in range(ITERS):
    history.append(cost(X, y, theta))
    grad = (1 / m) * X.T.dot(X.dot(theta) - y)
    theta = theta - ALPHA * grad

slope, intercept = theta[0, 0], theta[1, 0]   # on normalized scale

# 1) Raw data scatter
plt.figure(figsize=(7, 5))
plt.scatter(x_raw, y, color="#2563eb", alpha=0.8, edgecolors="white")
plt.title("Training data: price vs. mileage")
plt.xlabel("Mileage (km)")
plt.ylabel("Price")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("assets/data_scatter.png", dpi=110)
plt.close()

# 2) Fitted regression line over the data (mapped back to original scale)
xs = np.linspace(x_raw.min(), x_raw.max(), 100)
ys = slope * ((xs - mean_x) / std_x) + intercept
plt.figure(figsize=(7, 5))
plt.scatter(x_raw, y, color="#2563eb", alpha=0.8, edgecolors="white", label="data")
plt.plot(xs, ys, color="#dc2626", linewidth=2.5, label="learned line")
plt.title("Learned linear model")
plt.xlabel("Mileage (km)")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("assets/regression_fit.png", dpi=110)
plt.close()

# 3) Cost vs iterations (gradient descent converging)
plt.figure(figsize=(7, 5))
plt.plot(range(ITERS), history, color="#059669", linewidth=2)
plt.title(r"Gradient descent: cost $J(\theta)$ per iteration")
plt.xlabel("Iteration")
plt.ylabel(r"Cost  $J(\theta)$")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("assets/cost_curve.png", dpi=110)
plt.close()

print(f"slope(norm)={slope:.4f} intercept={intercept:.4f}")
print(f"cost: start={history[0]:.1f} end={history[-1]:.1f}")
print("wrote assets/data_scatter.png, assets/regression_fit.png, assets/cost_curve.png")
