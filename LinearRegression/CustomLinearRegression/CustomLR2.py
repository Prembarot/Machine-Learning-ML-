import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load features
X = np.array([8, 10, 11, 14, 5])

# Load labels
y = np.array([4, 2, 21, 31, 23])

# Optional: visualize data
# plt.scatter(X, y, color="black")
# plt.xlabel("FEATURES", color="green", fontsize=20)
# plt.ylabel("LABELS", color="red", fontsize=20)
# plt.show()

## HYPER PARAMETERS
learning_rate = 0.00008
max_itr = 1000

# Hypothesis function
def h(X, m, b):
    return m * X + b

# Gradient calculation
def gradient(X, y, m, b):
    y_hat = h(X, m, b)
    dm = np.average((y_hat - y) * X)
    db = np.average(y_hat - y)
    return dm, db

# Loss function (MSE)
def loss(X, y, m, b):
    y_hat = h(X, m, b)
    return np.average(np.square(y - y_hat))

# Gradient Descent
def gradient_descent(X, y, learning_rate, max_itr):
    m = 0
    b = 0
    for i in range(max_itr):
        dm, db = gradient(X, y, m, b)
        m -= learning_rate * dm
        b -= learning_rate * db
        loss_value = loss(X, y, m, b)
        print("loss_value =", loss_value)
    return m, b

# Train model
m, b = gradient_descent(X, y, learning_rate, max_itr)

print("m =", m)
print("b =", b)