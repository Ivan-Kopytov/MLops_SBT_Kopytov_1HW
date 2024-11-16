from simple_linear_regression import linear_regression
import numpy as np

# Данные
X = [1, 2, 3, 4, 5]
Y = [2.1, 2.9, 4.2, 4.8, 5.6]

# Результаты вашей реализации
b0, b1 = linear_regression(X, Y)
print(f"Your implementation: Intercept (b0) = {b0}, Slope (b1) = {b1}")

# Результаты numpy
X_np = np.array(X)
Y_np = np.array(Y)
A = np.vstack([X_np, np.ones(len(X_np))]).T
b1_np, b0_np = np.linalg.lstsq(A, Y_np, rcond=None)[0]
print(f"Numpy implementation: Intercept (b0) = {b0_np}, Slope (b1) = {b1_np}")
