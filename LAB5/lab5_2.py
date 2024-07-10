import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def rescale(x):
    return (x + 1)

def f_rescaled(x):
    return np.sqrt(rescale(x))

def weight_function(x):
    return (1 - x**2)**(-1/2)

def chebyshev_coefficient(f, n):
    integrand = lambda x: f(x) * np.cos(n * np.arccos(x)) * weight_function(x)
    coefficient = (2 / np.pi) * quad(integrand, -1, 1)[0] if n > 0 else (1 / np.pi) * quad(integrand, -1, 1)[0]
    print(coefficient)
    return coefficient


a0 = chebyshev_coefficient(f_rescaled, 0)
a1 = chebyshev_coefficient(f_rescaled, 1)
a2 = chebyshev_coefficient(f_rescaled, 2)

def approximated_function(x):
    return a0 + a1 * np.cos(np.arccos(x)) + a2 * np.cos(2 * np.arccos(x))


x_values = np.linspace(-1, 1, 100)
y_true = f_rescaled(x_values)
y_approx = approximated_function(x_values)





plt.figure(figsize=(10, 5))
plt.plot(x_values, y_true, label='Original Function', color='blue')
plt.plot(x_values, y_approx, label='Chebyshev Approximation', color='red', linestyle='--')
plt.title('Comparison of Original Function and Chebyshev Approximation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()


