import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.tan(x)

def f_prime(x):
    return 1 + np.tan(x)**2

def f_double_prime(x):
    return 2 * np.tan(x) * (1 + np.tan(x)**2)

def f_triple_prime(x):
    return 2 * (1 + np.tan(x)**2) * (1 + 3*np.tan(x)**2)

def numerical_derivative(x, h):
    return (f(x + h) - f(x)) / h

def central_difference(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Obliczenie pochodnej numerycznej i błędów obliczeniowych
h_values = np.logspace(-16, 0, num=17, base=10)
numerical_derivatives = [0]*17
computional_errors = [0]*17
numerical_derivatives_central = [0]*17
computional_errors_central = [0]*17
i = 0
x0 = 1
for h in h_values:
    numerical_derivatives[i] = numerical_derivative(x0, h)
    numerical_derivatives_central[i] = central_difference(x0, h)
    true_derivative = f_prime(x0)
    computional_errors[i] = np.abs(numerical_derivatives[i] - true_derivative)
    computional_errors_central[i] = np.abs(numerical_derivatives_central[i] - true_derivative)
    i += 1

# Obliczenie błędu maszynowego epsilon
epsilon_machine = np.finfo(float).eps
M = np.abs(f_double_prime(x0))
M_central = np.abs(f_triple_prime(x0))

# Obliczenie i porównanie h minimalnego
h_min = 2 * np.sqrt(epsilon_machine / M)
h_min_computational = min(computional_errors)
h_min_index = computional_errors.index(h_min_computational)
h_min_difference = np.abs(h_min - h_min_computational)
print("Difference of h_min in forward difference method" , h_min_difference)
print("h_min x: ", h_values[h_min_index], "y: ", h_min_computational)

h_min_central = np.cbrt(3 * epsilon_machine / M_central)
h_min_computational_central = min(computional_errors_central)
h_min_index_central = computional_errors_central.index(h_min_computational_central)
h_min_difference_central = np.abs(h_min_central - h_min_computational_central)
print("Difference of h_min in central difference method" , h_min_difference_central)
print("h_min_central x: ", h_values[h_min_index_central], "y: ", h_min_computational_central)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) 

# Wykres błędu obliczeniowego
plt.loglog(h_values, computional_errors, label='Computional Error', marker='o')

# Wykres błędu metody
truncation_errors = 0.5 * M * h_values
plt.loglog(h_values, truncation_errors, label='Truncation Error', marker='o')

# Wykres błędu numerycznego
rounding_errors = 2 * epsilon_machine / h_values
plt.loglog(h_values, rounding_errors, label='Rounding Error', marker='o')

plt.xlabel('h')
plt.ylabel('Error')
plt.title('Errors for forward difference method of tan(x)')
plt.legend()

plt.subplot(1, 2, 2)

# Wykres błędu obliczeniowego
plt.loglog(h_values, computional_errors_central, label='Computional Error', marker='o')

# Wykres błędu metody
truncation_errors_central = (M_central * h_values**2)/6
plt.loglog(h_values, truncation_errors_central, label='Truncation Error', marker='o')

# Wykres błędu numerycznego
rounding_errors = epsilon_machine / h_values
plt.loglog(h_values, rounding_errors, label='Rounding Error', marker='o')

plt.xlabel('h')
plt.ylabel('Error')
plt.title('Errors for central difference method of tan(x)')
plt.legend()

# plt.figtext(0.5, 0.01, f"Difference of h_min in forward difference method: {h_min_difference:.2e}, Difference of h_min in central difference method: {h_min_difference_central:.2e} ", 
# ha="center", fontsize=12, bbox={"facecolor":"blue", "alpha":0.5, "pad":5})
plt.show()
