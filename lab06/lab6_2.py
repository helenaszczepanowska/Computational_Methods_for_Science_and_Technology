import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

# Funkcja podcałkowa
def f(x):
    return 4 / (1 + x**2)

# Metoda Gaussa-Legendre'a
def gauss_legendre_integration(n):
    # Węzły i wagi Legendre'a dla n punktów
    x, w = roots_legendre(n)
    
    # Skalowanie węzłów do przedziału [0, 1]
    x_scaled = 0.5 * (x + 1)
    w_scaled = 0.5 * w
    
    # Obliczanie wartości całki
    integral = np.sum(w_scaled * f(x_scaled))
    return integral

# Dokładna wartość całki
exact_integral = np.pi

# Liczba punktów ewaluacji funkcji podcałkowej
n_values = np.arange(1, 100)

# Lista przechowująca wartości bezwzględnych błędów względnych
errors_gauss_legendre = []
numerical_errors = []
method_errors = []

# Obliczanie błędów dla różnych wartości n
for n in n_values:
    integral_approx = gauss_legendre_integration(n)

    numerical_error = np.abs(exact_integral - integral_approx)
    numerical_errors.append(numerical_error)
    
    # Obliczanie błędu metody (błąd bezwzględny względem dokładnej wartości całki)
    method_error = np.abs((exact_integral - integral_approx))
    method_errors.append(method_error)
    

    
    error = np.abs((exact_integral - integral_approx) / exact_integral)
    errors_gauss_legendre.append(error)

# Rysowanie wykresu wartości bezwzględnej błędu względnego
plt.plot(n_values + 1, errors_gauss_legendre, label='Gauss-Legendre Integration')
plt.plot(n_values + 1, numerical_errors, label='Numerical Error')
# plt.plot(n_values + 1, method_errors, label='Method Error')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of evaluations')
plt.ylabel('Absolute relative error')
plt.title('Convergence of Gauss-Legendre Integration')
plt.legend()
plt.grid(True)
plt.show()
