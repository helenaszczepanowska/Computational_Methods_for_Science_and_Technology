import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz, simps

# Definicja funkcji podcałkowej
def f(x):
    return 4 / (1 + x**2)

# Funkcja obliczająca wartość błędu względnego
def relative_error(exact, approx):
    return np.abs((exact - approx) / exact)

# Zakres wartości m
m_values = np.arange(1, 26)

# Listy przechowujące błędy dla każdej metody
errors_trapezoidal = []
errors_simpson = []
errors_midpoint = []

# Pętla obliczająca wartości całek i błędów dla różnych wartości m
for m in m_values:
    n = 2 ** m + 1
    x = np.linspace(0, 1, n)
    y = f(x)
    
    # Metoda trapezów
    integral_trapezoidal = trapz(y, x)
    error_trapezoidal = relative_error(np.pi, integral_trapezoidal)
    errors_trapezoidal.append(error_trapezoidal)
    
    # Metoda Simpsona
    integral_simpson = simps(y, x)
    error_simpson = relative_error(np.pi, integral_simpson)
    errors_simpson.append(error_simpson)
    
    # Metoda punktu środkowego
    x_mid = (x[1:] + x[:-1]) / 2
    y_mid = f(x_mid)
    integral_midpoint = np.sum(y_mid * (x[1:] - x[:-1]))
    error_midpoint = relative_error(np.pi, integral_midpoint)
    errors_midpoint.append(error_midpoint)

print(integral_trapezoidal,integral_simpson,integral_midpoint)
n_values = 2**m_values +1

# Rysowanie wykresu
plt.figure(figsize=(10, 6))
plt.plot(n_values, errors_trapezoidal, label='Trapezoidal Rule')
plt.plot(n_values, errors_simpson, label='Simpson\'s Rule')
plt.plot(n_values, errors_midpoint, label='Mid-point Rule')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of evaluations')
plt.ylabel('Absolute relative error')
plt.title('Convergence of Numerical Integration Methods')
plt.legend()
plt.grid(True)
plt.show()



# Obliczenie h_min

def calculate_hmin(method):
    h = 1.0
    previous_error = 1.0

    while True:
        x = np.linspace(0, 1, int(1 / h) + 1)
        y = f(x)

        exact_value = np.pi
        integral_value = method(y, x)
        error = relative_error(exact_value, integral_value)

        if error >= previous_error or np.isnan(error):
            break

        previous_error = error
        h /= 2

    return h

h_min_trapezoidal = calculate_hmin(trapz)
print("H_min dla metody trapezow wynosi:",h_min_trapezoidal)
h_min_Simpson = calculate_hmin(simps)
print("H_min dla metody Simpsona wynosi:",h_min_Simpson)


def calculate_error(method, h):
    x = np.linspace(0, 1, int(1 / h) + 1)
    y = f(x)

    exact_value = np.pi
    integral_value = method(y, x)

    error = np.abs((exact_value - integral_value) / exact_value)

    return error


def calculate_convergence_order(errors, hs):
    p_values = []

    for i in range(len(errors) - 1):
        if errors[i] == 0 or errors[i+1] == 0:
            continue
        p = np.log(errors[i+1] / errors[i]) / np.log(hs[i+1] / hs[i])
        p_values.append(p)

    return p_values

hs = np.logspace(-5, -1, 100)

errors_trapezoidal_empi = [calculate_error(trapz, h) for h in hs]
errors_simpson_empi = [calculate_error(simps, h) for h in hs]

p_values_trapezoidal = calculate_convergence_order(errors_trapezoidal_empi, hs)
p_values_Simpson = calculate_convergence_order(errors_simpson_empi, hs)

print("Rząd zbieznosci dla metody trapezów: ",np.mean(p_values_trapezoidal))
print("Rząd zbieznosci dla metody Simpsona: ",np.mean(p_values_Simpson))