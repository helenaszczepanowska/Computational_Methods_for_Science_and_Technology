import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz, simps, quad, quad_vec

# Definicja funkcji podcałkowych
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
integral = []

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
    integral.append([integral_trapezoidal,integral_simpson,integral_midpoint])

print(integral_trapezoidal,integral_simpson,integral_midpoint)

n_values = 2**m_values +1

# Wyliczenie wartości całek i błędów dla metod adaptacyjnych trapezów i Gaussa-Kronroda dla tolerancji od 1 do 10^-14
a, b = 0, 1 
tolerances = np.logspace(0, -14, num=15, base=10)
results_trap = []
results_gauss = []
exact_value = np.pi

for tol in tolerances:

    integral_trap, err, info = quad_vec(f, a, b, epsabs=tol, norm='max', quadrature='trapezoid', full_output=True)
    relative_trap = relative_error(exact_value, integral_trap)
    results_trap.append((tol, info.neval, relative_trap, integral_trap))

    integral_gauss, err, info = quad_vec(f, a, b, epsabs=tol, norm='max', quadrature='gk21', full_output=True)
    relative_gauss = relative_error(exact_value, integral_gauss)
    results_gauss.append((tol, info.neval, relative_gauss, integral_gauss))

evals_trap = [result[1] for result in results_trap]
errors_trap = [result[2] for result in results_trap]

evals_gauss = [result[1] for result in results_gauss]
errors_gauss = [result[2] for result in results_gauss]

integral_trap = [result[3] for result in results_trap]
integral_gauss = [result[3] for result in results_gauss]

# Rysowanie wykresu
plt.figure(figsize=(10, 6))
plt.plot(n_values, errors_trapezoidal, label='Trapezoidal Rule')
plt.plot(n_values, errors_simpson, label='Simpson\'s Rule')
plt.plot(n_values, errors_midpoint, label='Mid-point Rule')
plt.plot(evals_trap, errors_trap, label='Adaptive Trapezoidal Rule')
plt.plot(evals_gauss, errors_gauss, label='Adaptive Gauss-Kronrod Rule')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of evaluations')
plt.ylabel('Relative error')
plt.title('Relative errors for f(x) = 4 / (1 + x^2)')
plt.legend()
plt.grid(True)
plt.show()

