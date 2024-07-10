import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz, simps, quad, quad_vec

# Definicje funkcji podcałkowych
def f1(x):
    return np.where(x > 0, np.sqrt(x) * np.log(x), 0)


def f2(x):
    term1 = 1 / ((x - 0.3)**2 + 0.001)
    term2 = 1 / ((x - 0.9)**2 + 0.004)
    return term1 + term2 - 6

def f2_exact_value(x0, x1, a, b):
    term1 = (np.arctan((1 - x0) / np.sqrt(a)) + np.arctan(x0 / np.sqrt(a))) / np.sqrt(a)
    term2 = (np.arctan((1 - x1) / np.sqrt(b)) + np.arctan(x1 / np.sqrt(b))) / np.sqrt(b)
    return term1 + term2 - 6


# Funkcja obliczająca wartość błędu względnego
def relative_error(exact, approx):
    return np.abs((exact - approx) / exact)

# Funkcja obliczająca wartości całek i błędów dla różnych metod

def count_intergrals(f, a, b, exact_value, title):

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
        error_trapezoidal = relative_error(exact_value, integral_trapezoidal)
        errors_trapezoidal.append(error_trapezoidal)
        
        # Metoda Simpsona
        integral_simpson = simps(y, x)
        error_simpson = relative_error(exact_value, integral_simpson)
        errors_simpson.append(error_simpson)
        
        # Metoda punktu środkowego
        x_mid = (x[1:] + x[:-1]) / 2
        y_mid = f(x_mid)
        integral_midpoint = np.sum(y_mid * (x[1:] - x[:-1]))
        error_midpoint = relative_error(exact_value, integral_midpoint)
        errors_midpoint.append(error_midpoint)

    print(integral_trapezoidal,integral_simpson,integral_midpoint)

    n_values = 2**m_values +1

    # Wyliczenie wartości całek i błędów dla metod adaptacyjnych trapezów i Gaussa-Kronroda dla tolerancji od 1 do 10^-14
    a, b = 0, 1 
    tolerances = np.logspace(0, -14, num=15, base=10)
    results_trap = []
    results_gauss = []

    for tol in tolerances:

        integral_trap, err, info = quad_vec(f, a, b, epsabs=tol, norm='max', quadrature='trapezoid', full_output=True)
        relative_trap = relative_error(exact_value, integral_trap)
        results_trap.append((tol, info.neval, relative_trap))
        
        integral_gauss, err, info = quad_vec(f, a, b, epsabs=tol, norm='max', quadrature='gk21', full_output=True)
        relative_gauss = relative_error(exact_value, integral_gauss)
        results_gauss.append((tol, info.neval, relative_gauss))
        

    evals_trap = [result[1] for result in results_trap]
    errors_trap = [result[2] for result in results_trap]

    evals_gauss = [result[1] for result in results_gauss]
    errors_gauss = [result[2] for result in results_gauss]

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
    plt.title('Relative errors for ' + title)
    plt.legend()
    plt.grid(True)
    plt.show()


a, b = 0, 1
exact_value_f1 = quad(f1, a, b)[0]
exact_value_f2 = f2_exact_value(0.3, 0.9, 0.001, 0.004)
count_intergrals(f1, a, b, exact_value_f1, 'f(x) = sqrt(x)*log(x)' )
count_intergrals(f2, a, b, exact_value_f2, 'f(x) = 1 / ((x - 0.3)^2 + 0.001) + 1 / ((x - 0.9)^2 + 0.004) - 6')