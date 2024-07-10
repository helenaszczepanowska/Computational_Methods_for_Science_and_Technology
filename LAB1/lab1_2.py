import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

def sequence(n, precision):
    x = np.zeros(n, dtype=precision)
    x[0] = precision(1/3)
    x[1] = precision(1/12)
    for k in range(1, n-1):
        x[k+1] = precision(2.25 * x[k]) - precision(0.5 * x[k-1])
    
    return x

def fraction_sequence(n):
    x = [Fraction(0) for _ in range(n)]
    x[0] = Fraction(1, 3)
    x[1] = Fraction(1, 12)

    for k in range(1, n-1):
        x[k+1] = Fraction(9, 4) * x[k] - Fraction(1, 2) * x[k-1]

    return x

def solution(n):
    solution = [0]*n
    for k in range(n):
        solution[k] = 4**(-k)/3

    return solution

# Obliczanie ciągów
n_single = 225
n_double = 60
n_fraction = 225

x_single = sequence(n_single, np.single)
x_double = sequence(n_double, np.double)
x_fraction = fraction_sequence(n_fraction)

# Obliczanie błędu względnego
solution_values_single = solution(n_single)
solution_values_double = solution(n_double)
solution_values_fraction = solution(n_fraction)

errors_single = np.abs((x_single - solution_values_single) / solution_values_single)
errors_double = np.abs((x_double - solution_values_double) / solution_values_double)
errors_fraction = [abs((x - sol) / sol) for x, sol in zip(x_fraction, solution_values_fraction)]

# Wykres wartości ciągu
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) 

plt.semilogy(range(n_fraction), x_fraction, label='Fraction Representation')
plt.semilogy(range(n_double), x_double, label='Double Precision')
plt.semilogy(range(n_single), x_single, label='Single Precision')
plt.xlabel('k')
plt.ylabel('Value')
plt.title('Sequence values by different precisions')
plt.legend()

# Wykres błędów względnych
plt.subplot(1, 2, 2) 

plt.semilogy(range(n_fraction), errors_fraction, label='Fraction Representation Error')
plt.semilogy(range(n_double), errors_double, label='Double Precision Error')
plt.semilogy(range(n_single), errors_single, label='Single Precision Error')
plt.xlabel('k')
plt.ylabel('Relative Error')
plt.title('Relative errors by different precisions')
plt.legend()
plt.show()



