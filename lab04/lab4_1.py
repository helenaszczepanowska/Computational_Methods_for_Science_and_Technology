import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
import numpy.random as npr

# Definicje funkcji
def f1(x):
    return 1 / (1 + 25 * x**2)

def f2(x):
    return np.exp(np.cos(x))

# Interpolacja Lagrange'a 
def lagrange_interpolation(x_points, y_points, x):
    n = len(x_points)
    sum = 0
    for i in range(n):
        product = 1
        for j in range(n):
            if i != j:
                product *= (x - x_points[j]) / (x_points[i] - x_points[j])
        sum += y_points[i] * product
    return sum

# Węzły równoodległe i Czebyszewa
def equidistant_nodes(a, b, n):
    return np.linspace(a, b, n)

def chebyshev_nodes(a, b, n):

    theta = np.pi * (2*np.arange(n) + 1) / (2*(n))
    x = np.cos(theta)
    x_transformed = a + (b - a) * (x + 1) / 2
    
    return x_transformed

n = 12
a, b = -1, 1
x_2 = np.linspace(a, b, 10*n)  # Gęstsza siatka dla wykresu
x_eq = equidistant_nodes(a, b, n)
x_cheb = chebyshev_nodes(a, b, n)

y_eq = f1(x_eq)
y_cheb = f1(x_cheb)

# Interpolacja
y_lagrange_eq = [lagrange_interpolation(x_eq, y_eq, x) for x in x_2]
y_lagrange_cheb = [lagrange_interpolation(x_cheb, y_cheb, x) for x in x_2]

# Kubiczne funkcje sklejane (dla węzłów równoodległych)
cubic_spline = interp1d(x_eq, y_eq, kind='cubic', fill_value="extrapolate")
y_cubic_spline = cubic_spline(x_2)

# Rysowanie wykresów
plt.figure(figsize=(12, 8))
plt.plot(x_2, f1(x_2), label='Funkcja Rungego', linewidth=2)
plt.plot(x_2, y_lagrange_eq, label='Interpolacja Lagrange’a (równoodległe)')
plt.plot(x_2, y_lagrange_cheb, label='Interpolacja Lagrange’a (Czebyszewa)')
plt.plot(x_2, y_cubic_spline, label='Kubiczne funkcje sklejane', linewidth=2)
plt.scatter(x_eq, y_eq, color='red', label='Węzły równoodległe', zorder=5)
plt.scatter(x_cheb, y_cheb, color='green', label='Węzły Czebyszewa', zorder=5)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Porównanie metod interpolacji dla funkcji Rungego')
plt.show()

# Podpunkt b
n_values = range(4, 51)
x_random = npr.uniform(-1, 1, 500) # Dla f1(x)
x_random_f2 = npr.uniform(0, 2 * np.pi, 500)  # Dla f2(x)
x_random.sort()
x_random_f2.sort()
errors_f1 = {'Lagrange Equi': [], 'Lagrange Cheb': [], 'Cubic Spline': []}
errors_f2 = {'Lagrange Equi': [], 'Lagrange Cheb': [], 'Cubic Spline': []}

# Obliczanie błędów dla n = 4,5,...,50 węzłów

for n in n_values:
    # Węzły równoodległe i Czebyszewa
    x_eq_f1 = np.linspace(-1, 1, n)
    x_cheb_f1 = chebyshev_nodes(-1, 1, n)
    x_eq_f2 = np.linspace(0, 2 * np.pi, n)
    x_cheb_f2 = chebyshev_nodes(0, 2 * np.pi, n)

    # Wartości funkcji
    y_eq_f1 = f1(x_eq_f1)
    y_cheb_f1 = f1(x_cheb_f1)
    y_eq_f2 = f2(x_eq_f2)
    y_cheb_f2 = f2(x_cheb_f2)

    # Interpolacje
    lagrange_eq_f1 = [lagrange_interpolation(x_eq_f1, y_eq_f1, x) for x in x_random]
    lagrange_cheb_f1 = [lagrange_interpolation(x_cheb_f1, y_cheb_f1, x) for x in x_random]
    cubic_spline_f1 = interp1d(x_eq_f1, y_eq_f1, kind='cubic', fill_value="extrapolate")
    y_cubic_spline_f1 = cubic_spline_f1(x_random)

    lagrange_eq_f2 = [lagrange_interpolation(x_eq_f2, y_eq_f2, x) for x in x_random_f2]
    lagrange_cheb_f2 = [lagrange_interpolation(x_cheb_f2, y_cheb_f2, x) for x in x_random_f2]
    cubic_spline_f2 = interp1d(x_eq_f2, y_eq_f2, kind='cubic', fill_value="extrapolate")
    y_cubic_spline_f2 = cubic_spline_f2(x_random_f2)

    # Błędy
    errors_f1['Lagrange Equi'].append(np.linalg.norm(f1(x_random) - lagrange_eq_f1))
    errors_f1['Lagrange Cheb'].append(np.linalg.norm(f1(x_random) - lagrange_cheb_f1))
    errors_f1['Cubic Spline'].append(np.linalg.norm(f1(x_random) - y_cubic_spline_f1))

    errors_f2['Lagrange Equi'].append(np.linalg.norm(f2(x_random_f2) - lagrange_eq_f2))
    errors_f2['Lagrange Cheb'].append(np.linalg.norm(f2(x_random_f2) - lagrange_cheb_f2))
    errors_f2['Cubic Spline'].append(np.linalg.norm(f2(x_random_f2) - y_cubic_spline_f2))

# Wykresy błędów
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for method, err in errors_f1.items():
    plt.plot(n_values, err, label=method)
plt.yscale("log")
plt.title('Błąd interpolacji dla f1(x)')
plt.xlabel('Liczba węzłów n')
plt.ylabel('Norma błędu')
plt.legend()

plt.subplot(1, 2, 2)
for method, err in errors_f2.items():
    plt.plot(list(n_values), err, label=method)
plt.yscale("log")
plt.title('Błąd interpolacji dla f2(x)')
plt.xlabel('Liczba węzłów n')
plt.legend()

plt.tight_layout()
plt.show()