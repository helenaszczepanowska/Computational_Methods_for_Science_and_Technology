import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative 

def f(x):
    return x ** 2 - 3 * x + 2

def g1(x):
    return (x ** 2 + 2) / 3

def g2(x):

    return np.sqrt(3 * x - 2) if 3 * x - 2 >= 0 else None

def g3(x):
    return 3 - 2 / x if x != 0 else None

def g4(x):
    return (x ** 2 - 2) / (2 * x - 3) if 2 * x - 3 != 0 else None

x_0 = 2
prime_value = [derivative(g1, x_0), derivative(g2, x_0), derivative(g3, x_0), derivative(g4, x_0)]
print(np.round(prime_value, 2))

def iterate(g, initial_value, true_value, n):
    x = initial_value
    errors = []
    for i in range(10):
        x = g(x)
        errors.append(abs(x - true_value)/true_value)
    
    return errors

initial_value = 1.6
true_value = 2
n = 10

errors_g1 = iterate(g1, initial_value, true_value, n)
errors_g2 = iterate(g2, initial_value, true_value, n)
errors_g3 = iterate(g3, initial_value, true_value, n)
errors_g4= iterate(g4, initial_value, true_value, n)

def convergence_rate(errors):
    rates = []
    for k in range(1, len(errors) - 1):
        if errors[k-1] != 0 and errors[k] != 0 and errors[k+1] != 0:
            r = np.log(errors[k] / errors[k+1]) / np.log(errors[k-1] / errors[k])
            rates.append(r)
    return rates

rates_g1 = convergence_rate(errors_g1)
rates_g2 = convergence_rate(errors_g2)
rates_g3 = convergence_rate(errors_g3)
rates_g4 = convergence_rate(errors_g4)

average_rate_g1 = np.mean(rates_g1)
average_rate_g2 = np.mean(rates_g2)
average_rate_g3 = np.mean(rates_g3)
average_rate_g4 = np.mean(rates_g4)


print("Wartość rzędu zbieżności dla g1(x):", round(rates_g1[-1], 2))
print("Wartość rzędu zbieżności dla g2(x):", round(rates_g2[-1], 2))
print("Wartość rzędu zbieżności dla g3(x):", round(rates_g3[-1], 2))
print("Wartość rzędu zbieżności dla g4(x):", round(rates_g4[-1], 2))

iterations = np.array(range(0,n))
plt.figure(figsize=(10, 6))
plt.plot(iterations, errors_g1, label='g1(x)')
plt.plot(iterations, errors_g2, label='g2(x)')
plt.plot(iterations, errors_g3, label='g3(x)')
plt.plot(iterations, errors_g4, label='g4(x)')
plt.title('Błąd względny w zależności od ilości iteracji')
plt.xlabel('Numer iteracji')
plt.ylabel('Błąd względny')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

iterations = np.array(range(0,n))
plt.figure(figsize=(10, 6))
if prime_value[0] < 1:
    plt.plot(iterations, errors_g1, label='g1(x)')
if prime_value[1] < 1:
    plt.plot(iterations, errors_g2, label='g2(x)')
if prime_value[2] < 1:
    plt.plot(iterations, errors_g3, label='g3(x)')
if prime_value[3] < 1:
    plt.plot(iterations, errors_g4, label='g4(x)')
    
plt.title('Błąd względny w zależności od ilości iteracji dla metod zbieżnych')
plt.xlabel('Numer iteracji')
plt.ylabel('Błąd względny')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()