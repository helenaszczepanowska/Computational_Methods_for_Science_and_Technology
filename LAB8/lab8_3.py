import numpy as np

def f_a(x):
    return x ** 3 - 2 * x - 5

def f_b(x):
    return np.exp(-x) - x

def f_c(x):
    return x * np.sin(x) - 1

def f_a_prim(x):
    return 3 * x ** 2 - 2

def f_b_prim(x):
    return -np.exp(-x) - 1

def f_c_prim(x):
    return np.sin(x) + x * np.cos(x)

def newton_iteration(f, f_prim, x0, tolerance, max_iteration=10):
    x = x0
    iterations = 0
    while iterations < max_iteration:
        x_next = x - f(x) / f_prim(x)
        iterations += 1
        if abs(x_next - x) < tolerance:
            break
        x = x_next
    return x_next, iterations

tolerance_24_bit = 1 / 2 ** 24
tolerance_53_bit = 1 / 2 ** 53

x0 = 1.5
x0_a = 2.0945
x0_b = 0.5671
x0_c = 1.1141

result_a_24_bit, iterations_a_24_bit = newton_iteration(f_a, f_a_prim, x0_a, tolerance_24_bit)
result_b_24_bit, iterations_b_24_bit = newton_iteration(f_b, f_b_prim, x0_b, tolerance_24_bit)
result_c_24_bit, iterations_c_24_bit = newton_iteration(f_c, f_c_prim, x0_c, tolerance_24_bit)

result_a_53_bit, iterations_a_53_bit = newton_iteration(f_a, f_a_prim, x0_a, tolerance_53_bit)
result_b_53_bit, iterations_b_53_bit = newton_iteration(f_b, f_b_prim, x0_b, tolerance_53_bit)
result_c_53_bit, iterations_c_53_bit = newton_iteration(f_c, f_c_prim, x0_c, tolerance_53_bit)

print("24-bitowa dokładność:")
print("Dla równania (a): Liczba iteracji =", iterations_a_24_bit)
print("Dla równania (b): Liczba iteracji =", iterations_b_24_bit)
print("Dla równania (c): Liczba iteracji =", iterations_c_24_bit)

print("\n53-bitowa dokładność:")
print("Dla równania (a): Liczba iteracji =", iterations_a_53_bit)
print("Dla równania (b): Liczba iteracji =", iterations_b_53_bit)
print("Dla równania (c): Liczba iteracji =", iterations_c_53_bit)