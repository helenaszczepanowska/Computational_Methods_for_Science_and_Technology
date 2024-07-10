
from scipy.optimize import newton

def f_a(x):
    return x ** 3 - 5 * x

def f_b(x):
    return x ** 3 - 3 * x + 1

def f_c(x):
    return 2 - x ** 5

def f_d(x):
    return x ** 4 - 4.29 * x ** 2 - 5.29

x0_a = 1
x0_b = 1
x0_c = 0.01
x0_d = 0.8

result_a = newton(f_a, x0_a)
result_b = newton(f_b, x0_b)
result_c = newton(f_c, x0_c)
result_d = newton(f_d, x0_d)

print("Rozwiązanie dla funkcji a:", round(result_a, 2))
print("Rozwiązanie dla funkcji b:", round(result_b, 2))
print("Rozwiązanie dla funkcji c:", round(result_c, 2))
print("Rozwiązanie dla funkcji d:", round(result_d, 2))

x0_a = 2
x0_b = 1.5
x0_c = 1.1
x0_d = 2

result_a = newton(f_a, x0_a)
result_b = newton(f_b, x0_b)
result_c = newton(f_c, x0_c)
result_d = newton(f_d, x0_d)

print("Rozwiązanie dla funkcji a:", round(result_a, 2))
print("Rozwiązanie dla funkcji b:", round(result_b, 2))
print("Rozwiązanie dla funkcji c:", round(result_c, 2))
print("Rozwiązanie dla funkcji d:", round(result_d, 2))