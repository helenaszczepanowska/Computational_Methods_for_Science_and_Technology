import numpy as np
import matplotlib.pyplot as plt

def f_1(x_1,x_2):
    return x_1 ** 2 + x_2 ** 2 - 1

def f_2(x_1,x_2):
    return x_1 ** 2 - x_2

def jacobian(x_1, x_2):
    return np.array([[2 * x_1, 2 * x_2], [2 * x_1, -1]])

def newton_system(f1, f2, jacobian, x0, exact_solution_1, exact_solution_2, tolerance=1e-15, max_iterations=20):
    x = np.array(x0, dtype=float)
    errors = []

    for i in range(max_iterations):
        f = np.array([f1(x[0], x[1]), f2(x[0], x[1])])
        J = jacobian(x[0], x[1])
        delta_x = np.linalg.solve(J, -f)
        x += delta_x
        error = np.sqrt((x[0]- exact_solution_1)**2 + (x[1] - exact_solution_2)**2)
        errors.append(error)
        if error < tolerance:
            return x, i, errors
        
exact_solution_1 = np.sqrt(np.sqrt(5)/2 - 0.5)
exact_solution_2 = np.sqrt(5)/2 - 0.5
exact_value_norm = np.sqrt(exact_solution_1**2 + exact_solution_2**2)

x0 = [1, 1]
solution, iterations, errors = newton_system(f_1, f_2, jacobian, x0, exact_solution_1, exact_solution_2)

relative_error_1 = abs(solution[0] - exact_solution_1) / abs(exact_solution_1)
relative_error_2 = abs(solution[1] - exact_solution_2) / abs(exact_solution_2)

print("Rozwiązanie:", np.round(solution, 2))
print("Liczba iteracji:", iterations)
print("Błąd względny x1:", relative_error_1)
print("Błąd względny x2:", relative_error_2)

n_range = np.array(range(0, iterations+1))
plt.plot(n_range, errors/exact_value_norm)
plt.yscale('log')
plt.xlabel('Numer iteracji')
plt.ylabel('Błąd względny')
plt.title('Błąd względny w zależności od numeru iteracji')
plt.show()