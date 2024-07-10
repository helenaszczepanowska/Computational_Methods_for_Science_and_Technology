import numpy as np
import matplotlib.pyplot as plt

# Dane populacyjne
years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
population = np.array([76212168, 92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203302031, 226542199])
year_1990_true = 248709873  # Prawdziwa wartość populacji w roku 1990

# Stopień wielomianu aproksymującego
m_values = np.arange(7)


# Obliczanie macierzy S
def calculate_S(years, m):
    n = len(years)
    S = np.zeros((m+1, m+1))
    for i in range(m+1):
        for j in range(m+1):
            S[i][j] = np.sum(years ** (i+j))
    return S


# Obliczanie wektora T
def calculate_T(years, population, m):
    n = len(years)
    T = np.zeros(m+1)
    for i in range(m+1):
        T[i] = np.sum(years ** i * population)
    return T


# Rozwiązanie układu równań
def solve_equations(S, T):
    coefficients = np.linalg.solve(S, T)
    return coefficients


def p(coefficients,x):
    return np.sum(coefficients*(x**np.arange(len(coefficients))))


# Ekstrapolacja do roku 1990
def extrapolate_to_1990(coefficients, year):
    return p(coefficients,year)


S = []
T = []
coefficients = []
population_1990_approx =[]
relative_error_1990 =[]
y_values = []

# Obliczenia
for m in m_values:
    S.append(calculate_S(years, m))
    T.append(calculate_T(years, population, m))
    coefficients.append(solve_equations(S[m], T[m]))
    population_1990_approx.append(extrapolate_to_1990(coefficients[m], 1990))

    # Obliczenie błędu względnego ekstrapolacji
    relative_error_1990.append(np.abs(population_1990_approx[m] - year_1990_true) / year_1990_true * 100)
    print("Wynik ekstrapolacji populacji w roku 1990 (przybliżony) dla m = ", m," : ", round(population_1990_approx[m],2))
    print("Błąd względny ekstrapolacji dla roku 1990 2: ", round(relative_error_1990[m],2),"% \n")
    y_values.append([p(coefficients[m],year) for year in years] )


plt.figure(figsize=(10, 6))
plt.plot(years, population, 'ro', label='Dane populacyjne')
for m in m_values:
    if m == 6: break
    elif m != 1 and m != 2:
        plt.plot(years, y_values[m], label='Aproksymacja wielomianem stopnia {}'.format(m), linestyle='--')

    else:
        plt.plot(years, y_values[m], label='Aproksymacja wielomianem stopnia {}'.format(m))
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.title('Aproksymacja populacji przy użyciu metody najmniejszych kwadratów')
plt.legend()
plt.grid(True)
plt.show()


# plot with m = 6
plt.figure(figsize=(10, 6))
plt.plot(years, population, 'ro', label='Dane populacyjne')
for m in m_values:
    if m != 1 and m != 2:
        plt.plot(years, y_values[m], label='Aproksymacja wielomianem stopnia {}'.format(m), linestyle='--')

    else:
        plt.plot(years, y_values[m], label='Aproksymacja wielomianem stopnia {}'.format(m))
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.title('Aproksymacja populacji przy użyciu metody najmniejszych kwadratów uwzględniając m = 6')
plt.legend()
plt.grid(True)
plt.show()


# Funkcja obliczająca AIC
def calculate_AIC(y, y_hat, n, k):
    residual_sum_squares = np.sum((y - y_hat) ** 2)
    AIC = 2 * k + n * np.log(residual_sum_squares / n)
    return AIC

# Funkcja obliczająca AICc
def calculate_AICc(AIC, n, k):
    AICc = AIC + (2 * k * (k + 1)) / (n - k - 1)
    return AICc


# Wartości początkowe
n = len(years)  # Liczba danych
AICc_min = np.inf
best_m = None
AIC_tab = []
AICc_tab = []

# Obliczanie optymalnego stopnia wielomianu
for m in m_values:  # Próbujemy stopnie wielomianu od 1 do 6
    k = m + 1  # Liczba parametrów modelu
    y_hat =[p(coefficients[m], year) for year in years]# Przewidywane wartości
    AIC = calculate_AIC(population, y_hat, n, k)
    AICc = calculate_AICc(AIC, n, k)
    AIC_tab.append(AIC)
    AICc_tab.append(AICc)
    if AICc < AICc_min:
        AICc_min = AICc
        best_m = m

min_error = min(relative_error_1990)
best_m_a = 0
for m in m_values:
    if relative_error_1990[m] == min_error:
        best_m_a = m
        break

print("wartości AIC: ", AIC_tab)
print("wartości AICc: ", AICc_tab)
print("Optymalny stopień wielomianu wyznaczony w pkt a:", best_m_a)
print("Optymalny stopień wielomianu wyznaczony za pomocą AICc:", best_m)
print("Wartość AICc dla optymalnego stopnia:", AICc_min)
