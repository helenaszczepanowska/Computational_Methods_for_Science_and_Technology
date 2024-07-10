import numpy as np
import matplotlib.pyplot as plt

# Dane
years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
population = np.array([76212168, 92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203302031, 226542199])

j = np.arange(9)


# Funkcje bazowe
def phi1(t,i):
    return t ** i

def phi2(t,i):
    return (t - 1900) ** i

def phi3(t,i):
    return (t - 1940) ** i

def phi4(t,i):
    return ((t - 1940) / 40) ** i


T_phi = [phi1,phi2,phi3,phi4]


# (a) Macierze Vandermonde'a
V1 = np.array([[phi1(t,i) for i in j] for t in years ])
V2 = np.array([[phi2(t,i) for i in j] for t in years ])
V3 = np.array([[phi3(t,i) for i in j] for t in years ])
V4 = np.array([[phi4(t,i) for i in j] for t in years ])

T_V = [V1,V2,V3,V4]

# (b) Współczynniki uwarunkowania
cond_V1 = np.linalg.cond(V1)
cond_V2 = np.linalg.cond(V2)
cond_V3 = np.linalg.cond(V3)
cond_V4 = np.linalg.cond(V4)

print("Współczynniki uwarunkowania:")
print("(1) ", cond_V1)
print("(2) ", cond_V2)
print("(3) ", cond_V3)
print("(4) ", cond_V4)

T_cond = [cond_V1,cond_V2,cond_V3,cond_V4]
min_cond = min(T_cond)


# (c) Wielomian interpolacyjny
best_V = T_V[T_cond.index(min_cond)]
phi = T_phi[T_cond.index(min_cond)]


B = lambda t: np.array([phi(t,i) for i in j]) # B - baza
coefficients = np.linalg.solve(best_V, population.T) # A - współczynnik wielomianu interpolacyjnego
p = lambda t: B(t) @ coefficients.T


# Wartości wielomianu na przedziale [1900, 1990] w odstępach jednorocznych
all_years = np.arange(1900, 1991)
interpolated_population = np.array([p(x) for x in all_years])

# Wykres
plt.figure(figsize=(10, 6))
plt.plot(years, population, 'ro', label='Węzły interpolacji')
plt.plot(all_years, interpolated_population, label='Wielomian interpolacyjny')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.title('Interpolacja populacji Stanów Zjednoczonych')
plt.legend()
plt.grid(True)
plt.show()

# (d) Ekstrapolacja do roku 1990
extrapolated_population = p(1990)
true_value_1990 = 248709873
error = abs(extrapolated_population - true_value_1990) / true_value_1990 * 100

print("Wartość ekstrapolowana dla roku 1990:", extrapolated_population)
print("Prawdziwa wartość dla roku 1990:", true_value_1990)
print("Błąd względny ekstrapolacji dla roku 1990:", error, "%")


# (e) Wielomian interpolacyjny Lagrange'a

l = lambda t, j: np.prod(t-years[years!=years[j]])/np.prod(years[j]-years[years!=years[j]])
p_lagerange = lambda t: np.sum([population[j] * l(t, j) for j in range(9)])
interpolated_population_lagrange = np.array([p_lagerange(t) for t in all_years])

# wykres
plt.figure(figsize=(10, 6))
plt.plot(years, population, 'ro', label='Węzły interpolacji')
plt.plot(all_years, interpolated_population_lagrange, label='Wielomian interpolacyjny')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.title('Interpolacja Lagrange populacji Stanów Zjednoczonych')
plt.legend()
plt.grid(True)
plt.show()


# (f) Wielomian interpolacyjny Newtona

# funkcja bazowa
l_n = lambda t, j: np.prod(t-years[years<years[j]])


def x(i, j):
    if i == j: return population[i]
    return (x(i+1, j) - x(i, j-1))/(years[j] - years[i])


X = [x(0, i) for i in range(9)]
p_n = lambda t: np.sum([X[j] * l_n(t, j) for j in range(9)])


def horner(base, coefficients):
    n = len(coefficients) - 1
    W = coefficients[-1]
    for i in range(n-1, -1, -1):
        W = W*base[i] + coefficients[i]
    return W


interpolated_population_newton = np.array([p_n(t) for t in all_years])

# wykres
plt.figure(figsize=(10, 6))
plt.plot(years, population, 'ro', label='Węzły interpolacji')
plt.plot(all_years, interpolated_population_newton, label='Wielomian interpolacyjny')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.title('Interpolacja Newtona populacji Stanów Zjednoczonych')
plt.legend()
plt.grid(True)
plt.show()

# (g) Wielomian interpolacyjny dla danych zaokrąglonych
population_rounded = np.round(population / 1e6) * 1e6
print(population_rounded)
coefficients_rounded = np.linalg.solve(best_V, population_rounded)

p_rounded = lambda t: B(t) @ coefficients.T
interpolated_population_rounded = np.array([p_rounded(x) for x in all_years])

# wykres
plt.figure(figsize=(10, 6))
plt.plot(years, population_rounded, 'ro', label='Węzły interpolacji')
plt.plot(all_years, interpolated_population_rounded, label='Wielomian interpolacyjny')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.title('Interpolacja populacji Stanów Zjednoczonych dla danych zaokrąglonych')
plt.legend()
plt.grid(True)
plt.show()

# Porównanie wyznaczonych współczynników
print("Porównanie współczynników interpolacyjnych dla danych zaokrąglonych:\n")
print("Oryginalne współczynniki: ", coefficients,"\n")
print("Współczynniki dla danych zaokrąglonych: ", coefficients_rounded,"\n")
abs_coefficients_difference = np.abs(coefficients_rounded - coefficients)
print("Różnica współczynników: ", abs_coefficients_difference)

# Porównanie wartości wielomianów interpolacyjnych dla Lagrange'a i Newtona
polynomial_difference = np.abs(interpolated_population_lagrange - interpolated_population_newton)
print(polynomial_difference)
