import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve

S = 762
I = 1
R = 0

N = S + I + R
beta = 1
gamma = 1/7
y0 = [S,I,R]
h = 0.2
t_values = np.arange(0, 14.2, h)

S_values_explicit = np.zeros(len(t_values))
I_values_explicit = np.zeros(len(t_values))
R_values_explicit = np.zeros(len(t_values))

S_values_explicit[0]= S
I_values_explicit[0]= I
R_values_explicit[0]= R

# Funkcje pochodnych
def dS_dt(S, I, R):
    return -beta * S * I / N

def dI_dt(S, I, R):
    return beta * S * I / N - gamma * I

def dR_dt(S, I, R):
    return gamma * I


k = 1
for t in t_values:
    if t==0: continue
    R_old = R
    S_old = S
    I_old = I
    R = R_old + h*gamma*I_old
    S = S_old - h*beta*I_old*S_old/N
    I = I_old + h*beta*I_old*S_old/N - h*gamma*I_old
    S_values_explicit[k] = S
    I_values_explicit[k] = I
    R_values_explicit[k] = R
    k+=1

plt.figure(figsize=(12, 8))
plt.title("Rozwiązanie jawną metodą Eulera")
plt.plot(t_values, S_values_explicit, label='S(t) - Zdrowi podatni')
plt.plot(t_values, I_values_explicit, label='I(t) - Zainfekowani')
plt.plot(t_values, R_values_explicit, label='R(t) - Ozdrowiali')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()


def f(t, y):
    S, I, R = y
    dS = -beta * I * S / N
    dI = beta * I * S / N - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR])

def implicit_euler_method(y0, h, t_values):
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for k in range(1, len(t_values)):
        t_next = t_values[k]
        y_prev = y_values[k-1]
        
        def func(y_next):
            return y_next - y_prev - h * f(t_next, y_next)
        
        y_next = fsolve(func, y_prev)
        y_values[k] = y_next
    
    return y_values

y_implicit = implicit_euler_method(y0, h, t_values)


plt.figure(figsize=(12, 8))
plt.title("Rozwiązanie niejawną metodą Eulera")
plt.plot(t_values, y_implicit[:, 0],label='S(t) - Zdrowi podatni')
plt.plot(t_values, y_implicit[:, 1],label='I(t) - Zainfekowani')
plt.plot(t_values, y_implicit[:, 2],label='R(t) - Ozdrowiali')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()


def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = np.array(-beta / N * I * S)
    dIdt = np.array(beta / N * I * S - gamma * I)
    dRdt = np.array(gamma * I)
    return np.array([dSdt, dIdt, dRdt])

def rk4_method(f, t, y0, h, beta, gamma):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        k1 = np.array(f(t[i], y[i], beta, gamma))
        k2 = np.array(f(t[i] + h/2, y[i] + h*k1/2, beta, gamma))
        k3 = np.array(f(t[i] + h/2, y[i] + h*k2/2, beta, gamma))
        k4 = np.array(f(t[i] + h, y[i] + h*k3, beta, gamma))
        y[i + 1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return y


y_rk4 = rk4_method(sir_model, t_values, y0, 0.2, beta, gamma)

plt.figure(figsize=(12, 8))
plt.title("Rozwiązanie metodą RK4")
plt.plot(t_values, y_rk4[:, 0], label='S(t) - Zdrowi podatni')
plt.plot(t_values, y_rk4[:, 1], label='I(t) - Zainfekowani')
plt.plot(t_values, y_rk4[:, 2],  label='R(t) - Ozdrowiali')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()



plt.figure(figsize=(12, 8))
plt.title("Rozwiązania")
plt.plot(t_values, S_values_explicit, label='met. jawna S(t) - Zdrowi podatni', linestyle='dashed')
plt.plot(t_values, I_values_explicit, label='met. jawna I(t) - Zainfekowani', linestyle='dashed')
plt.plot(t_values, R_values_explicit, label='met. jawna R(t) - Ozdrowiali', linestyle='dashed')
plt.plot(t_values, y_implicit[:, 0],label='met. niejawna S(t) - Zdrowi podatni', linestyle='dotted')
plt.plot(t_values, y_implicit[:, 1],label='met. niejawna I(t) - Zainfekowani', linestyle='dotted')
plt.plot(t_values, y_implicit[:, 2],label='met. niejawna R(t) - Ozdrowiali', linestyle='dotted')
plt.plot(t_values, y_rk4[:, 0], label='met. RK4 S(t) - Zdrowi podatni')
plt.plot(t_values, y_rk4[:, 1], label='met. RK4 I(t) - Zainfekowani')
plt.plot(t_values, y_rk4[:, 2],  label='met. RK4 R(t) - Ozdrowiali')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(12, 8))
plt.subplot(1,3,1)
plt.plot(t_values, y_implicit[:, 0] + y_implicit[:, 1] + y_implicit[:, 2], label='Euler niejawna')
plt.xlabel('t')
plt.ylabel('Liczba osób')
plt.legend()
plt.title('Niezmiennik S(t) + I(t) + R(t) dla niejawnej metody eulera')
plt.grid()

plt.figure(figsize=(12, 8))
plt.subplot(1,3,2)
plt.plot(t_values, S_values_explicit + I_values_explicit + R_values_explicit,linestyle='dotted',label='Euler jawna')
plt.xlabel('t')
plt.ylabel('Liczba osób')
plt.legend()
plt.title('Niezmiennik S(t) + I(t) + R(t) dla jawnej metody eulera')
plt.grid()


plt.figure(figsize=(12, 8))
plt.subplot(1,3,3)
plt.plot(t_values, y_rk4[:, 0] + y_rk4[:, 1] + y_rk4[:, 2], linestyle='dashed',label='RK4')
plt.xlabel('t')
plt.ylabel('Liczba osób')
plt.legend()
plt.title('Niezmiennik S(t) + I(t) + R(t) dla metody RK4')
plt.grid()

plt.show()


true_infected = np.array([1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4])
days = np.arange(15)


def cost_function(params, t, true_infected):
    beta, gamma = params
    h = 1
    y_rk4 = rk4_method(sir_model, days, y0, h, beta, gamma)
    I_pred = y_rk4[:,1]
    solution = np.sum((np.array(true_infected) - I_pred) ** 2)
    return solution


def log_likelihood_function(params, t, true_infected):
    beta, gamma = params
    t_span = (0, 14)
    h = 1
    y_rk4 = rk4_method(sir_model, days, y0, h, beta, gamma)
    I_pred = y_rk4[:,1]
    solution = np.sum(-np.array(true_infected * np.log(I_pred)) + I_pred)
    return solution


options = {'maxiter': 200, 'maxfun': 300, 'disp': False}
initial_guess = [beta, gamma]

result = minimize(cost_function, initial_guess, args=(days, true_infected), method='Nelder-Mead', options=options)
beta_est, gamma_est = result.x
R0_est = beta_est / gamma_est
print(f"Estymowane wartości - suma kwadratów reszt: beta = {beta_est}, gamma = {gamma_est}, R0 = {R0_est}")

result_log_likelihood = minimize(log_likelihood_function, initial_guess, args=(days, true_infected), method='Nelder-Mead', options=options)
beta_est_log, gamma_est_log = result_log_likelihood.x
R0_est_log = beta_est_log / gamma_est_log
print(f"Estymowane wartości - log-likelihood: beta = {beta_est_log}, gamma = {gamma_est_log}, R0 = {R0_est_log}")