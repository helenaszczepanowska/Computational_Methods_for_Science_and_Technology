import numpy as np
import scipy
import matplotlib.pyplot as plt

lambd = -5
h = 0.5

explicit_euler_amp_factor = lambda lambd,h: 1 + lambd*h
implicit_euler_amp_factor = lambda lambd,h: 1/(1-lambd*h)

print("Amplification factors:")
print("explicit euler method: ",round(explicit_euler_amp_factor(lambd,h),2), "\n implicit euler method: ",round(implicit_euler_amp_factor(lambd,h),2))


n = 5

# Explicit Euler Method

t_values = np.arange(0, n + h, h)
y_values = y_values = np.zeros(len(t_values))
y_values[0] = 1
for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1]*(1+h*(-lambd))


print(f"For t={t_values[1]}, y={y_values[1]}")

# Plot
plt.title("Explicit Euler Method solution")
plt.plot(t_values, y_values, label='explicit euler method')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()


# Implicit Euler Method

t_values_imp = np.arange(0, n + h, h)
y_values_imp  = np.zeros(len(t_values_imp))
y_values_imp[0] = 1
for i in range(1, len(t_values_imp)):
        y_values_imp[i] = y_values_imp[i-1]/(1-lambd*h)

print(f"For t={t_values_imp[1]}, y={y_values_imp[1]}")

# Plot
plt.title("Implicit Euler Method solution")
plt.plot(t_values_imp, y_values_imp, label='implicit euler method')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()


# Plot
plt.title("Both Solutions")
plt.plot(t_values_imp, y_values_imp, label='implicit euler method')
plt.plot(t_values, y_values, label='explicit euler method')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()