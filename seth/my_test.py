import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
from IPython.display import HTML

### Parameters ### 
c = 1. 
k = 12
L = 10. 
T = L / (2*c)
#T = 0.02
print("True Final Time = ", T)

### Domain ###
a = - L
b = L
M = 2**k
h = (b - a) / (M)
N = int(c * T / h)
tau = T / (N)
print("mesh size h = ", h)
print("tau = ", tau)
print("Number of spatial dofs = ", M)
print("Number of time steps = ", N)
time = 0.0

xs = [a + i * h for i in range(M + 1)]
ts = [n * tau for n in range(N + 1)]

### Initialize ### 
U_initial = np.zeros(M+1)
U_very_old = np.zeros(M+1)
U_old = np.zeros(M+1)
U_new = np.zeros(M+1)
U_final = np.zeros(M+1)
U_1 = np.zeros(M+1)
V_0 = np.zeros(M+1)
F_at_time = np.zeros(M+1)
U_m_1 = np.zeros(M+1)
U_m_1_exact = np.zeros(M + 1)
U_exact_final = np.zeros(M+1)

def u0(x, t): 
    u = np.exp( -(x-c*t)**2)
    return u

def v(x, t): 
    v = 2*c*(x - c*t) * np.exp( -(x-c*t)**2)
    return v

def f(x,t):
    return 0.

# Interpolate initial condition for u
for i in range(0,M+1):
    U_initial[i] = u0(xs[i], 0)
    
# Interpolate (exact) initial condition for u(*, - taus)
for i in range(0, M+1):
    U_m_1_exact[i] = u0(xs[i],-tau)
    
# Interpolate initial condition for v_0(x)
for i in range(0,M+1):
    V_0[i] = v(xs[i], 0)
    
for i in range(0,M+1):
    U_exact_final[i] = u0(xs[i], T)


### Helper functions

def compute_f_at_time_t(t,F): 
    for i in range(0,M+1):
        F[i] = f(xs[i], t)
    return F

def step(u_old, u_very_old, u_new, time):
    F_at_time_t = compute_f_at_time_t(time, F_at_time)
    # Left boundary condition
    u_new[0] = 2*u_old[0] - u_very_old[0]  + c**2 * tau**2 / (h**2) * (u_old[1] - 2 *u_old[0] + u_old[M-1]) + tau**2 *F_at_time[0]
    # Loop over space
    for i in range(1,M):
        u_new[i] = 2*u_old[i] - u_very_old[i] + c**2 * tau**2 /(h**2) *( u_old[i+1] - 2*u_old[i] + u_old[i-1]) + tau**2*F_at_time[i]
    # Right boundary condition
    u_new[M] = u_new[0]
    return u_new

def compute_first_time_step(u_new, u_initial, v_initial):
    F_at_time_t = compute_f_at_time_t(tau, F_at_time)
    # Left boundary conditions
    u_new[0] = u_initial[0] + tau * v_initial[0] + c**2 * tau**2 /(2* h**2) * (u_initial[1] - 2*u_initial[0] + u_initial[M-1]) + tau**2/(2.) * F_at_time[0]
    for i in range(1,M):
        u_new[i] = u_initial[i] + tau * v_initial[i] + c**2 * tau**2 /(2*h**2) * (u_initial[i+1] - 2*u_initial[i] + u_initial[i-1]) + tau**2/2. * F_at_time[i] 
    # Right boundary condition 
    u_new[M] = u_new[0]
    return u_new


### True way ###
U_old = U_initial.copy()

plt.figure(figsize=(8,5))
plt.plot(xs, U_initial, label="Numerical solution")

plt.xlabel("x")
plt.ylabel("u(x,0)")
plt.title("Initial Condition")
plt.legend()
plt.grid(True)
plt.show()


compute_first_time_step(U_new, U_old, V_0)
U_very_old = U_initial.copy()
U_old = U_new.copy()
time = time + tau

plt.figure(figsize=(8,5))

plt.plot(xs, U_old, label="Numerical solution")
plt.xlabel("x")
plt.ylabel("u(x,tau)")
plt.title("First time step")
plt.legend()
plt.grid(True)
plt.show()


### Time loop ###
for n in range(1,N+1):
    F_at_time = compute_f_at_time_t(time, F_at_time)
    U_new = step(U_old, U_very_old, U_new, time)
    time = time + tau
    U_very_old = U_old.copy()
    U_old = U_new.copy()


    
final_time = time - tau
    
print("Final computed time = ", final_time)

# Final Numerical Solution
U_final = U_old.copy()

plt.figure(figsize=(8,5))

plt.plot(xs, U_final, label="Numerical solution")
#plt.plot(xs, U_exact_final, label="Exact solution")

plt.xlabel("x")
plt.ylabel("u(x,T)")
plt.title("Wave equation solution at final time")
plt.legend()
plt.grid(True)
plt.show()

