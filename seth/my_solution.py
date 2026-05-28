import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
from IPython.display import HTML

def solve_dispsersive_pde(domain_length, final_time, refinement, c, cfl, u_0, v_0, g, f, boundary_condition="periodic"):
    L = domain_length
    a = -L
    b = L 
    M = 2**refinement
    h = (b - a) / (M)
    N = int(c * final_time / (h*cfl))
    tau = final_time / N
    
    boundary = boundary_condition
    
    xs = [a + i * h for i in range(M + 1)]
    ts = [n * tau for n in range(N + 1)]

    # U[i,n] approximates u(x_i, t_n)
    U = np.zeros((M + 1, N + 1))

    # each timestep t_{n+1}, n = 1,...,N, we explicitly update U[:,n+1]

    # initialize t_0
    for i in range(1, M):
        U[i, 0] = u_0(xs[i])

    # boundary conditions at t_0
    U[0, 0] = u_0(xs[0])
    U[M, 0] = U[0, 0]
    
        # initialize t_1
    for i in range(1, M):
        U[i, 1] = (U[i, 0] + tau * v_0(xs[i])
                  + tau**2 * c**2 / (2 * h**2) * (U[i + 1, 0] - 2 * U[i, 0] + U[i - 1, 0]) - tau**2/2 *g(U[i,0])
                  + tau**2 / 2 * f(xs[i], ts[0])
                  )

    # absorbing boundary conditions at t_1
    if boundary == "absorbing":
        U[0, 1] = U[1,0]
        U[M, 1] = U[M-1, 0]

    # periodic boundary conditions at t_1
    if boundary == "periodic":
        U[0, 1] = (U[0, 0] + tau * v_0(xs[0])
                  + tau**2 * c**2 / (2 * h**2) * (U[1, 0] - 2 * U[0, 0] + U[M - 1, 0]) - tau**2/2 * g(U[0,0])
                  + tau**2 / 2 * f(xs[0], ts[0])
              )
        U[M, 1] = U[0, 1]
        
        # explicit timestepping
    for n in range(1, N):
        for i in range(1, M):
            U[i, n + 1] = (
                2 * U[i, n]
                - U[i, n - 1]
                + tau**2 * c**2 / h**2 * (U[i + 1, n] - 2 * U[i, n] + U[i - 1, n]) - tau**2 * g(U[i,n])
                + tau**2 * f(xs[i], ts[n])
            )

        # absorbing boundary conditions at t_1
        if boundary == "absorbing":
              U[0, n+1] = U[1,n]
              U[M, n+1] = U[M-1, n]

        if boundary == "periodic":
              U[0, n + 1] = (
                  2 * U[0, n]
                  - U[0, n - 1]
                  + tau**2 * c**2 / h**2 * (U[1, n] - 2 * U[0, n] + U[M - 1, n]) - tau**2 * g(U[0,n])
                  + tau**2 * f(xs[0], ts[n])
              )
              U[M, n + 1] = U[0, n + 1]
     
    return U

### WAVE EQUATION ### 
print("### SOLVING THE WAVE EQUATION ###")
c_wave = 1
g_wave = lambda u : 0
u_0_wave = lambda x : np.exp(-(x**2))
v_0_wave = lambda x : 2*c_wave*x*np.exp(-(x**2))
f_wave = lambda x,t : 0
L_wave = 10
T_wave = L_wave/(2*c_wave)
k_wave = 12
CFL_wave = 1.0

def u_wave_exact(x, t):
    c = 1
    return np.exp(-(x - c*t)**2)

def test_wave_equation():
    U_wave = solve_dispsersive_pde(L_wave, T_wave, 12, c_wave, CFL_wave, u_0_wave, v_0_wave, g_wave, f_wave)
    M=2**(12)
    a = -10
    b = 10
    h = (b - a) / (M)
    xs = [a + i * h for i in range(M + 1)]
    U_wave_exact = [u_wave_exact(x, T_wave) for x in xs]
    plt.plot(xs, U_wave_exact, label="exact")
    plt.plot(xs, U_wave[: , -1], label="approximate")
    plt.legend()
    plt.show()

def compute_wave_errors(): 
    k_list = [10, 11, 12]
    h_list = []
    error_list = []
    rate_list = []
    for k in k_list:
        U =  solve_dispsersive_pde(L_wave, T_wave, k, c_wave, CFL_wave, u_0_wave, v_0_wave, g_wave, f_wave)

        # exact solution at the discrete grid points
        M = 2**k
        h = (2*L_wave) / (M)
        h_list.append(h)
        xs = [-L_wave + i * h for i in range(M + 1)]
        U_exact = [u_wave_exact(x,T_wave) for x in xs]

        E_M = max([abs(U_exact[i] - U[i, -1]) for i in range(M + 1)])
        error_list.append(E_M)
        # table entry
        print("{:e}\t{:e}".format(h, E_M))

    for j in [1,2]:
        r_k = np.log(error_list[j]/error_list[j-1])/np.log(h_list[j]/h_list[j-1])
        rate_list.append(r_k)

    print(rate_list)

# compute_wave_errors()


### KLEIN GORDON ### 
print("### SOLVING THE KLEIN-GORDON EQUATION ###")
c_klein = 1
g_klein = lambda u : u
u_0_klein = lambda x : np.cos(np.pi*x)
v_0_klein = lambda x : -np.sqrt((np.pi**2+1))*np.sin(np.pi*x)
f_klein = lambda x,t : 0
L_klein = 1
T_klein = 2*np.pi/np.sqrt((np.pi**2 +1))
k_klein = 12
CFL_klein = 0.25

def u_klein_exact(x,t):
    u = np.cos(np.pi*x + np.sqrt((np.pi**2 + 1))*t)
    return u

def test_klein_equation():
    U_wave = solve_dispsersive_pde(L_klein, T_klein, 12, c_klein, CFL_klein, u_0_klein, v_0_klein, g_klein, f_klein)
    M=2**(12)
    a = -10
    b = 10
    h = (b - a) / (M)
    xs = [a + i * h for i in range(M + 1)]
    U_klein_exact = [u_klein_exact(x, T_wave) for x in xs]
    plt.plot(xs, U_klein_exact, label="exact")
    plt.plot(xs, U_wave[: , -1], label="approximate")
    plt.legend()
    plt.show()

def compute_klein_errors(): 
    k_list = [8, 9, 10]
    h_list = []
    error_list = []
    rate_list = []
    for k in k_list:
        U =  solve_dispsersive_pde(L_klein, T_klein, k, c_klein, CFL_klein, u_0_klein, v_0_klein, g_klein, f_klein)

        # exact solution at the discrete grid points
        M = 2**k
        h = (2*L_klein) / (M)
        h_list.append(h)
        xs = [-L_klein + i * h for i in range(M + 1)]
        U_exact = [u_klein_exact(x,T_klein) for x in xs]

        E_M = max([abs(U_exact[i] - U[i, -1]) for i in range(M + 1)])
        error_list.append(E_M)
        # table entry
        print("{:e}\t{:e}".format(h, E_M))

    for j in [1,2]:
        r_k = np.log(error_list[j]/error_list[j-1])/np.log(h_list[j]/h_list[j-1])
        rate_list.append(r_k)

    print(rate_list)
    
# compute_klein_errors()



### SINE GORDON ### 
print("### SOLVING THE SINE-GORDON EQUATION ###")

c_sine = 1
g_sine = lambda u : np.sin(u)
omega = 0.99
alpha = np.sqrt(1 - omega**2)
u_0_sine = lambda x : 4*np.arctan(alpha/(omega*np.cosh(alpha*x)))
v_0_sine = lambda x : 0
f_sine = lambda x,t : 0
L_sine = 100
T_sine = 2*np.pi/np.sqrt((np.pi**2 +1))
k_sine = 12
CFL_sine = 0.25

def u_sine_exact(x, t, omega=0.99):
        alpha = np.sqrt(1 - omega**2)
        u = 4 * np.arctan(
        (np.cos(omega * t) * alpha) /
        (omega * np.cosh(alpha * x))
        )
        return u
    
def u_traveling_breather(x,t, omega=0.90):
    c = 1
    v = 1 / np.sqrt(2)
    omega = 0.99
    gamma = 1 / np.sqrt(1-v**2)
    delta = 0
    numerator = np.cos(omega*gamma*t-omega*(x-delta)*np.sqrt(gamma**2-1))*np.sqrt(1-omega**2)
    denominator = omega*np.cosh((gamma*(x-delta)-t*np.sqrt(gamma**2-1))*np.sqrt(1-omega**2))
    return 4 * np.arctan(numerator/denominator)

u_tbreather = lambda x: u_traveling_breather(x,0)

#c = sp.Integer(1)
#omega = sp.Rational(99, 100)
#v = sp.Integer(1)/(sp.sqrt(2))
#gamma = sp.Integer(1)/sp.sqrt(sp.Integer(1) - v**2)

#x,t = sp.symbols('x,t')
#u_tb = 4 * sp.atan((sp.cos(omega*gamma*t- omega*x*sp.sqrt(gamma**2 - sp.Integer(1)))*sp.sqrt(sp.Integer(1) - omega**2))/(omega*sp.cosh(sp.sqrt(sp.Integer(1) - omega**2)*(gamma*x- t *sp.sqrt(gamma**2 -1)))))
    
#dutb_dt = sp.diff(u_tb, t) 
#v = dutb_dt.subs(t,0)

#v_func = sp.lambdify(x, v, "numpy")

def compute_traveling_breather_errors():
    k_list = [8, 9, 10, 11]
    h_list = []
    error_list = []
    rate_list = []
    
    v = 1 / np.sqrt(2)
    gamma = 1 / np.sqrt(1-v**2)
    u_tbreather = lambda x: u_traveling_breather(x,0)
    T_tb = 20
    v_0_tbreather = lambda x: -4*omega*(-gamma*omega*np.sqrt(1 - omega**2)*np.sin(omega*(x)*np.sqrt(gamma**2 - 1))*np.cosh(gamma*(x)*np.sqrt(1 - omega**2)) + np.sqrt(gamma**2 - 1)*(omega**2 - 1)*np.cos(omega*(x)*np.sqrt(gamma**2 - 1))*np.sinh(gamma*(x)*np.sqrt(1 - omega**2)))/(omega**2*np.cosh(gamma*(x)*np.sqrt(1 - omega**2))**2 - (omega**2 - 1)*np.cos(omega*(x)*np.sqrt(gamma**2 - 1))**2)
    for k in k_list:
        U =  solve_dispsersive_pde(L_sine, T_tb, k, c_sine, 0.9, u_tbreather, v_0_tbreather, g_sine, f_sine)

        # exact solution at the discrete grid points
        M = 2**k
        h = (2*L_sine) / (M)
        h_list.append(h)
        xs = [-L_sine + i * h for i in range(M + 1)]
        U_exact = [u_traveling_breather(x,T_tb) for x in xs]
        U_initial = [u_traveling_breather(x,0) for x in xs]

        E_M = max([abs(U_exact[i] - U[i, -1]) for i in range(M + 1)])
        error_list.append(E_M)
        # table entry
        print("{:e}\t{:e}".format(h, E_M))

    for j in [1,2,3]:
        r_k = np.log(error_list[j]/error_list[j-1])/np.log(h_list[j]/h_list[j-1])
        rate_list.append(r_k)

    print(rate_list)
    
    plt.plot(xs, U_initial, label="exact initial")
    plt.plot(xs, U[: , 0], label="approximate initial")
    plt.legend()
    plt.show()
    
    plt.plot(xs, U_exact, label="exact final")
    plt.plot(xs, U[: , -1], label="approximate")
    plt.legend()
    plt.show()
    
# compute_traveling_breather_errors()
    
def compute_sine_errors(): 
    k_list = [8, 9, 10]
    h_list = []
    error_list = []
    rate_list = []
    for k in k_list:
        U =  solve_dispsersive_pde(L_sine, T_sine, k, c_sine, CFL_sine, u_0_sine, v_0_sine, g_sine, f_sine)

        # exact solution at the discrete grid points
        M = 2**k
        h = (2*L_sine) / (M)
        h_list.append(h)
        xs = [-L_sine + i * h for i in range(M + 1)]
        U_exact = [u_sine_exact(x,T_sine) for x in xs]
        U_initial = [u_sine_exact(x,0) for x in xs]

        E_M = max([abs(U_exact[i] - U[i, -1]) for i in range(M + 1)])
        error_list.append(E_M)
        # table entry
        print("{:e}\t{:e}".format(h, E_M))

    for j in [1,2]:
        r_k = np.log(error_list[j]/error_list[j-1])/np.log(h_list[j]/h_list[j-1])
        rate_list.append(r_k)

    print(rate_list)
    plt.plot(xs, U_initial, label="exact initial")
    plt.plot(xs, U[: , 0], label="approximate initial")
    plt.legend()
    plt.show()
    
    plt.plot(xs, U_exact, label="exact final")
    plt.plot(xs, U[: , -1], label="approximate")
    plt.legend()
    plt.show()
    

compute_sine_errors()
    



