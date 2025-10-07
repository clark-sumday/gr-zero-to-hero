# Lesson 3: Differential Equations

**Topics:** First-order ODEs, second-order ODEs, systems of ODEs, numerical methods, phase portraits
**Prerequisites:** Lessons 1-2 (Linear Algebra, Multivariable Calculus)
**Time:** 4-6 hours

## Table of Contents
1. [First-Order ODEs](#1-first-order-odes)
2. [Second-Order Linear ODEs](#2-second-order-linear-odes)
3. [Systems of ODEs](#3-systems-of-odes)
4. [Phase Portraits](#4-phase-portraits)
5. [Numerical Methods](#5-numerical-methods)
6. [Boundary Value Problems](#6-boundary-value-problems)

---

## 1. First-Order ODEs

### 📖 Concept

A **first-order ordinary differential equation (ODE)** involves a function and its first derivative:

```
dy/dt = f(t, y)
```

**Separable ODEs:** Can be written as g(y)dy = h(t)dt
- Solve by integrating both sides

**Linear first-order:** dy/dt + p(t)y = q(t)
- Solve using integrating factor: μ(t) = e^∫p(t)dt

**Connection to GR:** The geodesic equation (equation of motion in curved spacetime) is a system of second-order ODEs.

### 💻 Code Example

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
from scipy.integrate import odeint  # ODE solver for initial value problems
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Example: exponential decay dy/dt = -ky
def exponential_decay(y, t, k=0.5):
    """dy/dt = -ky"""
    return -k * y

# Analytical solution: y(t) = y₀ e^(-kt)
def analytical_solution(t, y0=1.0, k=0.5):
    return y0 * np.exp(-k * t)  # np.exp() computes exponential e^x

# Solve numerically
y0 = 1.0
k = 0.5
t = np.linspace(0, 10, 100)  # np.linspace() creates evenly spaced array between start and end

y_numerical = odeint(exponential_decay, y0, t, args=(k,))  # odeint() solves ODE system with given initial conditions
y_exact = analytical_solution(t, y0, k)

print("First-order ODE: dy/dt = -ky")
print(f"Initial condition: y(0) = {y0}")
print(f"Decay constant: k = {k}")
print(f"\nAnalytical solution: y(t) = {y0}·e^(-{k}t)")
print(f"\nNumerical vs Exact at t=5:")
print(f"  Numerical: {y_numerical[50][0]:.6f}")
print(f"  Exact: {y_exact[50]:.6f}")
print(f"  Error: {abs(y_numerical[50][0] - y_exact[50]):.2e}")
```

### 📊 Visualization

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Solution curves
ax1.plot(t, y_exact, '-', color=COLORS['blue'], linewidth=2, label='Analytical')
ax1.plot(t, y_numerical, '--', color=COLORS['red'], linewidth=2, label='Numerical')
ax1.set_xlabel('t')
ax1.set_ylabel('y(t)')
ax1.set_title('Exponential Decay: dy/dt = -ky')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Slope field (direction field)
t_field = np.linspace(0, 10, 20)  # np.linspace() creates evenly spaced array between start and end
y_field = np.linspace(0, 2, 20)  # np.linspace() creates evenly spaced array between start and end
T, Y = np.meshgrid(t_field, y_field)  # np.meshgrid() creates coordinate matrices from coordinate vectors
dY = -k * Y  # dy/dt = -ky
dT = np.ones_like(dY)

# Normalize vectors for display
M = np.sqrt(dT**2 + dY**2)  # np.sqrt() computes square root
dT_norm = dT / M
dY_norm = dY / M

ax2.quiver(T, Y, dT_norm, dY_norm, M, alpha=0.6, cmap='viridis')
ax2.plot(t, y_exact, color=COLORS['red'], linewidth=3, label=f'Solution y₀={y0}')

# Additional solution curves
for y0_test in [0.5, 1.5, 2.0]:
    y_test = analytical_solution(t, y0_test, k)
    ax2.plot(t, y_test, '--', color=COLORS['red'], linewidth=1.5, alpha=0.5)

ax2.set_xlabel('t')
ax2.set_ylabel('y')
ax2.set_title('Direction Field and Solution Curves')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

### 🔬 Explore

Try these ODEs:
1. Logistic growth: dy/dt = ky(1 - y/M) (population with carrying capacity)
2. Newton's cooling: dT/dt = -k(T - T_env)
3. RC circuit: dV/dt = -(1/RC)V + E/RC

### 🎯 Practice Question

**Q:** Solve dy/dt = 2ty with y(0) = 1 using separation of variables.

<details><summary>Hint 1</summary>
Separate: dy/y = 2t dt
</details>

<details><summary>Hint 2</summary>
Integrate both sides: ln|y| = t² + C
</details>

<details><summary>Answer</summary>
dy/y = 2t dt
∫dy/y = ∫2t dt
ln|y| = t² + C
y = Ae^(t²)

Using y(0) = 1: 1 = Ae^0 → A = 1

**Solution: y(t) = e^(t²)**
</details>

---

## 2. Second-Order Linear ODEs

### 📖 Concept

**Second-order linear ODE:**
```
y'' + p(t)y' + q(t)y = f(t)
```

**Homogeneous case** (f(t) = 0): Use characteristic equation
- For constant coefficients: y'' + ay' + by = 0
- Characteristic equation: r² + ar + b = 0
- Solutions depend on roots:
  - Two real roots r₁, r₂: y = c₁e^(r₁t) + c₂e^(r₂t)
  - Repeated root r: y = (c₁ + c₂t)e^(rt)
  - Complex roots α ± iβ: y = e^(αt)(c₁cos(βt) + c₂sin(βt))

**Physical interpretation:** Oscillators, waves, damped systems

**Connection to GR:** Wave equations for gravitational waves are second-order PDEs. Geodesic deviation equation is second-order.

### 💻 Code Example

```python
# Simple harmonic oscillator: y'' + ω²y = 0
def harmonic_oscillator(y, t, omega=1.0):
    """
    Convert 2nd order ODE to system of 1st order:
    y'' + ω²y = 0

    Let v = y', then:
    y' = v
    v' = -ω²y
    """
    y_val, v = y
    dydt = v
    dvdt = -omega**2 * y_val
    return [dydt, dvdt]

# Damped harmonic oscillator: y'' + 2γy' + ω²y = 0
def damped_oscillator(y, t, gamma=0.1, omega=1.0):
    """
    y' = v
    v' = -2γv - ω²y
    """
    y_val, v = y
    dydt = v
    dvdt = -2*gamma*v - omega**2 * y_val
    return [dydt, dvdt]

# Solve undamped oscillator
omega = 2.0  # Angular frequency
y0 = [1.0, 0.0]  # Initial position y(0)=1, velocity y'(0)=0
t = np.linspace(0, 10, 500)  # np.linspace() creates evenly spaced array between start and end

y_undamped = odeint(harmonic_oscillator, y0, t, args=(omega,))  # odeint() solves ODE system with given initial conditions

# Analytical solution: y(t) = cos(ωt) for these initial conditions
y_analytical = np.cos(omega * t)  # np.cos() computes cosine (element-wise for arrays)

print("Simple Harmonic Oscillator: y'' + ω²y = 0")
print(f"Angular frequency ω = {omega}")
print(f"Period T = 2π/ω = {2*np.pi/omega:.3f}")
print(f"Initial conditions: y(0) = {y0[0]}, y'(0) = {y0[1]}")
print(f"\nAnalytical solution: y(t) = cos(ωt)")

# Solve damped oscillator
gamma = 0.2
y_damped = odeint(damped_oscillator, y0, t, args=(gamma, omega))  # odeint() solves ODE system with given initial conditions

print(f"\nDamped Oscillator: y'' + 2γy' + ω²y = 0")
print(f"Damping coefficient γ = {gamma}")
print(f"Damping ratio ζ = γ/ω = {gamma/omega:.3f}")

if gamma < omega:
    print("Underdamped: oscillates with decaying amplitude")
elif gamma == omega:
    print("Critically damped: fastest return to equilibrium")
else:
    print("Overdamped: slow return without oscillation")
```

### 📊 Visualization

```python
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Undamped oscillator
ax1.plot(t, y_undamped[:, 0], color=COLORS['blue'], linewidth=2, label='Position y(t)')
ax1.plot(t, y_undamped[:, 1]/omega, color=COLORS['red'], linewidth=2, label="Velocity y'(t)/ω")
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.set_xlabel('t')
ax1.set_ylabel('y')
ax1.set_title(f'Undamped Harmonic Oscillator: y\'\' + {omega**2}y = 0')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Damped oscillator - underdamped
gamma_under = 0.2
y_under = odeint(damped_oscillator, y0, t, args=(gamma_under, omega))  # odeint() solves ODE system with given initial conditions
envelope = np.exp(-gamma_under * t)  # np.exp() computes exponential e^x

ax2.plot(t, y_under[:, 0], color=COLORS['blue'], linewidth=2, label='Position y(t)')
ax2.plot(t, envelope, '--', color=COLORS['red'], linewidth=1.5, label='Envelope e^(-γt)')
ax2.plot(t, -envelope, '--', color=COLORS['red'], linewidth=1.5)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.set_xlabel('t')
ax2.set_ylabel('y')
ax2.set_title(f'Underdamped (γ = {gamma_under}): Oscillation with decay')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Comparison of damping regimes
gamma_critical = omega
gamma_over = 1.5 * omega

y_critical = odeint(damped_oscillator, y0, t, args=(gamma_critical, omega))  # odeint() solves ODE system with given initial conditions
y_over = odeint(damped_oscillator, y0, t, args=(gamma_over, omega))  # odeint() solves ODE system with given initial conditions

ax3.plot(t, y_under[:, 0], color=COLORS['blue'], linewidth=2,
         label=f'Underdamped (γ={gamma_under})')
ax3.plot(t, y_critical[:, 0], color=COLORS['green'], linewidth=2,
         label=f'Critical (γ={gamma_critical})')
ax3.plot(t, y_over[:, 0], color=COLORS['red'], linewidth=2,
         label=f'Overdamped (γ={gamma_over})')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.set_xlabel('t')
ax3.set_ylabel('y')
ax3.set_title('Comparison of Damping Regimes')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

### 🎯 Practice Question

**Q:** Solve y'' - 3y' + 2y = 0 with y(0) = 1, y'(0) = 0.

<details><summary>Hint 1</summary>
Use characteristic equation: r² - 3r + 2 = 0
</details>

<details><summary>Hint 2</summary>
Factor: (r - 1)(r - 2) = 0, so r₁ = 1, r₂ = 2
</details>

<details><summary>Answer</summary>
General solution: y = c₁e^t + c₂e^(2t)

Apply initial conditions:
- y(0) = 1: c₁ + c₂ = 1
- y'(0) = 0: c₁ + 2c₂ = 0

Solving: c₂ = -c₁ and c₁ - c₁ = 1 → c₁ = 2, c₂ = -1

**Solution: y(t) = 2e^t - e^(2t)**
</details>

---

## 3. Systems of ODEs

### 📖 Concept

**System of first-order ODEs:**
```
dx/dt = f(x, y)
dy/dt = g(x, y)
```

**Matrix form:** d**r**/dt = A**r** (linear systems)

**Eigenvalue method:**
1. Find eigenvalues λ and eigenvectors **v** of matrix A
2. Solutions have form **r**(t) = c**v**e^(λt)
3. General solution is linear combination

**Connection to GR:** Field equations involve coupled systems. Geodesics in 4D spacetime form a system of ODEs.

### 💻 Code Example

```python
# Predator-prey (Lotka-Volterra) system
def lotka_volterra(y, t, alpha=1.0, beta=0.1, delta=0.075, gamma=1.5):
    """
    Predator-prey model:
    dx/dt = αx - βxy  (prey growth - predation)
    dy/dt = δxy - γy  (predation benefit - predator death)

    x = prey population
    y = predator population
    """
    x, y_pred = y
    dxdt = alpha * x - beta * x * y_pred
    dydt = delta * x * y_pred - gamma * y_pred
    return [dxdt, dydt]

# Initial conditions
x0 = 10  # Initial prey
y0 = 5   # Initial predators
y_init = [x0, y0]

t = np.linspace(0, 50, 1000)  # np.linspace() creates evenly spaced array between start and end
solution = odeint(lotka_volterra, y_init, t)  # odeint() solves ODE system with given initial conditions

x_prey = solution[:, 0]
y_predator = solution[:, 1]

print("Lotka-Volterra Predator-Prey Model")
print("dx/dt = αx - βxy  (prey)")
print("dy/dt = δxy - γy  (predators)")
print(f"\nInitial populations: {x0} prey, {y0} predators")
print(f"\nPeriodic oscillations observed")
print(f"Prey peak: {x_prey.max():.1f}")
print(f"Predator peak: {y_predator.max():.1f}")
```

### 📊 Visualization

```python
fig = plt.figure(figsize=(14, 5))  # plt.figure() creates a new figure for plotting

# Time series
ax1 = fig.add_subplot(131)
ax1.plot(t, x_prey, color=COLORS['green'], linewidth=2, label='Prey (x)')
ax1.plot(t, y_predator, color=COLORS['red'], linewidth=2, label='Predators (y)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Population')
ax1.set_title('Population Dynamics Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Phase portrait (x vs y)
ax2 = fig.add_subplot(132)
ax2.plot(x_prey, y_predator, color=COLORS['blue'], linewidth=2)
ax2.plot(x0, y0, 'o', color=COLORS['red'], markersize=10, label='Start')
ax2.set_xlabel('Prey (x)')
ax2.set_ylabel('Predators (y)')
ax2.set_title('Phase Portrait: Closed Orbit')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Direction field in phase space
ax3 = fig.add_subplot(133)
x_field = np.linspace(0, 30, 20)  # np.linspace() creates evenly spaced array between start and end
y_field = np.linspace(0, 15, 20)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x_field, y_field)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# Parameters
alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
U = alpha * X - beta * X * Y
V = delta * X * Y - gamma * Y

# Normalize
M = np.sqrt(U**2 + V**2)  # np.sqrt() computes square root
M[M == 0] = 1  # Avoid division by zero
U_norm = U / M
V_norm = V / M

ax3.quiver(X, Y, U_norm, V_norm, M, alpha=0.6, cmap='viridis')
ax3.plot(x_prey, y_predator, color=COLORS['red'], linewidth=2, label='Trajectory')
ax3.plot(x0, y0, 'o', color=COLORS['orange'], markersize=10)
ax3.set_xlabel('Prey (x)')
ax3.set_ylabel('Predators (y)')
ax3.set_title('Direction Field with Trajectory')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

### 🔬 Explore

Try these systems:
1. SIR epidemic model: dS/dt = -βSI, dI/dt = βSI - γI, dR/dt = γI
2. Van der Pol oscillator: x'' - μ(1-x²)x' + x = 0
3. Lorenz system (chaos): dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz

### 🎯 Practice Question

**Q:** For the linear system d**r**/dt = A**r** where A = [[0, 1], [-1, 0]], find the general solution.

<details><summary>Hint 1</summary>
Find eigenvalues: det(A - λI) = 0
</details>

<details><summary>Hint 2</summary>
λ² + 1 = 0 gives λ = ±i (pure imaginary)
</details>

<details><summary>Answer</summary>
Eigenvalues: λ = ±i
Eigenvectors: **v**₁ = [1, i], **v**₂ = [1, -i]

General solution involves sin(t) and cos(t):
x(t) = c₁cos(t) + c₂sin(t)
y(t) = -c₁sin(t) + c₂cos(t)

This represents **circular motion** in the phase plane!
</details>

---

## 4. Phase Portraits

### 📖 Concept

**Phase portrait:** Geometric representation of all solution trajectories in state space

**Critical points (equilibria):** Points where d**r**/dt = **0**

**Classification of equilibria** (for 2D linear systems):
- **Node:** Both eigenvalues real, same sign (stable if negative, unstable if positive)
- **Saddle:** Real eigenvalues, opposite signs (unstable)
- **Spiral:** Complex eigenvalues α ± iβ (stable if α < 0, unstable if α > 0)
- **Center:** Pure imaginary eigenvalues (neutrally stable, closed orbits)

**Connection to GR:** Phase space analysis helps understand dynamics of particles and fields in curved spacetime.

### 💻 Code Example

```python
# Analyze different types of critical points
def linear_system(y, t, A):
    """General 2D linear system dy/dt = Ay"""
    return A @ y    # @ is matrix multiplication operator (equivalent to np.dot for matrices)

# Different system matrices and their behaviors
systems = {
    'Stable node': np.array([[-1, 0], [0, -2]]),  # np.array() converts Python list/tuple to efficient numpy array
    'Unstable node': np.array([[1, 0], [0, 2]]),  # np.array() converts Python list/tuple to efficient numpy array
    'Saddle': np.array([[1, 0], [0, -1]]),  # np.array() converts Python list/tuple to efficient numpy array
    'Stable spiral': np.array([[-0.5, -1], [1, -0.5]]),  # np.array() converts Python list/tuple to efficient numpy array
    'Center': np.array([[0, -1], [1, 0]])  # np.array() converts Python list/tuple to efficient numpy array
}

for name, A in systems.items():
    eigenvalues = np.linalg.eigvals(A)  # np.linalg.eigvals() computes eigenvalues only
    print(f"\n{name}:")
    print(f"  Matrix A:\n  {A}")
    print(f"  Eigenvalues: {eigenvalues}")

    if np.all(np.isreal(eigenvalues)):
        if np.all(eigenvalues < 0):
            stability = "Stable"
        elif np.all(eigenvalues > 0):
            stability = "Unstable"
        else:
            stability = "Saddle (unstable)"
    else:
        real_part = eigenvalues[0].real
        if real_part < 0:
            stability = "Stable spiral"
        elif real_part > 0:
            stability = "Unstable spiral"
        else:
            stability = "Center (neutrally stable)"

    print(f"  Type: {stability}")
```

### 📊 Visualization

```python
# Visualize all equilibrium types
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

t = np.linspace(0, 10, 500)  # np.linspace() creates evenly spaced array between start and end

for idx, (name, A) in enumerate(systems.items()):
    ax = axes[idx]

    # Draw multiple trajectories
    initial_conditions = [
        [1, 0], [0, 1], [-1, 0], [0, -1],
        [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]
    ]

    for y0 in initial_conditions:
        sol = odeint(linear_system, y0, t, args=(A,))  # odeint() solves ODE system with given initial conditions
        ax.plot(sol[:, 0], sol[:, 1], color=COLORS['blue'], linewidth=1.5, alpha=0.6)
        ax.plot(y0[0], y0[1], 'o', color=COLORS['red'], markersize=4)

    # Mark equilibrium
    ax.plot(0, 0, 'o', color=COLORS['orange'], markersize=12, zorder=10)

    # Direction field
    x = np.linspace(-2, 2, 15)  # np.linspace() creates evenly spaced array between start and end
    y = np.linspace(-2, 2, 15)  # np.linspace() creates evenly spaced array between start and end
    X, Y = np.meshgrid(x, y)  # np.meshgrid() creates coordinate matrices from coordinate vectors
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    M = np.sqrt(U**2 + V**2)  # np.sqrt() computes square root
    M[M == 0] = 1
    ax.quiver(X, Y, U/M, V/M, alpha=0.3)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Remove extra subplot
axes[-1].axis('off')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

### 🎯 Practice Question

**Q:** Classify the equilibrium at origin for d**r**/dt = A**r** where A = [[-2, 1], [1, -2]].

<details><summary>Hint 1</summary>
Find eigenvalues using det(A - λI) = 0
</details>

<details><summary>Hint 2</summary>
(-2-λ)² - 1 = 0 → λ² + 4λ + 3 = 0
</details>

<details><summary>Answer</summary>
Characteristic equation: λ² + 4λ + 3 = 0
(λ + 1)(λ + 3) = 0
Eigenvalues: λ₁ = -1, λ₂ = -3

Both eigenvalues are real and negative.
**Type: STABLE NODE**

All trajectories approach origin as t → ∞.
</details>

---

## 5. Numerical Methods

### 📖 Concept

When analytical solutions are impossible, we use numerical methods:

**Euler's method:** y_{n+1} = y_n + h·f(t_n, y_n)
- Simple but low accuracy (O(h))

**Runge-Kutta (RK4):** Uses weighted average of slopes
- Much better accuracy (O(h⁴))
- Industry standard for ODE solving

**Adaptive methods:** Adjust step size h automatically
- scipy.integrate.solve_ivp uses adaptive RK methods

**Connection to GR:** Numerical relativity simulations (black hole mergers, gravitational waves) rely on sophisticated ODE/PDE solvers.

### 💻 Code Example

```python
# Implement Euler's method and compare with RK4
def euler_step(f, t, y, h):
    """Single Euler step"""
    return y + h * f(y, t)

def rk4_step(f, t, y, h):
    """Single RK4 step"""
    k1 = f(y, t)
    k2 = f(y + h*k1/2, t + h/2)
    k3 = f(y + h*k2/2, t + h/2)
    k4 = f(y + h*k3, t + h)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# Test on y' = -y, y(0) = 1 (solution: y = e^(-t))
def f(y, t):
    return -y

y0 = 1.0
t_start = 0
t_end = 5
h_values = [0.5, 0.1, 0.05]

print("Comparing Euler and RK4 methods")
print("ODE: y' = -y, y(0) = 1")
print("Exact solution: y(t) = e^(-t)")
print()

for h in h_values:
    n_steps = int((t_end - t_start) / h)
    t = t_start

    # Euler method
    y_euler = y0
    for _ in range(n_steps):
        y_euler = euler_step(f, t, y_euler, h)
        t += h

    # RK4 method
    t = t_start
    y_rk4 = y0
    for _ in range(n_steps):
        y_rk4 = rk4_step(f, t, y_rk4, h)
        t += h

    # Exact
    y_exact = np.exp(-t_end)  # np.exp() computes exponential e^x

    error_euler = abs(y_euler - y_exact)
    error_rk4 = abs(y_rk4 - y_exact)

    print(f"Step size h = {h}:")
    print(f"  Euler:  y({t_end}) = {y_euler:.6f}, error = {error_euler:.2e}")
    print(f"  RK4:    y({t_end}) = {y_rk4:.6f}, error = {error_rk4:.2e}")
    print(f"  Exact:  y({t_end}) = {y_exact:.6f}")
    print(f"  RK4 is {error_euler/error_rk4:.1f}x more accurate\n")
```

### 📊 Visualization

```python
# Visualize convergence of numerical methods
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Solution with different step sizes
t_exact = np.linspace(0, 5, 200)  # np.linspace() creates evenly spaced array between start and end
y_exact = np.exp(-t_exact)  # np.exp() computes exponential e^x

ax1.plot(t_exact, y_exact, '-', color='k', linewidth=2, label='Exact', zorder=10)

colors = [COLORS['red'], COLORS['blue'], COLORS['green']]
for h, color in zip([1.0, 0.5, 0.25], colors):
    n_steps = int(5 / h)
    t_vals = [0]
    y_euler = [y0]
    y_rk4 = [y0]

    t = 0
    y_e = y0
    y_r = y0

    for _ in range(n_steps):
        y_e = euler_step(f, t, y_e, h)
        y_r = rk4_step(f, t, y_r, h)
        t += h
        t_vals.append(t)
        y_euler.append(y_e)
        y_rk4.append(y_r)

    ax1.plot(t_vals, y_euler, 'o-', color=color, linewidth=1.5,
             markersize=4, alpha=0.7, label=f'Euler h={h}')

ax1.set_xlabel('t')
ax1.set_ylabel('y')
ax1.set_title('Euler Method with Different Step Sizes')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Error vs step size (log-log plot)
h_range = np.logspace(-2, 0, 20)
errors_euler = []
errors_rk4 = []

for h in h_range:
    n_steps = int(5 / h)

    t = 0
    y_e = y0
    y_r = y0

    for _ in range(n_steps):
        y_e = euler_step(f, t, y_e, h)
        y_r = rk4_step(f, t, y_r, h)
        t += h

    errors_euler.append(abs(y_e - np.exp(-5)))  # np.exp() computes exponential e^x
    errors_rk4.append(abs(y_r - np.exp(-5)))  # np.exp() computes exponential e^x

ax2.loglog(h_range, errors_euler, 'o-', color=COLORS['red'],
           linewidth=2, label='Euler (O(h))')
ax2.loglog(h_range, errors_rk4, 's-', color=COLORS['blue'],
           linewidth=2, label='RK4 (O(h⁴))')

# Reference lines
ax2.loglog(h_range, h_range, '--', color='gray', alpha=0.5, label='O(h)')
ax2.loglog(h_range, h_range**4, '--', color='gray', alpha=0.5, label='O(h⁴)')

ax2.set_xlabel('Step size h')
ax2.set_ylabel('Error at t=5')
ax2.set_title('Convergence: Error vs Step Size')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

### 🎯 Practice Question

**Q:** Why is RK4 preferred over Euler's method despite being more computationally expensive per step?

<details><summary>Hint 1</summary>
Consider the accuracy order: Euler is O(h), RK4 is O(h⁴)
</details>

<details><summary>Hint 2</summary>
For the same accuracy, RK4 can use a much larger step size
</details>

<details><summary>Answer</summary>
RK4 is O(h⁴) accurate vs Euler's O(h). This means:

- To get error ε with Euler: need h ~ ε → N ~ 1/ε steps
- To get error ε with RK4: need h ~ ε^(1/4) → N ~ 1/ε^(1/4) steps

**Example:** For error 10⁻⁸:
- Euler: ~10⁸ steps
- RK4: ~100 steps

Even though RK4 is ~4x more work per step, it's vastly more efficient overall!
</details>

---

## 6. Boundary Value Problems

### 📖 Concept

**Initial value problem (IVP):** Specify conditions at one point (e.g., y(0) = 1, y'(0) = 0)

**Boundary value problem (BVP):** Specify conditions at two or more points (e.g., y(0) = 1, y(L) = 0)

**Shooting method:** Convert BVP to IVP
1. Guess initial slope y'(0) = s
2. Solve IVP
3. Check if y(L) matches boundary condition
4. Adjust s and repeat (Newton's method or bisection)

**Connection to GR:** Einstein's equations with boundary conditions (e.g., asymptotically flat spacetime, black hole horizons)

### 💻 Code Example

```python
from scipy.integrate import solve_bvp  # Boundary value problem solver

# Example: y'' + y = 0, y(0) = 0, y(π) = 0
# Solution: y(t) = A sin(t), boundary conditions give A = any value (eigenvalue problem)

def bvp_ode(t, y):
    """
    Convert y'' + y = 0 to first-order system:
    y₁ = y
    y₂ = y'

    dy₁/dt = y₂
    dy₂/dt = -y₁
    """
    return np.vstack([y[1], -y[0]])  # np.vstack() stacks arrays vertically (row-wise)

def boundary_conditions(ya, yb):
    """
    ya = y at left boundary (t=0)
    yb = y at right boundary (t=π)

    Want: y(0) = 0, y(π) = 0
    """
    return np.array([ya[0], yb[0]])  # np.array() converts Python list/tuple to efficient numpy array

# Initial mesh
t_mesh = np.linspace(0, np.pi, 10)  # np.linspace() creates evenly spaced array between start and end
y_guess = np.zeros((2, t_mesh.size))  # np.zeros() creates array filled with zeros
y_guess[0] = np.sin(t_mesh)  # Initial guess

# Solve BVP
sol = solve_bvp(bvp_ode, boundary_conditions, t_mesh, y_guess)

print("Boundary Value Problem: y'' + y = 0")
print("Boundary conditions: y(0) = 0, y(π) = 0")
print("\nThis is an eigenvalue problem!")
print("Solution: y(t) = A sin(t) for any constant A")
print(f"\nNumerical solution found: {sol.success}")
print(f"Peak value: {np.max(np.abs(sol.sol(t_mesh)[0])):.6f}")
```

### 📊 Visualization

```python
# Shooting method visualization
def shooting_method_demo():
    """Demonstrate shooting method for y'' + y = 0, y(0)=0, y(π)=1"""

    def ode_ivp(y, t):
        return [y[1], -y[0]]

    # Try different initial slopes
    slopes = [0.5, 1.0, 1.5, 2.0]
    t = np.linspace(0, np.pi, 100)  # np.linspace() creates evenly spaced array between start and end

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for s in slopes:
        y0 = [0, s]  # y(0)=0, y'(0)=s
        sol = odeint(ode_ivp, y0, t)  # odeint() solves ODE system with given initial conditions
        y_final = sol[-1, 0]

        label = f"y'(0)={s:.1f}, y(π)={y_final:.2f}"
        ax1.plot(t, sol[:, 0], linewidth=2, label=label)

    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.plot([0, np.pi], [0, 1], 'o', color=COLORS['red'],
             markersize=10, label='Boundary conditions')
    ax1.set_xlabel('t')
    ax1.set_ylabel('y(t)')
    ax1.set_title('Shooting Method: Try Different Initial Slopes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Converge to correct slope
    t_fine = np.linspace(0, np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
    t_mesh = np.linspace(0, np.pi, 10)  # np.linspace() creates evenly spaced array between start and end
    y_guess = np.zeros((2, t_mesh.size))  # np.zeros() creates array filled with zeros
    y_guess[0] = t_mesh / np.pi

    def bc(ya, yb):
        return np.array([ya[0], yb[0] - 1])  # y(0)=0, y(π)=1

    sol_bvp = solve_bvp(bvp_ode, bc, t_mesh, y_guess)
    y_solution = sol_bvp.sol(t_fine)

    ax2.plot(t_fine, y_solution[0], color=COLORS['blue'],
             linewidth=3, label='BVP solution')
    ax2.plot([0, np.pi], [0, 1], 'o', color=COLORS['red'],
             markersize=10, label='Boundary conditions')
    ax2.set_xlabel('t')
    ax2.set_ylabel('y(t)')
    ax2.set_title('Converged BVP Solution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
    plt.show()  # plt.show() displays the figure window

    print(f"Correct initial slope: y'(0) = {sol_bvp.sol(0)[1]:.6f}")

shooting_method_demo()
```

### 🎯 Practice Question

**Q:** For y'' = 6x with y(0) = 0 and y(1) = 1, find the solution.

<details><summary>Hint 1</summary>
Integrate twice: y'' = 6x → y' = 3x² + C₁ → y = x³ + C₁x + C₂
</details>

<details><summary>Hint 2</summary>
Use boundary conditions to find C₁ and C₂
</details>

<details><summary>Answer</summary>
General solution: y = x³ + C₁x + C₂

Boundary conditions:
- y(0) = 0: 0 + 0 + C₂ = 0 → C₂ = 0
- y(1) = 1: 1 + C₁ + 0 = 1 → C₁ = 0

**Solution: y(x) = x³**
</details>

---

## Summary and Next Steps

You've mastered differential equations! Key concepts:

✓ First-order ODEs (separable, linear, numerical solutions)
✓ Second-order linear ODEs (harmonic oscillators, damping)
✓ Systems of ODEs (predator-prey, coupled dynamics)
✓ Phase portraits and stability analysis
✓ Numerical methods (Euler, RK4, adaptive methods)
✓ Boundary value problems (shooting method)

**Connection to General Relativity:**
- Geodesic equation: d²x^μ/dτ² + Γ^μ_νλ (dx^ν/dτ)(dx^λ/dτ) = 0 (system of 2nd-order ODEs)
- Einstein field equations: 2nd-order nonlinear PDEs
- Schwarzschild solution: BVP with asymptotic flatness
- Gravitational wave equations: wave equations (PDEs)

**Next Lesson:** Curves and Surfaces (differential geometry begins!)

---
