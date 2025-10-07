# Lesson 8: Classical Mechanics

**Topics:** Lagrangian Mechanics, Euler-Lagrange Equations, Hamiltonian Mechanics, Principle of Least Action, Noether's Theorem
**Prerequisites:** Lessons 1-7 (Calculus, Differential Equations, Tensors)
**Time:** ~5-7 hours

---

## Table of Contents

1. [From Newton to Lagrange](#1-from-newton-to-lagrange)
2. [The Principle of Least Action](#2-the-principle-of-least-action)
3. [Euler-Lagrange Equations](#3-euler-lagrange-equations)
4. [Hamiltonian Mechanics](#4-hamiltonian-mechanics)
5. [Symmetries and Conservation Laws](#5-symmetries-and-conservation-laws)
6. [Noether's Theorem](#6-noethers-theorem)
7. [Applications to Field Theory](#7-applications-to-field-theory)
8. [Practice Questions](#practice-questions)

---

## How to Use This Lesson

**Three-Panel Setup:**
1. **Left:** This markdown file (your "textbook")
2. **Middle:** Python terminal for running code snippets
3. **Right:** AI assistant if you get stuck

Start your Python terminal with the project environment:
```bash
cd /Users/clarkcarter/Claude/personal/gr
source venv/bin/activate  # or venv/Scripts/activate on Windows
python
```

---

## 1. From Newton to Lagrange

### 📖 Concept

**Newton's approach:** F = ma in Cartesian coordinates
- Forces and accelerations
- Vector equations
- Coordinates must be Cartesian (or deal with fictitious forces)

**Lagrange's approach:** Reformulate mechanics using energy
- Uses generalized coordinates (any convenient coordinates!)
- Single scalar function (the Lagrangian) encodes all physics
- Works in curved spaces and constrained systems
- Foundation for quantum mechanics and field theory

**The Lagrangian:**
```
L = T - V
```
where T is kinetic energy and V is potential energy.

**Key idea:** Nature chooses the path that makes the **action** stationary:
```
S = ∫ L dt
```

**Why this matters for GR:**
- Einstein's field equations come from varying an action
- Geodesics are paths that extremize proper time
- The action principle is more fundamental than force laws
- Generalizes naturally to curved spacetime

---

### 💻 Code Example: Simple Harmonic Oscillator

Let's compare Newton's and Lagrange's approaches:

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
from scipy.integrate import odeint  # ODE solver for initial value problems
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# System parameters
m = 1.0  # mass
k = 1.0  # spring constant
omega = np.sqrt(k/m)  # natural frequency

# NEWTON'S APPROACH: F = ma
# mx'' + kx = 0

def newton_ode(y, t, m, k):
    """
    Newton's equation: F = ma
    y = [x, v]
    """
    x, v = y
    dxdt = v
    dvdt = -k/m * x  # a = F/m = -kx/m
    return [dxdt, dvdt]

# LAGRANGE'S APPROACH: L = T - V
# T = (1/2)mv², V = (1/2)kx²
# L = (1/2)mv² - (1/2)kx²

def lagrange_to_ode(y, t, m, k):
    """
    Euler-Lagrange equation gives same result
    y = [x, v]
    """
    x, v = y
    dxdt = v
    # From E-L equation: d/dt(∂L/∂v) - ∂L/∂x = 0
    # d/dt(mv) - (-kx) = 0
    # ma = -kx
    dvdt = -k/m * x
    return [dxdt, dvdt]

# Solve both (they're identical, just different derivations!)
t = np.linspace(0, 10, 500)  # np.linspace() creates evenly spaced array between start and end
y0 = [1.0, 0.0]  # Initial: x=1, v=0

sol_newton = odeint(newton_ode, y0, t, args=(m, k))  # odeint() solves ODE system with given initial conditions
sol_lagrange = odeint(lagrange_to_ode, y0, t, args=(m, k))  # odeint() solves ODE system with given initial conditions

print("Simple Harmonic Oscillator")
print(f"Mass m = {m}, Spring constant k = {k}")
print(f"Angular frequency ω = √(k/m) = {omega:.3f}")
print(f"\nBoth approaches give identical results!")
print(f"Position difference: {np.max(np.abs(sol_newton[:, 0] - sol_lagrange[:, 0])):.10f}")

# Plot
plt.figure(figsize=(12, 4))  # plt.figure() creates a new figure for plotting

plt.subplot(1, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(t, sol_newton[:, 0], color=COLORS['blue'], linewidth=2, label='Newton: F=ma')  # plt.plot() draws line plot
plt.plot(t, sol_lagrange[:, 0], color=COLORS['orange'], linewidth=2,  # plt.plot() draws line plot
         linestyle='--', label='Lagrange: δS=0')
plt.xlabel('Time t')  # plt.xlabel() sets x-axis label
plt.ylabel('Position x')  # plt.ylabel() sets y-axis label
plt.title('Position vs Time')  # plt.title() sets plot title
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

plt.subplot(1, 2, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
# Phase space plot
plt.plot(sol_newton[:, 0], sol_newton[:, 1], color=COLORS['green'], linewidth=2)  # plt.plot() draws line plot
plt.xlabel('Position x')  # plt.xlabel() sets x-axis label
plt.ylabel('Velocity v')  # plt.ylabel() sets y-axis label
plt.title('Phase Space Trajectory')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.axis('equal')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🔬 Explore: Generalized Coordinates

The power of Lagrangian mechanics: use ANY coordinates!

```python
# Pendulum: Natural to use angle θ instead of Cartesian (x,y)

def pendulum_lagrangian_coords(y, t, L, g):
    """
    Pendulum using angle coordinate θ
    L = T - V = (1/2)mL²θ'² - mgL(1-cos θ)
    """
    theta, theta_dot = y

    # Euler-Lagrange equation
    theta_ddot = -(g/L) * np.sin(theta)  # np.sin() computes sine (element-wise for arrays)

    return [theta_dot, theta_ddot]

# Parameters
L_pendulum = 1.0  # length
g = 9.81  # gravity

# Solve
t = np.linspace(0, 10, 500)  # np.linspace() creates evenly spaced array between start and end
y0 = [np.pi/4, 0]  # 45 degrees, at rest

sol = odeint(pendulum_lagrangian_coords, y0, t, args=(L_pendulum, g))  # odeint() solves ODE system with given initial conditions

plt.figure(figsize=(10, 4))  # plt.figure() creates a new figure for plotting

plt.subplot(1, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(t, np.degrees(sol[:, 0]), color=COLORS['blue'], linewidth=2)  # plt.plot() draws line plot
plt.xlabel('Time t (s)')  # plt.xlabel() sets x-axis label
plt.ylabel('Angle θ (degrees)')  # plt.ylabel() sets y-axis label
plt.title('Pendulum Motion')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot

plt.subplot(1, 2, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
# Animate pendulum position
for i in range(0, len(t), 20):
    theta = sol[i, 0]
    x = L_pendulum * np.sin(theta)  # np.sin() computes sine (element-wise for arrays)
    y = -L_pendulum * np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
    alpha = 0.2 + 0.8 * (i / len(t))
    plt.plot([0, x], [0, y], 'o-', color=COLORS['orange'], alpha=alpha, markersize=8)  # plt.plot() draws line plot

plt.xlabel('x')  # plt.xlabel() sets x-axis label
plt.ylabel('y')  # plt.ylabel() sets y-axis label
plt.title('Pendulum Trajectory')  # plt.title() sets plot title
plt.axis('equal')
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("\nNote: Using angle θ is much simpler than Cartesian (x,y) with constraints!")
```

---

## 2. The Principle of Least Action

### 📖 Concept

The **action** S is a functional (function of a function) that assigns a number to each possible path:

```
S[q(t)] = ∫[t₁ to t₂] L(q, q̇, t) dt
```

where:
- q(t) is the generalized coordinate (position, angle, etc.)
- q̇(t) is the generalized velocity
- L is the Lagrangian

**Principle of Least Action (Hamilton's Principle):**
The actual path taken by the system is the one that makes S **stationary** (usually a minimum):

```
δS = 0
```

**Mathematical statement:** Among all paths q(t) with fixed endpoints q(t₁) and q(t₂), the physical path is the one where small variations don't change S to first order.

**Why "stationary" not "minimum"?**
- Usually it IS a minimum
- Sometimes it's a saddle point
- In relativity, timelike geodesics maximize proper time!

---

### 💻 Code Example: Visualizing Different Paths

```python
# Compare different paths and their actions

def compute_action(t, q, L_func):
    """
    Compute action S = ∫ L dt for a path q(t)
    """
    q_dot = np.gradient(q, t)
    L_values = L_func(q, q_dot)
    S = np.trapz(L_values, t)
    return S

def harmonic_lagrangian(q, q_dot, m=1.0, k=1.0):
    """L = T - V = (1/2)mv² - (1/2)kx²"""
    T = 0.5 * m * q_dot**2
    V = 0.5 * k * q**2
    return T - V

# Time grid
t = np.linspace(0, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end

# Boundary conditions
q0, qf = 0.0, 0.0  # Start and end at origin

# True solution (minimize action)
q_true = np.sin(t)  # np.sin() computes sine (element-wise for arrays)

# Alternative paths (satisfy boundary conditions)
q_path1 = np.sin(t)  # np.sin() computes sine (element-wise for arrays)
q_path2 = t * (2*np.pi - t) / (2*np.pi)  # Parabola
q_path3 = np.sin(2*t) / 2  # Different frequency
q_path4 = t * (2*np.pi - t) * np.sin(t) / (2*np.pi)  # Mixed

paths = [q_path1, q_path2, q_path3, q_path4]
path_labels = ['sin(t) [TRUE]', 't(2π-t)/(2π)', 'sin(2t)/2', 'Mixed']
colors_paths = [COLORS['green'], COLORS['blue'], COLORS['orange'], COLORS['purple']]

# Compute actions
actions = []
for q in paths:
    S = compute_action(t, q, harmonic_lagrangian)
    actions.append(S)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Paths
for i, (q, label, color) in enumerate(zip(paths, path_labels, colors_paths)):
    linewidth = 3 if i == 0 else 2
    linestyle = '-' if i == 0 else '--'
    ax1.plot(t, q, color=color, linewidth=linewidth, linestyle=linestyle, label=label)

ax1.scatter([t[0], t[-1]], [q0, qf], color=COLORS['red'], s=100, zorder=5,
           label='Fixed endpoints')
ax1.set_xlabel('Time t')
ax1.set_ylabel('Position q(t)')
ax1.set_title('Different Paths with Same Endpoints')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)

# Right: Actions (bar chart)
bars = ax2.bar(range(len(actions)), actions, color=colors_paths, edgecolor='black', linewidth=2)
ax2.set_xticks(range(len(actions)))
ax2.set_xticklabels(['Path ' + str(i+1) for i in range(len(actions))], rotation=45)
ax2.set_ylabel('Action S')
ax2.set_title('Action for Each Path')
ax2.grid(True, alpha=0.3, axis='y')

# Highlight minimum
min_idx = np.argmin(actions)
bars[min_idx].set_edgecolor(COLORS['green'])
bars[min_idx].set_linewidth(4)

for i, (action, label) in enumerate(zip(actions, path_labels)):
    ax2.text(i, action + 0.5, f'S = {action:.2f}', ha='center', fontsize=9)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Actions for each path:")
for label, S in zip(path_labels, actions):
    marker = " ← MINIMUM" if S == min(actions) else ""
    print(f"{label:20s}: S = {S:8.4f}{marker}")
```

---

### 🎯 Practice Question #1

**Q:** For a free particle (V=0), what is the Lagrangian and what path minimizes the action?

<details>
<summary>💡 Hint 1</summary>

Free particle: L = T - V = T = (1/2)mv².
</details>

<details>
<summary>💡 Hint 2</summary>

What path minimizes ∫ v² dt for fixed endpoints?
</details>

<details>
<summary>✅ Answer</summary>

**Lagrangian:** L = (1/2)mv²

**Path:** Straight line with constant velocity

For fixed endpoints, the path minimizing ∫ v² dt is uniform motion (constant v). This gives a straight line in space.

```python
# Free particle action
def free_particle_lagrangian(q, q_dot, m=1.0):
    return 0.5 * m * q_dot**2

# True solution: straight line q = q0 + v*t
```

This is Newton's first law derived from the action principle!
</details>

---

## 3. Euler-Lagrange Equations

### 📖 Concept

The condition δS = 0 leads to the **Euler-Lagrange equations** - the equations of motion in Lagrangian mechanics:

```
d/dt (∂L/∂q̇ⁱ) - ∂L/∂qⁱ = 0
```

for each generalized coordinate qⁱ.

**Derivation sketch:**
1. Vary the path: q(t) → q(t) + εη(t) where η(t₁) = η(t₂) = 0
2. Require δS = 0 for all variations η(t)
3. Use integration by parts and fundamental lemma of calculus of variations
4. Result: Euler-Lagrange equations

**Key terms:**
- **∂L/∂q̇ⁱ** = generalized momentum pᵢ conjugate to qⁱ
- **∂L/∂qⁱ** = generalized force

**For multiple coordinates:** One E-L equation per degree of freedom.

---

### 💻 Code Example: Deriving Equations of Motion

```python
import sympy as sp  # SymPy for symbolic mathematics

# Define symbolic variables
t = sp.symbols('t', real=True)  # symbols() creates symbolic variables
m, k, g, L = sp.symbols('m k g L', real=True, positive=True)  # symbols() creates symbolic variables

# Example 1: Harmonic Oscillator
print("=" * 60)
print("EXAMPLE 1: Harmonic Oscillator")
print("=" * 60)

x = sp.Function('x')(t)
x_dot = sp.diff(x, t)  # diff() computes symbolic derivative

# Lagrangian: L = (1/2)mv² - (1/2)kx²
T = sp.Rational(1, 2) * m * x_dot**2
V = sp.Rational(1, 2) * k * x**2
L_harmonic = T - V

print(f"\nKinetic energy: T = {T}")
print(f"Potential energy: V = {V}")
print(f"Lagrangian: L = {L_harmonic}")

# Apply Euler-Lagrange equation
p = sp.diff(L_harmonic, x_dot)  # Generalized momentum
print(f"\nGeneralized momentum: p = ∂L/∂ẋ = {p}")

dpdt = sp.diff(p, t)  # diff() computes symbolic derivative
print(f"Time derivative: dp/dt = {dpdt}")

force = sp.diff(L_harmonic, x)  # diff() computes symbolic derivative
print(f"Generalized force: ∂L/∂x = {force}")

# Euler-Lagrange equation
EL_eq = sp.simplify(dpdt - force)  # simplify() algebraically simplifies expression
print(f"\nEuler-Lagrange: d/dt(∂L/∂ẋ) - ∂L/∂x = {EL_eq} = 0")
print(f"Equation of motion: {sp.diff(x, t, 2)} = {-k/m * x}")  # diff() computes symbolic derivative

# Example 2: Pendulum
print("\n" + "=" * 60)
print("EXAMPLE 2: Pendulum")
print("=" * 60)

theta = sp.Function('theta')(t)
theta_dot = sp.diff(theta, t)  # diff() computes symbolic derivative

# Lagrangian: L = (1/2)mL²θ'² - mgL(1-cosθ)
T_pend = sp.Rational(1, 2) * m * L**2 * theta_dot**2
V_pend = m * g * L * (1 - sp.cos(theta))
L_pend = T_pend - V_pend

print(f"\nLagrangian: L = {L_pend}")

p_theta = sp.diff(L_pend, theta_dot)  # diff() computes symbolic derivative
print(f"\nGeneralized momentum: p_θ = ∂L/∂θ̇ = {p_theta}")

dpdt = sp.diff(p_theta, t)  # diff() computes symbolic derivative
force_theta = sp.diff(L_pend, theta)  # diff() computes symbolic derivative

EL_pend = sp.simplify(dpdt - force_theta)  # simplify() algebraically simplifies expression
print(f"\nEuler-Lagrange equation:")
print(f"{sp.diff(theta, t, 2)} = {-g/L * sp.sin(theta)}")  # diff() computes symbolic derivative
print(f"\nSmall angle approximation (sin θ ≈ θ):")
print(f"θ̈ = -(g/L)θ  [Simple harmonic motion!]")
```

---

### 📊 Visualization: Lagrangian Phase Space

```python
# Visualize Lagrangian and phase space for different systems

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# System 1: Harmonic Oscillator
x_vals = np.linspace(-2, 2, 100)  # np.linspace() creates evenly spaced array between start and end
v_vals = np.linspace(-2, 2, 100)  # np.linspace() creates evenly spaced array between start and end
X, V = np.meshgrid(x_vals, v_vals)  # np.meshgrid() creates coordinate matrices from coordinate vectors

m, k = 1.0, 1.0
T = 0.5 * m * V**2
V_pot = 0.5 * k * X**2
L = T - V_pot

ax = axes[0, 0]
contour = ax.contour(X, V, L, levels=15, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)
ax.set_xlabel('Position x')
ax.set_ylabel('Velocity v')
ax.set_title('Harmonic Oscillator: L(x,v) = ½mv² - ½kx²')
ax.grid(True, alpha=0.3)

# Add trajectory
t = np.linspace(0, 10, 200)  # np.linspace() creates evenly spaced array between start and end
x_traj = np.cos(t)  # np.cos() computes cosine (element-wise for arrays)
v_traj = -np.sin(t)  # np.sin() computes sine (element-wise for arrays)
ax.plot(x_traj, v_traj, color=COLORS['red'], linewidth=2, label='Trajectory')
ax.legend()

# System 2: Potential Energy Landscape
ax = axes[0, 1]
x_vals = np.linspace(-3, 3, 200)  # np.linspace() creates evenly spaced array between start and end
V_harmonic = 0.5 * k * x_vals**2
V_quartic = 0.1 * x_vals**4 - 0.5 * x_vals**2 + 0.5
V_double_well = 0.1 * (x_vals**2 - 1)**2

ax.plot(x_vals, V_harmonic, color=COLORS['blue'], linewidth=2, label='Harmonic: V = ½kx²')
ax.plot(x_vals, V_quartic, color=COLORS['orange'], linewidth=2, label='Quartic: V = 0.1x⁴ - 0.5x² + 0.5')
ax.plot(x_vals, V_double_well, color=COLORS['green'], linewidth=2, label='Double well')
ax.set_xlabel('Position x')
ax.set_ylabel('Potential Energy V')
ax.set_title('Different Potential Energy Functions')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 3])

# System 3: Pendulum phase space
ax = axes[1, 0]
theta_vals = np.linspace(-np.pi, np.pi, 20)  # np.linspace() creates evenly spaced array between start and end
omega_vals = np.linspace(-3, 3, 20)  # np.linspace() creates evenly spaced array between start and end

for theta0 in theta_vals[::2]:
    for omega0 in omega_vals[::2]:
        t_phase = np.linspace(0, 5, 100)  # np.linspace() creates evenly spaced array between start and end
        sol = odeint(pendulum_lagrangian_coords, [theta0, omega0], t_phase, args=(1.0, 9.81))  # odeint() solves ODE system with given initial conditions
        ax.plot(sol[:, 0], sol[:, 1], color=COLORS['blue'], alpha=0.3, linewidth=0.5)

ax.set_xlabel('Angle θ')
ax.set_ylabel('Angular velocity ω')
ax.set_title('Pendulum Phase Space')
ax.grid(True, alpha=0.3)
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-3, 3])

# System 4: Energy conservation
ax = axes[1, 1]
t = np.linspace(0, 10, 500)  # np.linspace() creates evenly spaced array between start and end
sol = odeint(pendulum_lagrangian_coords, [np.pi/4, 0], t, args=(1.0, 9.81))  # odeint() solves ODE system with given initial conditions

theta = sol[:, 0]
omega = sol[:, 1]

m_pend = 1.0
L_pend = 1.0
g_val = 9.81

KE = 0.5 * m_pend * L_pend**2 * omega**2
PE = m_pend * g_val * L_pend * (1 - np.cos(theta))  # np.cos() computes cosine (element-wise for arrays)
E_total = KE + PE

ax.plot(t, KE, color=COLORS['blue'], linewidth=2, label='Kinetic Energy T')
ax.plot(t, PE, color=COLORS['orange'], linewidth=2, label='Potential Energy V')
ax.plot(t, E_total, color=COLORS['green'], linewidth=2, linestyle='--', label='Total Energy E')
ax.set_xlabel('Time t')
ax.set_ylabel('Energy')
ax.set_title('Energy Conservation in Pendulum')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print(f"\nTotal energy variation: {np.std(E_total):.10f} (should be ~0)")
print("Energy is conserved! (up to numerical errors)")
```

---

### 🔬 Explore: Constraints and Lagrange Multipliers

```python
# Example: Bead on a rotating hoop (constraint: r = R)

print("Bead on Rotating Hoop")
print("=" * 60)
print("Constraint: Bead must stay on hoop of radius R")
print("Hoop rotates with angular velocity Ω")
print("\nGeneralized coordinate: θ (angle from vertical)")
print("Constraint built into choice of coordinates!")
print("\nLagrangian in spherical coords (r=R fixed):")
print("L = T - V = (1/2)m[R²θ̇² + R²sin²θ Ω²] - mgR(1-cosθ)")
```

---

## 4. Hamiltonian Mechanics

### 📖 Concept

The **Hamiltonian** H is obtained from the Lagrangian via **Legendre transformation**:

```
H(q, p) = p·q̇ - L(q, q̇)
```

where p = ∂L/∂q̇ is the generalized momentum.

**Hamilton's equations** (replace E-L equations):
```
dq/dt = ∂H/∂p
dp/dt = -∂H/∂q
```

**Key properties:**
- H is often (but not always) the total energy E = T + V
- Phase space formulation: (q, p) are independent variables
- Fundamental in quantum mechanics (Ĥ becomes the Hamiltonian operator)
- Preserves phase space volume (Liouville's theorem)
- Clearer connection to symmetries

**Why this matters for GR:**
- Hamilton-Jacobi formalism for geodesics
- ADM formulation of GR uses Hamiltonian approach
- Connection to quantum field theory

---

### 💻 Code Example: From Lagrangian to Hamiltonian

```python
import sympy as sp  # SymPy for symbolic mathematics

# Harmonic oscillator
t = sp.symbols('t', real=True)  # symbols() creates symbolic variables
m, k = sp.symbols('m k', real=True, positive=True)  # symbols() creates symbolic variables

x = sp.Function('x')(t)
x_dot = sp.diff(x, t)  # diff() computes symbolic derivative

# Lagrangian
L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2

print("Harmonic Oscillator: Lagrangian → Hamiltonian")
print("=" * 60)
print(f"Lagrangian: L = {L}")

# Compute generalized momentum
p = sp.diff(L, x_dot)  # diff() computes symbolic derivative
print(f"\nGeneralized momentum: p = ∂L/∂ẋ = {p}")

# Solve for x_dot in terms of p
x_dot_of_p = sp.solve(p - sp.symbols('p'), x_dot)[0]  # symbols() creates symbolic variables
print(f"Invert: ẋ = {x_dot_of_p}")

# Hamiltonian: H = p·q̇ - L
p_sym = sp.symbols('p')  # symbols() creates symbolic variables
H = p_sym * (p_sym / m) - L.subs(x_dot, p_sym / m)
H = sp.simplify(H)  # simplify() algebraically simplifies expression

print(f"\nHamiltonian: H = p·ẋ - L = {H}")
print(f"H = T + V = (p²/2m) + (kx²/2)")

# Hamilton's equations
print("\nHamilton's Equations:")
print(f"dq/dt = ∂H/∂p = {sp.diff(H, p_sym)}")  # diff() computes symbolic derivative
print(f"dp/dt = -∂H/∂q = {-sp.diff(H, x)}")  # diff() computes symbolic derivative

print("\n" + "=" * 60)
print("These are equivalent to Newton's F = ma!")
```

---

### 📊 Visualization: Hamiltonian Flow

```python
# Visualize Hamiltonian flow in phase space

def hamiltonian_flow(state, t, H_func):
    """
    Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
    state = [q, p]
    """
    q, p = state
    dq_dt, dp_dt = H_func(q, p)
    return [dq_dt, dp_dt]

def harmonic_H(q, p, m=1.0, k=1.0):
    """Hamiltonian for harmonic oscillator"""
    # H = p²/2m + kq²/2
    dq_dt = p / m
    dp_dt = -k * q
    return dq_dt, dp_dt

# Create phase space grid
q_vals = np.linspace(-2, 2, 20)  # np.linspace() creates evenly spaced array between start and end
p_vals = np.linspace(-2, 2, 20)  # np.linspace() creates evenly spaced array between start and end
Q, P = np.meshgrid(q_vals, p_vals)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# Compute vector field
m, k = 1.0, 1.0
dQ = P / m
dP = -k * Q

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Phase space flow (vector field)
ax1.quiver(Q, P, dQ, dP, alpha=0.6, color=COLORS['blue'])

# Add some trajectories
for E in [0.5, 1.0, 1.5, 2.0]:
    # Ellipse: E = p²/2m + kq²/2
    theta_circ = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
    q_traj = np.sqrt(2*E/k) * np.cos(theta_circ)  # np.cos() computes cosine (element-wise for arrays)
    p_traj = np.sqrt(2*E*m) * np.sin(theta_circ)  # np.sin() computes sine (element-wise for arrays)
    ax1.plot(q_traj, p_traj, color=COLORS['orange'], linewidth=2, alpha=0.8)

ax1.set_xlabel('Position q')
ax1.set_ylabel('Momentum p')
ax1.set_title('Hamiltonian Flow: Harmonic Oscillator')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Right: Energy contours
q_fine = np.linspace(-2, 2, 200)  # np.linspace() creates evenly spaced array between start and end
p_fine = np.linspace(-2, 2, 200)  # np.linspace() creates evenly spaced array between start and end
Q_fine, P_fine = np.meshgrid(q_fine, p_fine)  # np.meshgrid() creates coordinate matrices from coordinate vectors

H_values = P_fine**2 / (2*m) + k * Q_fine**2 / 2

contour = ax2.contourf(Q_fine, P_fine, H_values, levels=20, cmap='viridis', alpha=0.8)
plt.colorbar(contour, ax=ax2, label='Energy H')  # plt.colorbar() adds color scale bar to plot

ax2.set_xlabel('Position q')
ax2.set_ylabel('Momentum p')
ax2.set_title('Energy Contours (Conserved!)')
ax2.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Key insight: Hamiltonian flow preserves energy contours")
print("Each trajectory stays on a single energy surface")
```

---

### 🎯 Practice Question #2

**Q:** For a free particle (V=0), what is the Hamiltonian?

<details>
<summary>💡 Hint</summary>

Start with L = (1/2)mv², find p, then compute H = pq̇ - L.
</details>

<details>
<summary>✅ Answer</summary>

**H = p²/(2m)**

Derivation:
```
L = (1/2)mv²
p = ∂L/∂v = mv
v = p/m
H = pv - L = p(p/m) - (1/2)m(p/m)² = p²/m - p²/(2m) = p²/(2m)
```

This is just the kinetic energy! For conservative systems, H = T + V.
</details>

---

## 5. Symmetries and Conservation Laws

### 📖 Concept

**Symmetry** = transformation that leaves the physics unchanged

**Conservation law** = quantity that doesn't change with time

**Noether's Theorem** (next section) makes this precise: *Every continuous symmetry implies a conservation law.*

**Examples:**

| Symmetry | Conserved Quantity |
|----------|-------------------|
| Time translation (∂L/∂t = 0) | Energy |
| Space translation (homogeneity) | Momentum |
| Rotation (isotropy) | Angular momentum |
| Gauge symmetry | Electric charge |
| Lorentz boosts | 4-momentum |

**Cyclic coordinates:** If L doesn't depend on qⁱ (but may depend on q̇ⁱ), then qⁱ is **cyclic** or **ignorable**, and pᵢ = ∂L/∂q̇ⁱ is conserved.

---

### 💻 Code Example: Finding Conserved Quantities

```python
import sympy as sp  # SymPy for symbolic mathematics

# Example 1: Free particle in 1D
print("=" * 60)
print("EXAMPLE 1: Free Particle (translational symmetry)")
print("=" * 60)

t = sp.symbols('t', real=True)  # symbols() creates symbolic variables
m = sp.symbols('m', real=True, positive=True)  # symbols() creates symbolic variables
x = sp.Function('x')(t)
x_dot = sp.diff(x, t)  # diff() computes symbolic derivative

L_free = sp.Rational(1, 2) * m * x_dot**2

print(f"Lagrangian: L = {L_free}")
print(f"∂L/∂x = {sp.diff(L_free, x)}")  # diff() computes symbolic derivative
print("Since ∂L/∂x = 0, x is a cyclic coordinate")

p_x = sp.diff(L_free, x_dot)  # diff() computes symbolic derivative
print(f"\nConserved momentum: p = ∂L/∂ẋ = {p_x}")
print("Momentum is conserved!")

# Example 2: Central force (rotational symmetry)
print("\n" + "=" * 60)
print("EXAMPLE 2: Central Force (rotational symmetry)")
print("=" * 60)

r, theta = sp.symbols('r theta', real=True)  # symbols() creates symbolic variables
r_t = sp.Function('r')(t)
theta_t = sp.Function('theta')(t)
r_dot = sp.diff(r_t, t)  # diff() computes symbolic derivative
theta_dot = sp.diff(theta_t, t)  # diff() computes symbolic derivative

# Lagrangian in polar coordinates
# L = (1/2)m(ṙ² + r²θ̇²) - V(r)
V = sp.Function('V')(r_t)
L_central = sp.Rational(1, 2) * m * (r_dot**2 + r_t**2 * theta_dot**2) - V

print(f"Lagrangian: L = ½m(ṙ² + r²θ̇²) - V(r)")
print(f"\n∂L/∂θ = {sp.diff(L_central, theta_t)}")  # diff() computes symbolic derivative
print("Since ∂L/∂θ = 0, θ is a cyclic coordinate")

p_theta = sp.diff(L_central, theta_dot)  # diff() computes symbolic derivative
print(f"\nConserved angular momentum: L = ∂L/∂θ̇ = {p_theta}")
print("Angular momentum mr²θ̇ is conserved!")

# Example 3: Particle in uniform field (no time symmetry)
print("\n" + "=" * 60)
print("EXAMPLE 3: Driven Harmonic Oscillator (broken time symmetry)")
print("=" * 60)

omega, F0 = sp.symbols('omega F_0', real=True, positive=True)  # symbols() creates symbolic variables

x = sp.Function('x')(t)
x_dot = sp.diff(x, t)  # diff() computes symbolic derivative

# Time-dependent driving force
L_driven = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * m * omega**2 * x**2 + F0 * sp.cos(t) * x

print("Lagrangian: L = ½mẋ² - ½mω²x² + F₀cos(t)x")
print("\nTime dependence: ∂L/∂t ≠ 0")
print("Energy is NOT conserved (external driving force adds energy)")
```

---

### 📊 Visualization: Angular Momentum Conservation

```python
# Demonstrate angular momentum conservation in central force

def central_force_ode(state, t, k):
    """
    Central force: F = -kr
    In polar coords: L = ½m(ṙ² + r²θ̇²) - ½kr²
    """
    r, theta, r_dot, theta_dot = state

    # Angular momentum L = mr²θ̇ is conserved
    L_ang = r**2 * theta_dot  # m = 1

    # Equations of motion
    r_ddot = r * theta_dot**2 - k * r
    theta_ddot = -2 * r_dot * theta_dot / r

    return [r_dot, theta_dot, r_ddot, theta_ddot]

# Solve
t = np.linspace(0, 20, 1000)  # np.linspace() creates evenly spaced array between start and end
# Initial: r=1, theta=0, v_r=0, v_theta=1
state0 = [1.0, 0.0, 0.0, 1.0]
k = 1.0

sol = odeint(central_force_ode, state0, t, args=(k,))  # odeint() solves ODE system with given initial conditions

r = sol[:, 0]
theta = sol[:, 1]
r_dot = sol[:, 2]
theta_dot = sol[:, 3]

# Compute conserved quantities
L_angular = r**2 * theta_dot  # Should be constant
E_total = 0.5 * (r_dot**2 + r**2 * theta_dot**2) + 0.5 * k * r**2

# Convert to Cartesian for plotting
x = r * np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
y = r * np.sin(theta)  # np.sin() computes sine (element-wise for arrays)

# Plot
fig = plt.figure(figsize=(14, 10))  # plt.figure() creates a new figure for plotting

# Trajectory
ax1 = plt.subplot(2, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
ax1.plot(x, y, color=COLORS['blue'], linewidth=2)
ax1.plot([0], [0], 'o', color=COLORS['red'], markersize=10, label='Force center')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Orbital Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Angular momentum
ax2 = plt.subplot(2, 2, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
ax2.plot(t, L_angular, color=COLORS['green'], linewidth=2)
ax2.set_xlabel('Time t')
ax2.set_ylabel('Angular Momentum L')
ax2.set_title(f'Angular Momentum (conserved)\nVariation: {np.std(L_angular):.2e}')
ax2.grid(True, alpha=0.3)

# Energy
ax3 = plt.subplot(2, 2, 3)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
ax3.plot(t, E_total, color=COLORS['orange'], linewidth=2)
ax3.set_xlabel('Time t')
ax3.set_ylabel('Total Energy E')
ax3.set_title(f'Total Energy (conserved)\nVariation: {np.std(E_total):.2e}')
ax3.grid(True, alpha=0.3)

# Polar plot
ax4 = plt.subplot(2, 2, 4, projection='polar')  # plt.subplot() creates subplot in grid layout (rows, cols, position)
ax4.plot(theta, r, color=COLORS['purple'], linewidth=2)
ax4.set_title('Polar Plot')
ax4.grid(True)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print(f"Angular momentum L: mean = {np.mean(L_angular):.6f}, std = {np.std(L_angular):.2e}")
print(f"Total energy E: mean = {np.mean(E_total):.6f}, std = {np.std(E_total):.2e}")
print("\nBoth quantities conserved to numerical precision!")
```

---

## 6. Noether's Theorem

### 📖 Concept

**Noether's Theorem (1918):** Every differentiable symmetry of the action corresponds to a conservation law.

**Precise statement:** If the Lagrangian is invariant under a continuous transformation with parameter ε:
```
q → q + εδq
t → t + εδt
```
then the quantity:
```
Q = (∂L/∂q̇)δq - Hδt
```
is conserved (dQ/dt = 0).

**Examples:**

**1. Time translation symmetry** (δt = 1, δq = 0):
```
L(q, q̇, t) = L(q, q̇)  [no explicit time dependence]
→ Energy conserved: H = constant
```

**2. Space translation symmetry** (δq = 1, δt = 0):
```
L doesn't depend on q (only on q̇)
→ Momentum conserved: p = ∂L/∂q̇ = constant
```

**3. Rotational symmetry** (δθ = 1):
```
L invariant under rotations
→ Angular momentum conserved
```

**Why this matters for GR:**
- Energy-momentum conservation from spacetime translation symmetry
- General covariance ⟹ constraint equations (not conservation laws!)
- Difficulty defining global energy in curved spacetime

---

### 💻 Code Example: Verifying Noether's Theorem

```python
# Demonstrate Noether's theorem numerically

print("NOETHER'S THEOREM: Symmetry → Conservation Law")
print("=" * 70)

# System: Particle in potential V(x) = cos(x)
# This potential is periodic → translation symmetry by 2π

def periodic_potential_ode(state, t, V_period=2*np.pi):
    """
    L = ½ẋ² - V(x) where V(x+period) = V(x)
    For V(x) = cos(x), period = 2π
    """
    x, v = state
    V_x = np.sin(x)  # -dV/dx for V = cos(x)

    dx_dt = v
    dv_dt = V_x  # F = -dV/dx

    return [dx_dt, dv_dt]

# Solve
t = np.linspace(0, 20, 500)  # np.linspace() creates evenly spaced array between start and end
state0 = [0.5, 1.0]  # Initial position and velocity

sol = odeint(periodic_potential_ode, state0, t)  # odeint() solves ODE system with given initial conditions
x = sol[:, 0]
v = sol[:, 1]

# Energy (conserved due to time translation symmetry)
E = 0.5 * v**2 + np.cos(x)  # np.cos() computes cosine (element-wise for arrays)

# Plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Potential
x_plot = np.linspace(-2*np.pi, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
V_plot = np.cos(x_plot)  # np.cos() computes cosine (element-wise for arrays)

ax1.plot(x_plot, V_plot, color=COLORS['blue'], linewidth=2)
ax1.set_xlabel('Position x')
ax1.set_ylabel('Potential V(x)')
ax1.set_title('Periodic Potential: V(x) = cos(x)')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Add symmetry arrows
for shift in [-2*np.pi, 0, 2*np.pi]:
    ax1.axvline(x=shift, color=COLORS['red'], linestyle='--', alpha=0.5)
ax1.text(0, 1.2, 'Translation\nsymmetry', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Trajectory
ax2.plot(t, x, color=COLORS['green'], linewidth=2)
ax2.set_xlabel('Time t')
ax2.set_ylabel('Position x')
ax2.set_title('Position vs Time')
ax2.grid(True, alpha=0.3)

# Energy conservation
ax3.plot(t, E, color=COLORS['orange'], linewidth=2)
ax3.set_xlabel('Time t')
ax3.set_ylabel('Energy E')
ax3.set_title(f'Energy Conservation\n(Time symmetry → Energy conserved)\nΔE = {np.std(E):.2e}')
ax3.grid(True, alpha=0.3)

# Phase space
ax4.plot(x, v, color=COLORS['purple'], linewidth=2)
ax4.set_xlabel('Position x')
ax4.set_ylabel('Velocity v')
ax4.set_title('Phase Space')
ax4.grid(True, alpha=0.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print(f"\nTime translation symmetry (∂L/∂t = 0)")
print(f"→ Energy conserved: ΔE/E = {np.std(E)/np.mean(E):.2e}")
```

---

### 🔬 Explore: Breaking Symmetries

```python
# What happens when we break a symmetry?

def broken_symmetry_ode(state, t, driving_amplitude=0.5):
    """
    Add time-dependent driving force → breaks time translation symmetry
    L = ½ẋ² - cos(x) + A·sin(t)·x
    """
    x, v = state

    # Force including time-dependent driving
    F = np.sin(x) + driving_amplitude * np.sin(t)  # np.sin() computes sine (element-wise for arrays)

    return [v, F]

# Solve
sol_driven = odeint(broken_symmetry_ode, state0, t, args=(0.5,))  # odeint() solves ODE system with given initial conditions
x_driven = sol_driven[:, 0]
v_driven = sol_driven[:, 1]

# "Energy" (no longer conserved!)
E_driven = 0.5 * v_driven**2 + np.cos(x_driven)  # np.cos() computes cosine (element-wise for arrays)

plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

plt.subplot(1, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(t, E, color=COLORS['green'], linewidth=2, label='No driving (E conserved)')  # plt.plot() draws line plot
plt.plot(t, E_driven, color=COLORS['red'], linewidth=2, label='With driving (E not conserved)')  # plt.plot() draws line plot
plt.xlabel('Time t')  # plt.xlabel() sets x-axis label
plt.ylabel('Energy E')  # plt.ylabel() sets y-axis label
plt.title('Breaking Time Symmetry → Energy Not Conserved')  # plt.title() sets plot title
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

plt.subplot(1, 2, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(x, v, color=COLORS['green'], linewidth=2, label='No driving', alpha=0.7)  # plt.plot() draws line plot
plt.plot(x_driven, v_driven, color=COLORS['red'], linewidth=2, label='With driving', alpha=0.7)  # plt.plot() draws line plot
plt.xlabel('Position x')  # plt.xlabel() sets x-axis label
plt.ylabel('Velocity v')  # plt.ylabel() sets y-axis label
plt.title('Phase Space Comparison')  # plt.title() sets plot title
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Symmetry broken → Conservation law broken")
print(f"Energy variation (no driving): {np.std(E):.2e}")
print(f"Energy variation (with driving): {np.std(E_driven):.2e}")
```

---

### 🎯 Practice Question #3

**Q:** A particle moves in potential V(x) = x⁴. What symmetry does this system have, if any? What is conserved?

<details>
<summary>💡 Hint</summary>

Check if the Lagrangian depends explicitly on time.
</details>

<details>
<summary>✅ Answer</summary>

**Symmetry:** Time translation symmetry (∂L/∂t = 0)

**Conserved quantity:** Energy

The Lagrangian L = (1/2)mv² - x⁴ has no explicit time dependence, so energy E = T + V = (1/2)mv² + x⁴ is conserved.

Note: There's NO space translation symmetry (V depends on x) and NO parity symmetry (V(x) ≠ V(-x) for general x), so momentum is NOT conserved.
</details>

---

## 7. Applications to Field Theory

### 📖 Concept

Classical mechanics: Lagrangian L(q, q̇, t) depends on finite degrees of freedom

**Field theory:** Lagrangian density ℒ(φ, ∂_μφ, x) depends on fields φ(x,t) at each spacetime point

**Action for fields:**
```
S = ∫ ℒ(φ, ∂_μφ) d⁴x
```

**Euler-Lagrange equation for fields:**
```
∂_μ (∂ℒ/∂(∂_μφ)) - ∂ℒ/∂φ = 0
```

**Example:** Klein-Gordon equation (relativistic scalar field)
```
ℒ = (1/2)(∂_μφ)(∂^μφ) - (1/2)m²φ²
→ □φ + m²φ = 0
```

**Why this matters for GR:**
- General Relativity IS a field theory (metric g_μν is the field)
- Einstein-Hilbert action: S = ∫ R √(-g) d⁴x
- Varying S gives Einstein's field equations
- Matter fields add to the action

---

### 💻 Code Example: Scalar Field Lagrangian

```python
import sympy as sp  # SymPy for symbolic mathematics

print("FIELD THEORY: From Lagrangian Density to Field Equations")
print("=" * 70)

# Define spacetime coordinates and field
t, x = sp.symbols('t x', real=True)  # symbols() creates symbolic variables
phi = sp.Function('phi')(t, x)

# Derivatives
phi_t = sp.diff(phi, t)  # diff() computes symbolic derivative
phi_x = sp.diff(phi, x)  # diff() computes symbolic derivative

# Lagrangian density for Klein-Gordon field (1+1 D)
# ℒ = (1/2)(∂_t φ)² - (1/2)(∂_x φ)² - (1/2)m²φ²
m = sp.symbols('m', real=True, positive=True)  # symbols() creates symbolic variables

L_density = sp.Rational(1, 2) * phi_t**2 - sp.Rational(1, 2) * phi_x**2 - sp.Rational(1, 2) * m**2 * phi**2

print("Lagrangian density:")
print(f"ℒ = {L_density}")

# Apply Euler-Lagrange equation for fields
# ∂/∂t (∂ℒ/∂φ_t) + ∂/∂x (∂ℒ/∂φ_x) - ∂ℒ/∂φ = 0

term1 = sp.diff(sp.diff(L_density, phi_t), t)  # diff() computes symbolic derivative
term2 = sp.diff(sp.diff(L_density, phi_x), x)  # diff() computes symbolic derivative
term3 = sp.diff(L_density, phi)  # diff() computes symbolic derivative

field_eq = sp.simplify(term1 + term2 - term3)  # simplify() algebraically simplifies expression

print("\nEuler-Lagrange equation:")
print(f"∂_t(∂ℒ/∂φ_t) + ∂_x(∂ℒ/∂φ_x) - ∂ℒ/∂φ = 0")
print(f"\nField equation: {field_eq} = 0")
print("\nThis is the Klein-Gordon equation: ∂²φ/∂t² - ∂²φ/∂x² + m²φ = 0")
print("Or in covariant form: □φ + m²φ = 0")

# Einstein-Hilbert action (symbolic)
print("\n" + "=" * 70)
print("GENERAL RELATIVITY ACTION")
print("=" * 70)
print("Einstein-Hilbert action:")
print("S = (1/16πG) ∫ R √(-g) d⁴x")
print("\nwhere:")
print("  R = Ricci scalar (curvature)")
print("  g = det(g_μν) (metric determinant)")
print("\nVariation δS/δg_μν = 0 gives Einstein's field equations:")
print("  G_μν = R_μν - ½g_μν R = 0  (vacuum)")
print("  G_μν = 8πG/c⁴ T_μν  (with matter)")
```

---

### 📊 Visualization: Field Configuration Space

```python
# Visualize field configurations and their energies

# 1D scalar field φ(x)
x_vals = np.linspace(0, 10, 100)  # np.linspace() creates evenly spaced array between start and end

# Different field configurations
phi_1 = np.sin(x_vals)  # np.sin() computes sine (element-wise for arrays)
phi_2 = np.sin(2*x_vals)  # np.sin() computes sine (element-wise for arrays)
phi_3 = np.exp(-((x_vals - 5)**2) / 2)  # Gaussian
phi_4 = np.tanh(x_vals - 5)  # Kink

configs = [phi_1, phi_2, phi_3, phi_4]
labels = ['sin(x)', 'sin(2x)', 'Gaussian', 'Kink']
colors_field = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]

# Compute "energies" (gradient energy + potential energy)
m = 1.0

energies = []
for phi in configs:
    dphi_dx = np.gradient(phi, x_vals)
    # E = ∫ [(1/2)(∂φ/∂x)² + (1/2)m²φ²] dx
    E_density = 0.5 * dphi_dx**2 + 0.5 * m**2 * phi**2
    E_total = np.trapz(E_density, x_vals)
    energies.append(E_total)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Field configurations
for phi, label, color in zip(configs, labels, colors_field):
    ax1.plot(x_vals, phi, color=color, linewidth=2, label=label)

ax1.set_xlabel('Position x')
ax1.set_ylabel('Field φ(x)')
ax1.set_title('Different Field Configurations')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)

# Right: Energies
bars = ax2.bar(labels, energies, color=colors_field, edgecolor='black', linewidth=2)
ax2.set_ylabel('Total Energy E')
ax2.set_title('Energy of Each Configuration')
ax2.grid(True, alpha=0.3, axis='y')

for i, (E, label) in enumerate(zip(energies, labels)):
    ax2.text(i, E + 0.5, f'{E:.2f}', ha='center', fontsize=10)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Field configurations with higher gradients have higher energy")
print("(More rapid spatial variation costs energy)")
```

---

## 8. Practice Questions

### Lagrangian Mechanics

**Q1:** For a particle sliding on a frictionless hemisphere (constraint: r = R), what is the Lagrangian in terms of angle θ from the vertical?

<details>
<summary>💡 Hint</summary>

Use spherical coordinates with r fixed at R. T = (1/2)m(R²θ̇² + R²sin²θ φ̇²).
</details>

<details>
<summary>✅ Answer</summary>

For motion in a vertical plane (φ = constant):

**L = (1/2)mR²θ̇² - mgR(1 - cosθ)**

where T = (1/2)mR²θ̇² (kinetic) and V = mgR(1-cosθ) (potential, measured from bottom).
</details>

---

**Q2:** What is the generalized momentum for a cyclic coordinate?

<details>
<summary>✅ Answer</summary>

For cyclic coordinate q_i (where ∂L/∂q_i = 0):

**p_i = ∂L/∂q̇_i = constant**

The generalized momentum conjugate to a cyclic coordinate is conserved.
</details>

---

### Hamiltonian Mechanics

**Q3:** For the Lagrangian L = (1/2)mẋ² - mgx, find the Hamiltonian.

<details>
<summary>💡 Hint</summary>

Find p = ∂L/∂ẋ, then H = pẋ - L.
</details>

<details>
<summary>✅ Answer</summary>

**H = p²/(2m) + mgx**

Derivation:
```
L = (1/2)mẋ² - mgx
p = ∂L/∂ẋ = mẋ  →  ẋ = p/m
H = pẋ - L = p·(p/m) - [(1/2)m(p/m)² - mgx]
H = p²/m - p²/(2m) + mgx = p²/(2m) + mgx
```

This is T + V (total energy).
</details>

---

**Q4:** In Hamiltonian mechanics, what is dH/dt if H doesn't explicitly depend on time?

<details>
<summary>✅ Answer</summary>

**dH/dt = 0** (H is conserved)

From Hamilton's equations:
```
dH/dt = ∂H/∂t + (∂H/∂q)(dq/dt) + (∂H/∂p)(dp/dt)
      = ∂H/∂t + (∂H/∂q)(∂H/∂p) - (∂H/∂p)(∂H/∂q)
      = ∂H/∂t
```

If ∂H/∂t = 0, then dH/dt = 0.
</details>

---

### Symmetries and Noether's Theorem

**Q5:** A system has Lagrangian L = (1/2)m(ẋ² + ẏ²) - mgy. What quantities are conserved?

<details>
<summary>💡 Hint</summary>

Check for time, x-translation, and y-translation symmetries.
</details>

<details>
<summary>✅ Answer</summary>

**Conserved quantities:**
1. **Energy E** (time translation: ∂L/∂t = 0)
2. **x-momentum p_x** (x-translation: ∂L/∂x = 0)

**Not conserved:**
- y-momentum (broken by gravity: ∂L/∂y ≠ 0)

```
E = (1/2)m(ẋ² + ẏ²) + mgy
p_x = mẋ
```
</details>

---

**Q6:** State Noether's theorem in one sentence.

<details>
<summary>✅ Answer</summary>

**Every continuous symmetry of the action corresponds to a conserved quantity.**

Or equivalently: Symmetries ↔ Conservation laws
</details>

---

### Applications

**Q7:** For a relativistic particle with action S = -mc² ∫ √(1 - v²/c²) dt, what is the Lagrangian?

<details>
<summary>✅ Answer</summary>

**L = -mc² √(1 - v²/c²)**

This gives the correct relativistic momentum p = γmv and energy E = γmc² when you apply Lagrangian mechanics!
</details>

---

**Q8:** In General Relativity, what action do we vary to get Einstein's field equations?

<details>
<summary>✅ Answer</summary>

**Einstein-Hilbert action:**

**S = (c³/16πG) ∫ R √(-g) d⁴x**

Varying with respect to the metric g_μν gives:

**G_μν = (8πG/c⁴) T_μν**

(Einstein's field equations)
</details>

---

## Summary: Why Lagrangian/Hamiltonian Mechanics Matters for GR

| Concept | Connection to GR |
|---------|-----------------|
| Action principle | Einstein's equations from Einstein-Hilbert action |
| Generalized coordinates | Arbitrary coordinate systems in curved spacetime |
| Geodesics | Freely falling particles extremize proper time |
| Symmetries | Spacetime symmetries → conserved quantities |
| Noether's theorem | Energy-momentum conservation from symmetries |
| Field theory | Metric g_μν as dynamical field |
| Hamiltonian formulation | ADM formalism, canonical quantum gravity |

---

## Next Steps

✅ Master deriving equations of motion from Lagrangians
✅ Practice finding conserved quantities from symmetries
✅ Understand Legendre transformation (L ↔ H)
✅ Apply Noether's theorem to simple systems
✅ **Ready for Lesson 9: Special Relativity** (Lorentz transformations, spacetime)

**Additional Resources:**
- Landau & Lifshitz: "Mechanics" (Classic text)
- Goldstein: "Classical Mechanics" (Advanced)
- Susskind's "Theoretical Minimum: Classical Mechanics" (Videos)
- 3Blue1Brown: "Brachistochrone problem" (Visual intuition)

**Need Help?** Use the AI assistant:
```python
from utils.ai_assistant import AIAssistant
assistant = AIAssistant()
assistant.set_lesson_context("Lesson 8", "Classical Mechanics",
                            ["Lagrangian", "Hamiltonian", "Noether", "action"])
assistant.ask("Why do we need the Lagrangian formulation if Newton's laws work fine?")
```

---

**Ready to continue?** → [Lesson 9: Special Relativity](../09_special_relativity/LESSON.md)
