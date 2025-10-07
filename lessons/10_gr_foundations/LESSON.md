# Lesson 10: General Relativity Foundations

**Topics:** Equivalence Principle, Einstein Field Equations, Stress-Energy Tensor, Geodesic Equation, Weak Field Limit, Linearized Gravity

**Prerequisites:** Lessons 1-9 (especially Riemannian Geometry, Tensors, and Special Relativity)

**Time:** ~5-6 hours

---

## Table of Contents

1. [The Equivalence Principle](#1-the-equivalence-principle)
2. [From Special to General Relativity](#2-from-special-to-general-relativity)
3. [The Stress-Energy Tensor](#3-the-stress-energy-tensor)
4. [Einstein Field Equations](#4-einstein-field-equations)
5. [Geodesic Equation in General Relativity](#5-geodesic-equation-in-general-relativity)
6. [Weak Field Limit and Newtonian Gravity](#6-weak-field-limit-and-newtonian-gravity)
7. [Linearized Gravity and Gravitational Waves](#7-linearized-gravity-and-gravitational-waves)
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

**⚠️ Metric Signature Convention:** Throughout this lesson (and all GR lessons), we use the **(-,+,+,+) signature** for the spacetime metric. This is the "mostly plus" or "West Coast" convention used in most modern GR textbooks (Carroll, Wald, Hartle). The Minkowski metric is η_μν = diag(-1, 1, 1, 1).

---

## 1. The Equivalence Principle

### 📖 Concept

Einstein's **Equivalence Principle** is the cornerstone of General Relativity. It has two formulations:

**Weak Equivalence Principle (WEP):**
All objects fall at the same rate in a gravitational field, regardless of their mass or composition.
- Inertial mass = Gravitational mass
- Galileo's experiment from the Leaning Tower of Pisa
- Apollo 15: hammer and feather on the Moon

**Einstein's Equivalence Principle (EEP):**
Locally (in a small enough region of spacetime), the effects of gravity are indistinguishable from those of acceleration.
- An observer in a freely falling elevator feels no gravity
- An observer in an accelerating rocket feels "artificial gravity"
- No local experiment can distinguish gravity from acceleration

**Physical Insight:**
Imagine you're in a windowless elevator. If you drop a ball and see it accelerate toward the floor, you can't tell whether:
1. The elevator is at rest on Earth (gravity pulling down)
2. The elevator is accelerating upward in deep space (inertia pushing down)

This equivalence leads to the revolutionary idea: **gravity is not a force, but the curvature of spacetime**.

---

### 💻 Code Example: Free Fall vs Acceleration

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Scenario 1: Free fall in Earth's gravity (observer perspective)
# Scenario 2: Accelerating rocket in space

time = np.linspace(0, 2, 100)  # np.linspace() creates evenly spaced array between start and end

# In the falling elevator frame, objects appear weightless
# In the accelerating rocket frame, objects fall with a = g

g = 9.8  # m/s^2

# Position of dropped ball (relative to floor)
# Free fall elevator: ball floats (zero relative acceleration)
free_fall_position = np.zeros_like(time)

# Accelerating rocket: ball falls
rocket_position = -0.5 * g * time**2

print("Equivalence Principle Demonstration")
print("=" * 50)
print("\nScenario 1: Free fall elevator on Earth")
print(f"Ball position at t=1s: {free_fall_position[50]:.2f} m (floats!)")
print("\nScenario 2: Accelerating rocket in space")
print(f"Ball position at t=1s: {rocket_position[50]:.2f} m (falls!)")
print("\nConclusion: Locally, free fall cancels gravity!")
```

**Expected output:**
```
Equivalence Principle Demonstration
==================================================

Scenario 1: Free fall elevator on Earth
Ball position at t=1s: 0.00 m (floats!)

Scenario 2: Accelerating rocket in space
Ball position at t=1s: -4.90 m (falls!)

Conclusion: Locally, free fall cancels gravity!
```

---

### 📊 Visualization: Equivalence Principle

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Scenario 1: Stationary on Earth
ax = axes[0]
ax.add_patch(plt.Rectangle((0.3, 0), 0.4, 2, fill=False, edgecolor='black', linewidth=2))
ax.plot(0.5, 1.5, 'o', color=COLORS['red'], markersize=12, label='Person')
ax.arrow(0.5, 1.3, 0, -0.3, head_width=0.05, head_length=0.1, fc=COLORS['blue'], ec=COLORS['blue'])
ax.text(0.7, 1.1, 'g', fontsize=14, color=COLORS['blue'])
ax.text(0.5, -0.3, 'Ground', ha='center', fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 2.5)
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.axis('off')
ax.set_title('At Rest on Earth\n(feels weight)', fontsize=12, fontweight='bold')

# Scenario 2: Free fall
ax = axes[1]
ax.add_patch(plt.Rectangle((0.3, 0.5), 0.4, 2, fill=False, edgecolor='black', linewidth=2))
ax.plot(0.5, 2.0, 'o', color=COLORS['red'], markersize=12, label='Person')
ax.plot(0.5, 1.2, 'o', color=COLORS['orange'], markersize=8, label='Ball')
ax.arrow(0.1, 1.5, 0, -0.3, head_width=0.05, head_length=0.1, fc=COLORS['blue'], ec=COLORS['blue'])
ax.text(-0.1, 1.1, 'g', fontsize=14, color=COLORS['blue'])
ax.arrow(0.85, 1.5, 0, -0.3, head_width=0.05, head_length=0.1, fc=COLORS['green'], ec=COLORS['green'])
ax.text(1.0, 1.1, 'a=g', fontsize=12, color=COLORS['green'])
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 2.5)
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.axis('off')
ax.set_title('Free Fall (elevator)\n(feels weightless)', fontsize=12, fontweight='bold')

# Scenario 3: Accelerating rocket
ax = axes[2]
rocket_y = 1.0
ax.add_patch(plt.Rectangle((0.3, rocket_y), 0.4, 2, fill=False, edgecolor='black', linewidth=2))
ax.plot(0.5, rocket_y+1.5, 'o', color=COLORS['red'], markersize=12)
ax.arrow(0.5, rocket_y+1.3, 0, -0.3, head_width=0.05, head_length=0.1, fc=COLORS['purple'], ec=COLORS['purple'])
ax.text(0.7, rocket_y+1.1, 'feels g', fontsize=12, color=COLORS['purple'])
ax.arrow(0.5, rocket_y-0.3, 0, -0.4, head_width=0.08, head_length=0.15, fc=COLORS['orange'], ec=COLORS['orange'], linewidth=2)
ax.text(0.7, rocket_y-0.5, 'a=g', fontsize=14, color=COLORS['orange'])
ax.text(0.5, 0, 'Deep Space', ha='center', fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(-0.8, 3.5)
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.axis('off')
ax.set_title('Accelerating Rocket\n(feels weight)', fontsize=12, fontweight='bold')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

**What you should see:** Three scenarios showing how gravity and acceleration are equivalent.

---

### 🎯 Practice Question #1

**Q:** An astronaut in a windowless spacecraft feels a force pressing them against the floor. Can they determine whether they're (a) on Earth's surface or (b) accelerating at g = 9.8 m/s² in deep space?

<details>
<summary>💡 Hint 1</summary>

Think about what the Equivalence Principle says about local measurements.
</details>

<details>
<summary>💡 Hint 2</summary>

The key word is "locally" - within a small region of spacetime.
</details>

<details>
<summary>✅ Answer</summary>

**No**, locally they cannot distinguish between the two scenarios. This is the essence of the Equivalence Principle.

However, if they had a large enough spacecraft and precise enough instruments, they could detect **tidal forces** - the variation in gravitational field strength. On Earth, the gravitational field points toward the center, so two separated masses would accelerate slightly toward each other. In a uniformly accelerating rocket, they would remain parallel.

This is why we say "locally" - true equivalence only holds in regions small enough that tidal effects are negligible.
</details>

---

## 2. From Special to General Relativity

### 📖 Concept

Special Relativity works in **flat spacetime** (Minkowski space) with the metric:

```
ds² = -c²dt² + dx² + dy² + dz²
```

But the Equivalence Principle tells us gravity should curve spacetime. General Relativity generalizes this to **curved spacetime** with an arbitrary metric:

```
ds² = g_μν dx^μ dx^ν
```

where g_μν is the **metric tensor** that encodes the curvature of spacetime.

**Key Changes:**
1. **Flat → Curved:** Minkowski metric η_μν → general metric g_μν
2. **Global → Local:** Inertial frames exist only locally (freely falling frames)
3. **Partial derivatives → Covariant derivatives:** ∂_μ → ∇_μ (accounts for curvature)
4. **Straight lines → Geodesics:** Free particles follow curved paths in curved spacetime

**The Big Idea:**
- Mass and energy tell spacetime how to curve (Einstein equations)
- Curved spacetime tells matter how to move (geodesic equation)

---

### 💻 Code Example: Comparing Metrics

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations

# Special Relativity: Minkowski metric in Cartesian coordinates
eta = np.array([  # np.array() converts Python list/tuple to efficient numpy array
    [-1,  0,  0,  0],  # -c² (we set c=1)
    [ 0,  1,  0,  0],  # +dx²
    [ 0,  0,  1,  0],  # +dy²
    [ 0,  0,  0,  1]   # +dz²
])

print("Minkowski Metric (Special Relativity):")
print(eta)
print(f"Signature: ({np.sum(eta < 0)}, {np.sum(eta > 0)})")
print(f"Determinant: {np.linalg.det(eta):.0f}")  # np.linalg.det() computes matrix determinant

# General Relativity: Example curved spacetime metric
# (This is a simplified 2D curved space)
# ds² = -dt² + (1 + h(r))dr² where h(r) represents curvature

def curved_metric_2d(r):
    """Simple example of curved 2D spacetime."""
    h = 0.1 * r**2  # Curvature grows with radius
    return np.array([  # np.array() converts Python list/tuple to efficient numpy array
        [-1, 0],
        [0, 1 + h]
    ])

r_values = [0, 1, 2]
print("\n" + "="*50)
print("Curved Spacetime Metric (General Relativity):")
for r in r_values:
    g = curved_metric_2d(r)
    print(f"\nAt r = {r}:")
    print(g)
    print(f"Spatial component g_rr = {g[1,1]:.2f} (curvature = {g[1,1] - 1:.2f})")
```

---

### 🔬 Explore: Proper Time in Curved Spacetime

The **proper time** along a worldline is:

```
τ = ∫ √(-g_μν dx^μ dx^ν)
```

For a stationary observer (dx = dy = dz = 0), this gives:

```
dτ = √(-g_00) dt
```

If g_00 ≠ -1, time flows at different rates!

```python
# Example: Time dilation near a massive object
# (We'll see the full Schwarzschild metric in Lesson 11)

def gravitational_time_dilation(r, M, c=1, G=1):
    """
    Simplified time dilation factor near a spherical mass.
    Returns dτ/dt for a stationary observer at radius r.
    """
    r_s = 2 * G * M / c**2  # Schwarzschild radius
    if r <= r_s:
        return 0  # At or inside event horizon
    return np.sqrt(1 - r_s / r)  # np.sqrt() computes square root

# Example: Earth-like object
M_earth = 1.0  # Normalized units
r_earth = 100.0  # Surface radius in these units

radii = np.linspace(r_earth, 3*r_earth, 100)  # np.linspace() creates evenly spaced array between start and end
time_factor = [gravitational_time_dilation(r, M_earth) for r in radii]

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(radii, time_factor, color=COLORS['blue'], linewidth=2)  # plt.plot() draws line plot
plt.axhline(y=1, color=COLORS['gray'], linestyle='--', label='Flat spacetime (SR)')  # plt.axhline() draws horizontal line across plot
plt.xlabel('Radius r (surface = 100)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Time dilation factor dτ/dt', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Gravitational Time Dilation in Curved Spacetime', fontsize=14, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels
plt.ylim([0.95, 1.01])  # plt.ylim() sets y-axis limits
plt.show()  # plt.show() displays the figure window

print(f"At surface (r={r_earth}): dτ/dt = {gravitational_time_dilation(r_earth, M_earth):.6f}")
print(f"At r=2×surface: dτ/dt = {gravitational_time_dilation(2*r_earth, M_earth):.6f}")
print("Time runs slower closer to the mass!")
```

---

## 3. The Stress-Energy Tensor

### 📖 Concept

The **stress-energy tensor** T^μν describes the distribution of energy, momentum, and stress in spacetime. It's the source term in Einstein's equations.

**Components (in a local frame):**

```
T^μν = [
    [ρ,      p_x,     p_y,     p_z    ]  # Energy density and momentum density
    [p_x,    σ_xx,    σ_xy,    σ_xz   ]  # Momentum flux and stress
    [p_y,    σ_xy,    σ_yy,    σ_yz   ]
    [p_z,    σ_xz,    σ_yz,    σ_zz   ]
]
```

Where:
- **T^00 = ρ**: Energy density (includes rest mass energy ρ = ρ_0 c²)
- **T^0i = p_i**: Momentum density (energy flux / c²)
- **T^ij**: Stress (momentum flux)

**Physical Interpretation:**
- T^μν = flux of μ-momentum across a surface of constant x^ν
- T^μν is **symmetric**: T^μν = T^νμ
- **Conserved**: ∇_μ T^μν = 0 (energy-momentum conservation)

---

### 💻 Code Example: Stress-Energy Tensors

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations

# Example 1: Perfect fluid (e.g., ideal gas, radiation)
# T^μν = (ρ + p)u^μ u^ν + p η^μν

def perfect_fluid_stress_energy(rho, p, u):
    """
    Compute stress-energy tensor for a perfect fluid.

    Parameters:
    rho: energy density
    p: pressure
    u: 4-velocity (normalized: u_μ u^μ = -1)
    """
    eta = np.diag([-1, 1, 1, 1])  # Minkowski metric
    T = (rho + p) * np.outer(u, u) + p * eta
    return T

# Rest frame of fluid: u = (1, 0, 0, 0)
u_rest = np.array([1, 0, 0, 0])  # np.array() converts Python list/tuple to efficient numpy array
rho = 1.0  # Energy density
p = 0.3    # Pressure

T_rest = perfect_fluid_stress_energy(rho, p, u_rest)

print("Perfect Fluid Stress-Energy Tensor (Rest Frame):")
print(T_rest)
print(f"\nT^00 (energy density) = {T_rest[0,0]:.2f}")
print(f"T^11 = T^22 = T^33 (pressure) = {T_rest[1,1]:.2f}")
print(f"Off-diagonal terms = {T_rest[0,1]:.2f} (zero in rest frame)")

# Example 2: Dust (pressureless matter)
p_dust = 0
T_dust = perfect_fluid_stress_energy(rho, p_dust, u_rest)

print("\n" + "="*50)
print("Dust Stress-Energy Tensor (p=0):")
print(T_dust)

# Example 3: Radiation (p = ρ/3)
p_rad = rho / 3
T_rad = perfect_fluid_stress_energy(rho, p_rad, u_rest)

print("\n" + "="*50)
print("Radiation Stress-Energy Tensor (p=ρ/3):")
print(T_rad)
```

---

### 📊 Visualization: Energy-Momentum Distribution

```python
# Visualize the structure of stress-energy tensor

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

tensors = [
    (T_dust, "Dust (p=0)"),
    (T_rest, "Fluid (p=0.3ρ)"),
    (T_rad, "Radiation (p=ρ/3)")
]

for ax, (T, label) in zip(axes, tensors):
    im = ax.imshow(T, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels(['t', 'x', 'y', 'z'])
    ax.set_yticklabels(['t', 'x', 'y', 'z'])
    ax.set_title(label, fontsize=12, fontweight='bold')

    # Add values as text
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{T[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im, ax=axes, label='Component value')  # plt.colorbar() adds color scale bar to plot
plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🎯 Practice Question #2

**Q:** For a perfect fluid at rest with energy density ρ = 2 and pressure p = 0.5, what is T^00? What is T^11?

<details>
<summary>💡 Hint</summary>

In the rest frame, u = (1, 0, 0, 0). Use the perfect fluid formula.
</details>

<details>
<summary>✅ Answer</summary>

For a perfect fluid: T^μν = (ρ + p)u^μ u^ν + p η^μν

In the rest frame:
- **T^00** = (ρ + p)·1·1 + p·(-1) = (2 + 0.5)·1 + 0.5·(-1) = 2.5 - 0.5 = **2.0**
- **T^11** = (ρ + p)·0·0 + p·(+1) = 0 + 0.5 = **0.5**

Verify in Python:
```python
u = np.array([1, 0, 0, 0])  # np.array() converts Python list/tuple to efficient numpy array
T = perfect_fluid_stress_energy(2.0, 0.5, u)
print(f"T^00 = {T[0,0]}")  # 2.0
print(f"T^11 = {T[1,1]}")  # 0.5
```
</details>

---

## 4. Einstein Field Equations

### 📖 Concept

The **Einstein Field Equations** are the fundamental equations of General Relativity:

```
G_μν = (8πG/c⁴) T_μν
```

Or equivalently:

```
R_μν - (1/2)g_μν R = (8πG/c⁴) T_μν
```

Where:
- **G_μν**: Einstein tensor (describes spacetime curvature)
- **R_μν**: Ricci curvature tensor
- **R = g^μν R_μν**: Ricci scalar (trace of Ricci tensor)
- **T_μν**: Stress-energy tensor (matter/energy content)
- **g_μν**: Metric tensor
- **G**: Newton's gravitational constant
- **c**: Speed of light

**Physical Meaning:**
```
[Curvature of spacetime] = [Constant] × [Energy and momentum]
```

These are **10 coupled nonlinear partial differential equations** (the metric is symmetric, so only 10 independent components).

**In vacuum** (T_μν = 0), they simplify to:

```
R_μν = 0
```

This is still highly nontrivial! Many famous solutions (Schwarzschild, Kerr) solve the vacuum equations.

---

### 💻 Code Example: Einstein Tensor Components

```python
from sympy import  # Import symbolic math functions
from sympy.tensor.tensor import TensorIndexType, TensorIndex, tensor_indices

# For simple pedagogical example, we'll compute Einstein tensor
# for a 2D spherically symmetric spacetime

# Define symbolic variables
t, r = symbols('t r', real=True, positive=True)  # symbols() creates symbolic variables
M = symbols('M', real=True, positive=True)  # Mass parameter

# Schwarzschild-like metric in 2D: ds² = -(1-2M/r)dt² + dr²/(1-2M/r)
# (Simplified version; full solution in Lesson 11)

def schwarzschild_2d():
    """Simplified 2D Schwarzschild metric for demonstration."""
    f = 1 - 2*M/r
    g = Matrix([  # Matrix() creates symbolic matrix
        [-f, 0],
        [0, 1/f]
    ])
    return g

g_metric = schwarzschild_2d()

print("Simplified 2D Metric (Schwarzschild-like):")
print(g_metric)

# Computing curvature is complex; use SymPy or numerical libraries
# For now, let's show the structure

print("\n" + "="*50)
print("Einstein Field Equations Structure:")
print("Left side: G_μν (Einstein tensor from metric)")
print("Right side: (8πG/c⁴) T_μν (matter content)")
print("\nFor vacuum solutions: G_μν = 0, equivalently R_μν = 0")
```

---

### 📊 Visualization: Spacetime Curvature

```python
# Visualize the "rubber sheet" analogy for spacetime curvature

from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit

# Create a grid in 2D space
x = np.linspace(-5, 5, 50)  # np.linspace() creates evenly spaced array between start and end
y = np.linspace(-5, 5, 50)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x, y)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# Distance from center
R = np.sqrt(X**2 + Y**2)  # np.sqrt() computes square root

# "Curvature" induced by a mass at origin
# Z represents the distortion of the spatial slice
M_demo = 2.0
Z = -M_demo / (R + 0.5)  # Add 0.5 to avoid singularity

fig = plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Left: 3D surface
ax1 = fig.add_subplot(121, projection='3d')  projection='3d'  # Create 3D axes
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)  # .plot_surface() draws 3D surface plot
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_zlabel('Curvature', fontsize=12)
ax1.set_title('Spacetime Curvature (Rubber Sheet Analogy)', fontsize=12, fontweight='bold')
ax1.view_init(elev=20, azim=45)

# Right: Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(0, 0, 'o', color=COLORS['red'], markersize=15, label='Mass M')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Curvature Contours (Top View)', fontsize=12, fontweight='bold')
ax2.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("The 'dip' represents how mass curves spacetime.")
print("Objects follow geodesics (straight lines in curved space).")
print("This is why orbits appear curved!")
```

---

### 🔬 Explore: Cosmological Constant

Einstein later added a **cosmological constant** Λ to his equations:

```
R_μν - (1/2)g_μν R + Λg_μν = (8πG/c⁴) T_μν
```

This term represents the energy density of empty space itself ("dark energy").

```python
# The cosmological constant Λ affects the expansion of the universe

def friedmann_equation(a, a_dot, rho, Lambda, k=0):
    """
    Simplified Friedmann equation from Einstein equations.
    a: scale factor
    a_dot: time derivative of scale factor
    rho: matter density
    Lambda: cosmological constant
    k: spatial curvature (+1, 0, -1)
    """
    H = a_dot / a  # Hubble parameter
    H_squared_predicted = (8 * np.pi * rho / 3) + (Lambda / 3) - (k / a**2)
    return H**2, H_squared_predicted

# Example values (normalized units)
a = 1.0
a_dot = 0.1
rho_matter = 0.3
Lambda_dark = 0.7

H2_actual, H2_predicted = friedmann_equation(a, a_dot, rho_matter, Lambda_dark)

print(f"Hubble parameter H² = {H2_actual:.3f}")
print(f"From Friedmann eq: H² = {H2_predicted:.3f}")
print(f"\nMatter contribution: {8*np.pi*rho_matter/3:.3f}")
print(f"Dark energy (Λ) contribution: {Lambda_dark/3:.3f}")
print("\nDark energy dominates the expansion today!")
```

---

## 5. Geodesic Equation in General Relativity

### 📖 Concept

In flat spacetime, free particles move in straight lines. In curved spacetime, they follow **geodesics** - the "straightest possible" paths.

The **geodesic equation** is:

```
d²x^μ/dτ² + Γ^μ_νλ (dx^ν/dτ)(dx^λ/dτ) = 0
```

Where:
- x^μ(τ): the particle's worldline
- τ: proper time
- Γ^μ_νλ: Christoffel symbols (encode the curvature)

**Physical meaning:** This is the equation of motion for a free particle in curved spacetime - the GR version of Newton's F = ma with F = 0!

**Christoffel Symbols:**
```
Γ^μ_νλ = (1/2)g^μσ (∂_ν g_σλ + ∂_λ g_σν - ∂_σ g_νλ)
```

These tell us how vectors change as we move through curved space.

---

### 💻 Code Example: Computing Christoffel Symbols

```python
# Compute Christoffel symbols for a simple metric
# Example: 2D polar coordinates in flat space
# ds² = -dt² + dr² + r²dθ²

from sympy import  # Import symbolic math functions

t, r, theta = symbols('t r theta', real=True, positive=True)  # symbols() creates symbolic variables

# Metric in polar coordinates (still flat space, but non-Cartesian)
g_polar = Matrix([  # Matrix() creates symbolic matrix
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, r**2]
])

print("Metric in polar coordinates:")
print(g_polar)

# Inverse metric
g_inv = g_polar.inv()

print("\nInverse metric:")
print(g_inv)

# Compute Christoffel symbols
def christoffel_symbols(g, g_inv, coords):  # symbols() creates symbolic variables
    """Compute Christoffel symbols Γ^μ_νλ."""
    n = len(coords)
    Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]

    for mu in range(n):
        for nu in range(n):
            for lam in range(n):
                sum_term = 0
                for sigma in range(n):
                    sum_term += g_inv[mu, sigma] * (
                        diff(g[sigma, lam], coords[nu]) +  # diff() computes symbolic derivative
                        diff(g[sigma, nu], coords[lam]) -  # diff() computes symbolic derivative
                        diff(g[nu, lam], coords[sigma])  # diff() computes symbolic derivative
                    )
                Gamma[mu][nu][lam] = simplify(sum_term / 2)  # simplify() algebraically simplifies expression

    return Gamma

coords = [t, r, theta]
Gamma = christoffel_symbols(g_polar, g_inv, coords)  # symbols() creates symbolic variables

# Print non-zero components
print("\nNon-zero Christoffel symbols:")
for mu in range(3):
    for nu in range(3):
        for lam in range(3):
            if Gamma[mu][nu][lam] != 0:
                coord_names = ['t', 'r', 'θ']
                print(f"Γ^{coord_names[mu]}_{coord_names[nu]}{coord_names[lam]} = {Gamma[mu][nu][lam]}")
```

**Expected output:**
```
Non-zero Christoffel symbols:
Γ^r_θθ = -r
Γ^θ_rθ = 1/r
Γ^θ_θr = 1/r
```

These encode the "centrifugal effect" in polar coordinates!

---

### 📊 Visualization: Geodesics on a Sphere

```python
# Visualize geodesics (great circles) on a 2-sphere

from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit

fig = plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Create sphere
u = np.linspace(0, 2 * np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
v = np.linspace(0, np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
x_sphere = np.outer(np.cos(u), np.sin(v))  # np.sin() computes sine (element-wise for arrays)
y_sphere = np.outer(np.sin(u), np.sin(v))  # np.sin() computes sine (element-wise for arrays)
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))  # np.cos() computes cosine (element-wise for arrays)

# Left: Geodesic (great circle)
ax1 = fig.add_subplot(121, projection='3d')  projection='3d'  # Create 3D axes
ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='lightblue')  # .plot_surface() draws 3D surface plot

# Great circle: theta = pi/2 (equator)
theta_geo = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
x_geo = np.cos(theta_geo)  # np.cos() computes cosine (element-wise for arrays)
y_geo = np.sin(theta_geo)  # np.sin() computes sine (element-wise for arrays)
z_geo = np.zeros_like(theta_geo)
ax1.plot(x_geo, y_geo, z_geo, color=COLORS['red'], linewidth=3, label='Geodesic')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Geodesic on Sphere (Equator)', fontweight='bold')
ax1.legend()

# Right: Non-geodesic path
ax2 = fig.add_subplot(122, projection='3d')  projection='3d'  # Create 3D axes
ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='lightblue')  # .plot_surface() draws 3D surface plot

# Small circle at latitude 30°
lat = np.pi/6
theta_circle = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
x_circle = np.cos(lat) * np.cos(theta_circle)  # np.cos() computes cosine (element-wise for arrays)
y_circle = np.cos(lat) * np.sin(theta_circle)  # np.sin() computes sine (element-wise for arrays)
z_circle = np.sin(lat) * np.ones_like(theta_circle)  # np.sin() computes sine (element-wise for arrays)
ax2.plot(x_circle, y_circle, z_circle, color=COLORS['blue'], linewidth=3, label='Not a geodesic')

# Compare with geodesic
ax2.plot(x_geo, y_geo, z_geo, color=COLORS['red'], linewidth=2, alpha=0.5, linestyle='--', label='Geodesic')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Non-Geodesic Path (Small Circle)', fontweight='bold')
ax2.legend()

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Geodesics are the 'straightest' paths on curved surfaces.")
print("On a sphere, geodesics are great circles (like airplane routes!).")
```

---

### 🎯 Practice Question #3

**Q:** In the geodesic equation, what happens if all Christoffel symbols Γ^μ_νλ = 0?

<details>
<summary>💡 Hint</summary>

Look at the geodesic equation: d²x^μ/dτ² + Γ^μ_νλ (dx^ν/dτ)(dx^λ/dτ) = 0
</details>

<details>
<summary>✅ Answer</summary>

If all Γ^μ_νλ = 0, the geodesic equation becomes:

```
d²x^μ/dτ² = 0
```

This means **x^μ(τ) is a straight line** (constant velocity in each coordinate).

This happens in **flat spacetime with Cartesian coordinates** - i.e., Special Relativity! The Christoffel symbols encode the curvature and/or non-Cartesian coordinates.
</details>

---

## 6. Weak Field Limit and Newtonian Gravity

### 📖 Concept

How do we recover **Newtonian gravity** from General Relativity? We need the **weak field limit**:

**Assumptions:**
1. Gravitational field is weak: g_μν ≈ η_μν + h_μν where |h_μν| << 1
2. Velocities are small: v << c
3. Static field: ∂g_μν/∂t ≈ 0

The metric becomes:

```
g_00 = -(1 + 2Φ/c²)
g_ij = δ_ij
g_0i = 0
```

where Φ is the **Newtonian gravitational potential**.

**The geodesic equation** in this limit reduces to Newton's law:

```
d²x/dt² = -∇Φ
```

And the **Einstein equations** reduce to Poisson's equation:

```
∇²Φ = 4πGρ
```

**This is beautiful!** Newton's theory emerges as the weak-field, slow-motion limit of Einstein's equations.

---

### 💻 Code Example: Weak Field Metric

```python
# Compare weak field GR metric with Newtonian potential

def newtonian_potential(r, M, G=1):
    """Newton's gravitational potential Φ = -GM/r."""
    return -G * M / r

def weak_field_metric_00(r, M, G=1, c=1):
    """
    Weak field approximation: g_00 = -(1 + 2Φ/c²)
    For Φ = -GM/r, this gives g_00 = -(1 - 2GM/(rc²))
    """
    Phi = newtonian_potential(r, M, G)
    return -(1 + 2 * Phi / c**2)

# Example: Earth-like mass
M = 1.0
r_values = np.linspace(10, 100, 100)  # Avoid singularity at r=0

Phi = [newtonian_potential(r, M) for r in r_values]
g_00 = [weak_field_metric_00(r, M) for r in r_values]

plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Left: Newtonian potential
plt.subplot(1, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(r_values, Phi, color=COLORS['blue'], linewidth=2)  # plt.plot() draws line plot
plt.xlabel('Radius r', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Potential Φ', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Newtonian Gravitational Potential', fontsize=12, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

# Right: Metric component
plt.subplot(1, 2, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(r_values, g_00, color=COLORS['red'], linewidth=2, label='g₀₀')  # plt.plot() draws line plot
plt.axhline(y=-1, color=COLORS['gray'], linestyle='--', label='Flat space (g₀₀=-1)')  # plt.axhline() draws horizontal line across plot
plt.xlabel('Radius r', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Metric component g₀₀', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Weak Field Metric Component', fontsize=12, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

# Compute deviation from flat space
r_test = 20
Phi_test = newtonian_potential(r_test, M)
g00_test = weak_field_metric_00(r_test, M)
deviation = abs(g00_test + 1)  # Difference from -1

print(f"At r = {r_test}:")
print(f"  Newtonian potential: Φ = {Phi_test:.4f}")
print(f"  Metric component: g₀₀ = {g00_test:.6f}")
print(f"  Deviation from flat space: |g₀₀ + 1| = {deviation:.6f} << 1 ✓")
print("\nWeak field approximation is valid!")
```

---

### 🔬 Explore: Newtonian Limit of Geodesic Equation

```python
# Show that geodesic equation → Newton's F = ma in weak field

print("Geodesic Equation in Weak Field Limit")
print("=" * 60)
print()
print("Full GR geodesic equation:")
print("  d²x^μ/dτ² + Γ^μ_νλ (dx^ν/dτ)(dx^λ/dτ) = 0")
print()
print("For weak field: g_00 = -(1 + 2Φ/c²), slow motion (v << c):")
print()
print("  The time component gives: dt/dτ ≈ 1")
print("  The spatial Christoffel symbols give: Γ^i_00 ≈ ∂_i Φ/c²")
print()
print("  Geodesic equation becomes:")
print("  d²x^i/dt² ≈ -∂_i Φ")
print()
print("This is exactly Newton's law: a = -∇Φ")
print()
print("For a point mass M:")
print("  Φ = -GM/r")
print("  a = -∇Φ = -GM r̂/r² ✓")
print()
print("General Relativity → Newtonian Gravity in appropriate limit!")
```

---

## 7. Linearized Gravity and Gravitational Waves

### 📖 Concept

For **small perturbations** around flat spacetime:

```
g_μν = η_μν + h_μν    where |h_μν| << 1
```

Linearizing the Einstein equations in vacuum (T_μν = 0) and choosing an appropriate gauge, we get the **wave equation**:

```
□ h̄_μν = 0
```

where □ = -∂²/∂t² + ∇² is the d'Alembertian (wave operator) and h̄_μν is the trace-reversed perturbation.

**This predicts gravitational waves!** Solutions are:

```
h_μν = A_μν exp(i k_λ x^λ)
```

Plane waves traveling at the speed of light!

**Polarizations:**
- Two independent polarizations (+ and ×)
- Transverse: perpendicular to propagation direction
- Traceless: h = g^μν h_μν = 0 (in appropriate gauge)

**Physical Effect:**
A gravitational wave stretches and squeezes space periodically - this is how LIGO detects them!

---

### 💻 Code Example: Gravitational Wave Solution

```python
# Visualize a gravitational wave propagating

# Gravitational wave traveling in z-direction
# h_+ polarization: stretches x, squeezes y (and vice versa)

def gw_strain_plus(t, z, A, omega, c=1):
    """
    Plus polarization gravitational wave.
    Returns h_xx, h_yy at position z and time t.
    """
    phase = omega * (t - z/c)
    h_xx = A * np.cos(phase)  # np.cos() computes cosine (element-wise for arrays)
    h_yy = -A * np.cos(phase)  # np.cos() computes cosine (element-wise for arrays)
    return h_xx, h_yy

# Parameters
A = 0.1  # Amplitude (strain)
omega = 2 * np.pi  # Angular frequency
t_values = np.linspace(0, 2, 100)  # Two periods

# Fixed position z = 0
z = 0
h_xx_vals = []
h_yy_vals = []

for t in t_values:
    h_xx, h_yy = gw_strain_plus(t, z, A, omega)
    h_xx_vals.append(h_xx)
    h_yy_vals.append(h_yy)

plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Left: Strain vs time
plt.subplot(1, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(t_values, h_xx_vals, color=COLORS['blue'], linewidth=2, label='h_xx (stretch x)')  # plt.plot() draws line plot
plt.plot(t_values, h_yy_vals, color=COLORS['orange'], linewidth=2, label='h_yy (squeeze y)')  # plt.plot() draws line plot
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.xlabel('Time t', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Strain h', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Gravitational Wave: + Polarization', fontsize=12, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels

# Right: Effect on ring of particles
ax = plt.subplot(1, 2, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Initial circle of test masses
theta = np.linspace(0, 2*np.pi, 16, endpoint=False)  # np.linspace() creates evenly spaced array between start and end
x0 = np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
y0 = np.sin(theta)  # np.sin() computes sine (element-wise for arrays)

# At time t = 0.25 (quarter period)
t_snapshot = 0.25
h_xx_snap, h_yy_snap = gw_strain_plus(t_snapshot, z, A, omega)

# Deformed positions (linearized effect)
x_deformed = x0 * (1 + h_xx_snap)
y_deformed = y0 * (1 + h_yy_snap)

plt.plot(x0, y0, 'o--', color=COLORS['gray'], alpha=0.5, label='Original', markersize=8)  # plt.plot() draws line plot
plt.plot(x_deformed, y_deformed, 'o-', color=COLORS['red'], label='Deformed', markersize=8)  # plt.plot() draws line plot
plt.xlabel('x', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('y', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title(f'Effect on Test Masses (t={t_snapshot})', fontsize=12, fontweight='bold')  # plt.title() sets plot title
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.xlim(-1.5, 1.5)  # plt.xlim() sets x-axis limits
plt.ylim(-1.5, 1.5)  # plt.ylim() sets y-axis limits

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print(f"Gravitational wave amplitude: h ~ {A}")
print(f"Frequency: f = ω/(2π) = {omega/(2*np.pi):.2f} Hz")
print("\nThe wave alternately stretches and squeezes space!")
print("This is how LIGO detects gravitational waves.")
```

---

### 📊 Visualization: Animated Gravitational Wave

```python
# Create animation of gravitational wave passing through test masses

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.grid(True, alpha=0.3)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Gravitational Wave Passing Through Test Masses', fontsize=12, fontweight='bold')

# Initial ring of particles
n_particles = 20
theta = np.linspace(0, 2*np.pi, n_particles, endpoint=False)  # np.linspace() creates evenly spaced array between start and end
x0 = np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
y0 = np.sin(theta)  # np.sin() computes sine (element-wise for arrays)

# Plot elements
original_ring, = ax.plot(x0, y0, 'o', color=COLORS['gray'], alpha=0.3, markersize=6)
deformed_ring, = ax.plot([], [], 'o', color=COLORS['blue'], markersize=10)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

def animate(frame):
    t = frame / 30.0  # 30 frames per time unit
    h_xx, h_yy = gw_strain_plus(t, 0, 0.15, 2*np.pi)

    x_def = x0 * (1 + h_xx)
    y_def = y0 * (1 + h_yy)

    deformed_ring.set_data(x_def, y_def)
    time_text.set_text(f'Time: {t:.2f}')

    return deformed_ring, time_text

# Create animation (60 frames = 2 periods)
anim = FuncAnimation(fig, animate, frames=60, interval=50, blit=True)

# To display: anim  (in Jupyter)
# To save: anim.save('gw_animation.gif', writer='pillow')

plt.close()  # Don't show static plot

print("Animation created! (Run in Jupyter to see it)")
print("The ring of particles oscillates as the gravitational wave passes.")
```

---

### 🎯 Practice Question #4

**Q:** A gravitational wave has strain amplitude h ~ 10^(-21). If a LIGO arm is L = 4 km long, what is the change in length ΔL?

<details>
<summary>💡 Hint</summary>

The strain h is defined as h = ΔL/L.
</details>

<details>
<summary>✅ Answer</summary>

From h = ΔL/L:

ΔL = h × L = 10^(-21) × 4000 m = 4 × 10^(-18) m

**ΔL ≈ 4 × 10^(-18) meters**

This is about 1/1000th the diameter of a proton! This shows the incredible sensitivity of LIGO.

```python
h = 1e-21  # Strain
L = 4000   # meters
delta_L = h * L

print(f"Strain: h = {h}")
print(f"Arm length: L = {L} m")
print(f"Change in length: ΔL = {delta_L:.2e} m")
print(f"\nFor comparison:")
print(f"  Proton diameter ~ 1.7e-15 m")
print(f"  ΔL/proton ~ {delta_L/1.7e-15:.4f}")
```
</details>

---

## Practice Questions

Test your understanding of GR foundations:

### Equivalence Principle

1. An astronaut in a falling elevator drops a ball. What does the astronaut observe? Why?

2. Can tidal forces distinguish between gravity and acceleration? Explain.

3. Why does the equivalence principle only hold "locally"?

### Stress-Energy Tensor

4. For a perfect fluid with ρ = 3.0 and p = 1.0 in its rest frame, compute T^00 and T^11.

5. What is the difference between the stress-energy tensor for dust (p=0) and radiation (p=ρ/3)?

6. Why must the stress-energy tensor be symmetric?

### Einstein Equations and Geodesics

7. In vacuum, the Einstein equations reduce to R_μν = 0. Does this mean spacetime is flat?

8. What do the Christoffel symbols represent physically?

9. Write out the geodesic equation for a free particle. How is this related to Newton's first law?

### Weak Field Limit

10. In the weak field limit, show that g_00 ≈ -(1 + 2Φ/c²) where Φ is the Newtonian potential.

11. How do the Einstein equations reduce to Poisson's equation ∇²Φ = 4πGρ?

12. At Earth's surface, estimate the deviation of g_00 from -1.

### Gravitational Waves

13. How many independent polarizations do gravitational waves have? Name them.

14. Why don't gravitational waves have a longitudinal polarization (unlike sound waves)?

15. If a gravitational wave has frequency f = 100 Hz, what is its wavelength?

---

### 📝 Check Your Answers

Run the quiz script:
```bash
cd /Users/clarkcarter/Claude/personal/gr/lessons/10_gr_foundations
python quiz.py
```

---

## Next Steps

Congratulations! You've learned the foundations of General Relativity. You now understand:

- The equivalence principle and its consequences
- How energy and matter curve spacetime (Einstein equations)
- How particles move in curved spacetime (geodesic equation)
- The connection to Newtonian gravity (weak field limit)
- Gravitational waves as ripples in spacetime

**What's next:**
- Lesson 11: We'll solve the Einstein equations to find black holes, cosmological models, and more!
- Lesson 12: We'll explore the fascinating phenomena predicted by GR

**Additional Resources:**
- Sean Carroll's GR Lecture Notes (arXiv:gr-qc/9712019)
- MIT OCW 8.962: General Relativity (Scott Hughes)
- Leonard Susskind's "General Relativity" lectures
- Misner, Thorne, Wheeler: "Gravitation" (the classic textbook)

---

**Ready to solve the equations?** → [Lesson 11: GR Solutions](../11_gr_solutions/LESSON.md)
