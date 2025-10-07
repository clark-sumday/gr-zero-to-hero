# Lesson 2: Multivariable Calculus

**Topics:** Partial derivatives, gradients, directional derivatives, div/grad/curl, chain rule, optimization
**Prerequisites:** Lesson 1 (Linear Algebra), single-variable calculus
**Time:** 4-6 hours

## Table of Contents
1. [Partial Derivatives](#1-partial-derivatives)
2. [The Gradient Vector](#2-the-gradient-vector)
3. [Directional Derivatives](#3-directional-derivatives)
4. [Chain Rule in Multiple Variables](#4-chain-rule-in-multiple-variables)
5. [Divergence and Curl](#5-divergence-and-curl)
6. [Optimization and Critical Points](#6-optimization-and-critical-points)
7. [Lagrange Multipliers](#7-lagrange-multipliers)

---

## 1. Partial Derivatives

### 📖 Concept

For a function f(x, y), the **partial derivative** ∂f/∂x measures how f changes when x changes while y is held constant.

**Notation:**
- ∂f/∂x or f_x or ∂₁f (partial derivative with respect to x)
- ∂f/∂y or f_y or ∂₂f (partial derivative with respect to y)

**Geometric interpretation:** Partial derivatives give the slope of the surface z = f(x,y) in the direction of the coordinate axes.

**Connection to GR:** In spacetime, we'll take partial derivatives with respect to time and space coordinates (∂/∂t, ∂/∂x, etc.)

### 💻 Code Example

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Define function f(x,y) = x² + y²
def f(x, y):
    return x**2 + y**2

# Compute partial derivatives numerically
def partial_x(f, x, y, h=1e-5):
    """Compute ∂f/∂x using finite difference"""
    return (f(x + h, y) - f(x, y)) / h

def partial_y(f, x, y, h=1e-5):
    """Compute ∂f/∂y using finite difference"""
    return (f(x, y + h) - f(x, y)) / h

# Evaluate at a point
x0, y0 = 1, 2
fx = partial_x(f, x0, y0)
fy = partial_y(f, x0, y0)

print(f"f(x,y) = x² + y²")
print(f"At point ({x0}, {y0}):")
print(f"  ∂f/∂x = 2x = {fx:.4f}")
print(f"  ∂f/∂y = 2y = {fy:.4f}")

# Analytical derivatives for comparison
fx_exact = 2*x0
fy_exact = 2*y0
print(f"\nExact values: ∂f/∂x = {fx_exact}, ∂f/∂y = {fy_exact}")
```

### 📊 Visualization

```python
# Create surface plot with tangent planes
x = np.linspace(-2, 2, 50)  # np.linspace() creates evenly spaced array between start and end
y = np.linspace(-2, 2, 50)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x, y)  # np.meshgrid() creates coordinate matrices from coordinate vectors
Z = f(X, Y)

fig = plt.figure(figsize=(14, 5))  # plt.figure() creates a new figure for plotting

# 3D surface
ax1 = fig.add_subplot(131, projection='3d')  projection='3d'  # Create 3D axes
ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')  # .plot_surface() draws 3D surface plot
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Surface z = x² + y²')

# Slice in x-direction (y = y0)
ax2 = fig.add_subplot(132)
x_slice = np.linspace(-2, 2, 100)  # np.linspace() creates evenly spaced array between start and end
z_slice = f(x_slice, y0)
ax2.plot(x_slice, z_slice, color=COLORS['blue'], linewidth=2)
ax2.plot(x0, f(x0, y0), 'o', color=COLORS['red'], markersize=10)
# Tangent line with slope = ∂f/∂x
x_tangent = np.array([x0 - 0.5, x0 + 0.5])  # np.array() converts Python list/tuple to efficient numpy array
z_tangent = f(x0, y0) + fx_exact * (x_tangent - x0)
ax2.plot(x_tangent, z_tangent, '--', color=COLORS['red'], linewidth=2, label=f'Slope = ∂f/∂x = {fx_exact}')
ax2.set_xlabel('x')
ax2.set_ylabel(f'z (y={y0} fixed)')
ax2.set_title('Partial derivative ∂f/∂x')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Slice in y-direction (x = x0)
ax3 = fig.add_subplot(133)
y_slice = np.linspace(-2, 2, 100)  # np.linspace() creates evenly spaced array between start and end
z_slice = f(x0, y_slice)
ax3.plot(y_slice, z_slice, color=COLORS['blue'], linewidth=2)
ax3.plot(y0, f(x0, y0), 'o', color=COLORS['red'], markersize=10)
# Tangent line with slope = ∂f/∂y
y_tangent = np.array([y0 - 0.5, y0 + 0.5])  # np.array() converts Python list/tuple to efficient numpy array
z_tangent = f(x0, y0) + fy_exact * (y_tangent - y0)
ax3.plot(y_tangent, z_tangent, '--', color=COLORS['red'], linewidth=2, label=f'Slope = ∂f/∂y = {fy_exact}')
ax3.set_xlabel('y')
ax3.set_ylabel(f'z (x={x0} fixed)')
ax3.set_title('Partial derivative ∂f/∂y')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

### 🔬 Explore

Try these experiments:
- Change the function to f(x,y) = x³ + xy + y²
- Compute partial derivatives at different points
- Verify: the order of partial derivatives doesn't matter (∂²f/∂x∂y = ∂²f/∂y∂x)

### 🎯 Practice Question

**Q:** For f(x,y) = e^(xy), compute ∂f/∂x and ∂f/∂y at the point (0, 1).

<details><summary>Hint 1</summary>
Treat y as a constant when computing ∂f/∂x. Use the chain rule: d/dx[e^u] = e^u · du/dx
</details>

<details><summary>Hint 2</summary>
∂f/∂x = e^(xy) · y (by chain rule)
</details>

<details><summary>Answer</summary>
At (0, 1):
- ∂f/∂x = y·e^(xy) = 1·e^0 = 1
- ∂f/∂y = x·e^(xy) = 0·e^0 = 0

The function increases in the x-direction but is momentarily flat in the y-direction at this point.
</details>

---

## 2. The Gradient Vector

### 📖 Concept

The **gradient** ∇f (pronounced "del f" or "nabla f") is a vector containing all partial derivatives:

**2D:** ∇f = [∂f/∂x, ∂f/∂y]
**3D:** ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]

**Key properties:**
1. **Direction:** ∇f points in the direction of steepest ascent
2. **Magnitude:** |∇f| gives the rate of steepest ascent
3. **Level curves:** ∇f is perpendicular to level curves (curves where f = constant)

**Connection to GR:** The gradient generalizes to the covariant derivative, which measures how tensor fields change in curved spacetime.

### 💻 Code Example

```python
# Gradient of f(x,y) = x² + y²
def gradient(f, x, y, h=1e-5):
    """Compute gradient vector [∂f/∂x, ∂f/∂y]"""
    fx = (f(x + h, y) - f(x, y)) / h
    fy = (f(x, y + h) - f(x, y)) / h
    return np.array([fx, fy])  # np.array() converts Python list/tuple to efficient numpy array

# Evaluate gradient at several points
points = [(1, 0), (0, 1), (1, 1), (-1, 1)]

print("Gradient of f(x,y) = x² + y²:")
print("Analytical: ∇f = [2x, 2y]\n")

for (x, y) in points:
    grad = gradient(f, x, y)
    grad_exact = np.array([2*x, 2*y])  # np.array() converts Python list/tuple to efficient numpy array
    print(f"At ({x}, {y}): ∇f = {grad_exact}, |∇f| = {np.linalg.norm(grad_exact):.3f}")  # np.linalg.norm() computes vector magnitude (Euclidean norm)
```

### 📊 Visualization

```python
# Visualize gradient field with level curves
x = np.linspace(-2, 2, 20)  # np.linspace() creates evenly spaced array between start and end
y = np.linspace(-2, 2, 20)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x, y)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# Gradient field
Fx = 2*X  # ∂f/∂x
Fy = 2*Y  # ∂f/∂y

plt.figure(figsize=(10, 8))  # plt.figure() creates a new figure for plotting

# Level curves (contours where f = constant)
levels = [0.5, 1, 2, 4, 6, 8]
contours = plt.contour(X, Y, f(X, Y), levels=levels, colors=COLORS['blue'], alpha=0.5)  # plt.contour() draws contour lines (level curves)
plt.clabel(contours, inline=True, fontsize=10)

# Gradient vectors
plt.quiver(X, Y, Fx, Fy, color=COLORS['red'], alpha=0.6, scale=50)  # plt.quiver() draws arrow/vector field plot

# Highlight: gradient is perpendicular to level curves
x0, y0 = 1.5, 1.0
grad = np.array([2*x0, 2*y0])  # np.array() converts Python list/tuple to efficient numpy array
plt.arrow(x0, y0, grad[0]*0.3, grad[1]*0.3, head_width=0.15,  # plt.arrow() draws arrow from (x,y) with specified dx, dy
          head_length=0.1, fc=COLORS['orange'], ec=COLORS['orange'], linewidth=2)
plt.plot(x0, y0, 'o', color=COLORS['orange'], markersize=10)  # plt.plot() draws line plot
plt.text(x0 + 0.2, y0 + 0.5, '∇f', fontsize=14, color=COLORS['orange'])  # plt.text() adds text annotation at specified coordinates

plt.xlabel('x')  # plt.xlabel() sets x-axis label
plt.ylabel('y')  # plt.ylabel() sets y-axis label
plt.title('Gradient Field (red) and Level Curves (blue)\n∇f is perpendicular to level curves')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.axis('equal')
plt.show()  # plt.show() displays the figure window
```

### 🔬 Explore

- Compute the gradient of f(x,y) = sin(x) cos(y)
- Verify that ∇f is perpendicular to level curves
- Try f(x,y) = √(x² + y²) and visualize its gradient field

### 🎯 Practice Question

**Q:** A hiker on a hill described by h(x,y) = 100 - x² - 2y² is at point (3, 2). In which direction should they walk to ascend most steeply?

<details><summary>Hint 1</summary>
The gradient ∇h points in the direction of steepest ascent.
</details>

<details><summary>Hint 2</summary>
∇h = [-2x, -4y]. Evaluate at (3, 2).
</details>

<details><summary>Answer</summary>
∇h = [-2x, -4y] = [-6, -8] at (3, 2)

The hiker should walk in direction [-6, -8] or equivalently [-3, -4] (normalized: [-0.6, -0.8]) to ascend most steeply. This means walking mostly in the negative y-direction (downward in the y-coordinate) with some negative x-component.
</details>

---

## 3. Directional Derivatives

### 📖 Concept

The **directional derivative** D_u f measures the rate of change of f in an arbitrary direction u (a unit vector).

**Formula:** D_u f = ∇f · u = |∇f| cos(θ)

where θ is the angle between ∇f and u.

**Key insights:**
- Maximum when u points in direction of ∇f (θ = 0): D_u f = |∇f|
- Zero when u ⊥ ∇f (θ = 90°): moving along level curve
- Minimum when u points opposite to ∇f (θ = 180°): D_u f = -|∇f|

**Connection to GR:** In curved spacetime, we'll generalize this to derivatives along arbitrary curves.

### 💻 Code Example

```python
# Function f(x,y) = x² + y²
def f(x, y):
    return x**2 + y**2

# Gradient
def grad_f(x, y):
    return np.array([2*x, 2*y])  # np.array() converts Python list/tuple to efficient numpy array

# Directional derivative
def directional_derivative(grad, u):
    """D_u f = ∇f · u"""
    # Normalize u to be unit vector
    u_hat = u / np.linalg.norm(u)  # np.linalg.norm() computes vector magnitude (Euclidean norm)
    return np.dot(grad, u_hat)  # np.dot() computes dot product of two arrays

# Point of interest
x0, y0 = 1, 1
grad = grad_f(x0, y0)

print(f"At point ({x0}, {y0}):")
print(f"∇f = {grad}")
print(f"|∇f| = {np.linalg.norm(grad):.3f}\n")  # np.linalg.norm() computes vector magnitude (Euclidean norm)

# Try different directions
directions = {
    "Along ∇f": grad,
    "Perpendicular to ∇f": np.array([-grad[1], grad[0]]),  # np.array() converts Python list/tuple to efficient numpy array
    "Opposite to ∇f": -grad,
    "45° from ∇f": np.array([1, 0])  # np.array() converts Python list/tuple to efficient numpy array
}

for name, u in directions.items():
    D_u = directional_derivative(grad, u)
    u_hat = u / np.linalg.norm(u)  # np.linalg.norm() computes vector magnitude (Euclidean norm)
    angle = np.degrees(np.arccos(np.dot(grad, u_hat) / np.linalg.norm(grad)))  # np.dot() computes dot product of two arrays
    print(f"{name}: u = {u_hat}, D_u f = {D_u:.3f}, angle = {angle:.1f}°")
```

### 📊 Visualization

```python
# Visualize directional derivatives in all directions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Gradient and test directions
x = np.linspace(-2, 2, 20)  # np.linspace() creates evenly spaced array between start and end
y = np.linspace(-2, 2, 20)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x, y)  # np.meshgrid() creates coordinate matrices from coordinate vectors
Z = f(X, Y)

ax1.contour(X, Y, Z, levels=10, colors=COLORS['gray'], alpha=0.3)
ax1.plot(x0, y0, 'o', color=COLORS['red'], markersize=12)

# Draw gradient
ax1.arrow(x0, y0, grad[0]*0.3, grad[1]*0.3, head_width=0.1,
          color=COLORS['red'], linewidth=2, label='∇f (max increase)')

# Draw various directional derivatives
angles = np.linspace(0, 2*np.pi, 8, endpoint=False)  # np.linspace() creates evenly spaced array between start and end
for angle in angles:
    u = np.array([np.cos(angle), np.sin(angle)])  # np.array() converts Python list/tuple to efficient numpy array
    D_u = directional_derivative(grad, u)
    color = COLORS['green'] if D_u > 0 else COLORS['blue']
    ax1.arrow(x0, y0, u[0]*0.3, u[1]*0.3, head_width=0.05,
              color=color, linewidth=1, alpha=0.6)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Directional Derivatives\nGreen = increasing, Blue = decreasing')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
ax1.legend()

# Right: Polar plot of directional derivative vs angle
theta = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
r = np.array([directional_derivative(grad, np.array([np.cos(t), np.sin(t)])) for t in theta])  # np.array() converts Python list/tuple to efficient numpy array

ax2 = plt.subplot(122, projection='polar')  # plt.subplot() creates subplot in grid layout (rows, cols, position)
ax2.plot(theta, r, color=COLORS['blue'], linewidth=2)
ax2.fill(theta, r, alpha=0.3, color=COLORS['blue'])
ax2.set_title('Directional Derivative vs Direction\nMax along ∇f, zero perpendicular to ∇f', pad=20)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

### 🎯 Practice Question

**Q:** For f(x,y) = xy at point (2,3), find the directional derivative in the direction of u = [3, 4].

<details><summary>Hint 1</summary>
First compute ∇f = [y, x]. Then normalize u to get unit vector.
</details>

<details><summary>Hint 2</summary>
∇f = [3, 2] at (2,3). Unit vector: û = [3,4]/5 = [0.6, 0.8]
</details>

<details><summary>Answer</summary>
∇f = [y, x] = [3, 2] at (2,3)
û = [3,4]/√(9+16) = [3,4]/5 = [0.6, 0.8]
D_u f = ∇f · û = 3(0.6) + 2(0.8) = 1.8 + 1.6 = 3.4

The function increases at rate 3.4 in the direction [3, 4].
</details>

---

## 4. Chain Rule in Multiple Variables

### 📖 Concept

The **multivariable chain rule** tells us how to differentiate composite functions.

**Case 1: f(x(t), y(t))** - function of curve parameters
```
df/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt) = ∇f · r'(t)
```

**Case 2: f(x(s,t), y(s,t))** - change of variables
```
∂f/∂s = (∂f/∂x)(∂x/∂s) + (∂f/∂y)(∂y/∂s)
∂f/∂t = (∂f/∂x)(∂x/∂t) + (∂f/∂y)(∂y/∂t)
```

**Connection to GR:** Coordinate transformations in spacetime use the multivariable chain rule extensively. Tensor transformation laws are built on this foundation.

### 💻 Code Example

```python
# Example: Temperature T(x,y) = x² + y² along a path
def T(x, y):
    """Temperature field"""
    return x**2 + y**2

def grad_T(x, y):
    """Gradient of temperature"""
    return np.array([2*x, 2*y])  # np.array() converts Python list/tuple to efficient numpy array

# Path: x(t) = cos(t), y(t) = sin(t) (circle)
def path(t):
    """Circular path"""
    x = np.cos(t)  # np.cos() computes cosine (element-wise for arrays)
    y = np.sin(t)  # np.sin() computes sine (element-wise for arrays)
    return x, y

def path_derivative(t):
    """Velocity along path: r'(t)"""
    dx_dt = -np.sin(t)  # np.sin() computes sine (element-wise for arrays)
    dy_dt = np.cos(t)  # np.cos() computes cosine (element-wise for arrays)
    return np.array([dx_dt, dy_dt])  # np.array() converts Python list/tuple to efficient numpy array

# Compute dT/dt using chain rule
t = np.pi/4  # Time parameter
x, y = path(t)
grad = grad_T(x, y)
velocity = path_derivative(t)

dT_dt_chain = np.dot(grad, velocity)  # np.dot() computes dot product of two arrays

print(f"Temperature T(x,y) = x² + y² along circular path")
print(f"Path: x(t) = cos(t), y(t) = sin(t)")
print(f"\nAt t = π/4:")
print(f"  Position: ({x:.3f}, {y:.3f})")
print(f"  ∇T = {grad}")
print(f"  Velocity r'(t) = {velocity}")
print(f"  dT/dt = ∇T · r'(t) = {dT_dt_chain:.3f}")

# Verify by direct computation
def T_of_t(t):
    x, y = path(t)
    return T(x, y)

h = 1e-5
dT_dt_direct = (T_of_t(t + h) - T_of_t(t)) / h
print(f"\nDirect computation: dT/dt = {dT_dt_direct:.3f}")
print(f"Match? {np.isclose(dT_dt_chain, dT_dt_direct)}")  # np.isclose() tests if values are approximately equal (handles floating point)
```

### 📊 Visualization

```python
# Visualize temperature field and circular path
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Temperature field
x = np.linspace(-1.5, 1.5, 100)  # np.linspace() creates evenly spaced array between start and end
y = np.linspace(-1.5, 1.5, 100)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x, y)  # np.meshgrid() creates coordinate matrices from coordinate vectors
Z = T(X, Y)

contours = ax1.contourf(X, Y, Z, levels=20, cmap='hot', alpha=0.6)
plt.colorbar(contours, ax=ax1, label='Temperature')  # plt.colorbar() adds color scale bar to plot

# Circular path
t_path = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
x_path = np.cos(t_path)  # np.cos() computes cosine (element-wise for arrays)
y_path = np.sin(t_path)  # np.sin() computes sine (element-wise for arrays)
ax1.plot(x_path, y_path, color=COLORS['blue'], linewidth=3, label='Path')

# Point at t = π/4
t = np.pi/4
x, y = path(t)
grad = grad_T(x, y)
velocity = path_derivative(t)

ax1.plot(x, y, 'o', color=COLORS['cyan'], markersize=12)
ax1.arrow(x, y, grad[0]*0.15, grad[1]*0.15, head_width=0.08,
          color=COLORS['red'], linewidth=2, label='∇T')
ax1.arrow(x, y, velocity[0]*0.3, velocity[1]*0.3, head_width=0.08,
          color=COLORS['green'], linewidth=2, label="r'(t)")

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Temperature Field T(x,y) = x² + y²\nwith Circular Path')
ax1.legend()
ax1.axis('equal')
ax1.grid(True, alpha=0.3)

# Temperature vs time along path
t_vals = np.linspace(0, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
T_vals = [T(*path(t)) for t in t_vals]
dT_dt_vals = [np.dot(grad_T(*path(t)), path_derivative(t)) for t in t_vals]  # np.dot() computes dot product of two arrays

ax2_twin = ax2.twinx()
ax2.plot(t_vals, T_vals, color=COLORS['blue'], linewidth=2, label='T(t)')
ax2_twin.plot(t_vals, dT_dt_vals, color=COLORS['red'], linewidth=2,
              linestyle='--', label='dT/dt')

ax2.axhline(y=1, color='k', linestyle=':', alpha=0.5)
ax2_twin.axhline(y=0, color='k', linestyle=':', alpha=0.5)

ax2.set_xlabel('t')
ax2.set_ylabel('T(t)', color=COLORS['blue'])
ax2_twin.set_ylabel('dT/dt', color=COLORS['red'])
ax2.set_title('Temperature and Rate of Change Along Path')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("\nNote: On the circle, T(t) = cos²(t) + sin²(t) = 1 (constant!)")
print("Therefore dT/dt = 0 everywhere, as shown in the right plot.")
```

### 🎯 Practice Question

**Q:** If f(x,y) = x²y and x = 2t, y = t³, find df/dt at t = 1.

<details><summary>Hint 1</summary>
Use chain rule: df/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)
</details>

<details><summary>Hint 2</summary>
∂f/∂x = 2xy, ∂f/∂y = x²
dx/dt = 2, dy/dt = 3t²
</details>

<details><summary>Answer</summary>
At t = 1: x = 2, y = 1
∂f/∂x = 2xy = 2(2)(1) = 4
∂f/∂y = x² = 4
dx/dt = 2
dy/dt = 3t² = 3

df/dt = (4)(2) + (4)(3) = 8 + 12 = 20
</details>

---

## 5. Divergence and Curl

### 📖 Concept

For a vector field **F** = [F₁, F₂, F₃], we have two important differential operators:

**DIVERGENCE** (scalar output):
```
div F = ∇ · F = ∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z
```
Measures "outflow" from a point. Positive = source, negative = sink.

**CURL** (vector output):
```
curl F = ∇ × F = [∂F₃/∂y - ∂F₂/∂z, ∂F₁/∂z - ∂F₃/∂x, ∂F₂/∂x - ∂F₁/∂y]
```
Measures "rotation" of the field. Points along axis of rotation.

**Connection to GR:** Divergence generalizes to covariant divergence of tensors. The Einstein field equations involve the divergence of the stress-energy tensor.

### 💻 Code Example

```python
# Vector field F = [y, -x, 0] (rotation around z-axis)
def F_rotation(x, y, z):
    """Rotational field"""
    return np.array([y, -x, 0])  # np.array() converts Python list/tuple to efficient numpy array

# Vector field G = [x, y, z] (radial expansion)
def F_radial(x, y, z):
    """Radial field"""
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

def divergence(F, x, y, z, h=1e-5):
    """Compute div F = ∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z"""
    dF1_dx = (F(x+h, y, z)[0] - F(x, y, z)[0]) / h
    dF2_dy = (F(x, y+h, z)[1] - F(x, y, z)[1]) / h
    dF3_dz = (F(x, y, z+h)[2] - F(x, y, z)[2]) / h
    return dF1_dx + dF2_dy + dF3_dz

def curl(F, x, y, z, h=1e-5):
    """Compute curl F = ∇ × F"""
    # ∂F₃/∂y - ∂F₂/∂z
    curl_x = ((F(x, y+h, z)[2] - F(x, y, z)[2]) / h -
              (F(x, y, z+h)[1] - F(x, y, z)[1]) / h)
    # ∂F₁/∂z - ∂F₃/∂x
    curl_y = ((F(x, y, z+h)[0] - F(x, y, z)[0]) / h -
              (F(x+h, y, z)[2] - F(x, y, z)[2]) / h)
    # ∂F₂/∂x - ∂F₁/∂y
    curl_z = ((F(x+h, y, z)[1] - F(x, y, z)[1]) / h -
              (F(x, y+h, z)[0] - F(x, y, z)[0]) / h)
    return np.array([curl_x, curl_y, curl_z])  # np.array() converts Python list/tuple to efficient numpy array

# Test point
x, y, z = 1, 1, 0

print("ROTATIONAL FIELD F = [y, -x, 0]")
print(f"At ({x}, {y}, {z}):")
div_F = divergence(F_rotation, x, y, z)
curl_F = curl(F_rotation, x, y, z)
print(f"  div F = {div_F:.6f} (should be 0 - no sources/sinks)")
print(f"  curl F = {curl_F} (should be [0, 0, -2] - rotation around z)")

print("\nRADIAL FIELD G = [x, y, z]")
div_G = divergence(F_radial, x, y, z)
curl_G = curl(F_radial, x, y, z)
print(f"  div G = {div_G:.6f} (should be 3 - expansion)")
print(f"  curl G = {curl_G} (should be [0, 0, 0] - no rotation)")
```

### 📊 Visualization

```python
# Visualize divergence and curl
fig = plt.figure(figsize=(14, 6))  # plt.figure() creates a new figure for plotting

# Rotational field (has curl, no divergence)
ax1 = fig.add_subplot(121)
x = np.linspace(-2, 2, 15)  # np.linspace() creates evenly spaced array between start and end
y = np.linspace(-2, 2, 15)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x, y)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# F = [y, -x, 0]
U = Y
V = -X

ax1.quiver(X, Y, U, V, color=COLORS['blue'], alpha=0.6)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Rotational Field F = [y, -x, 0]\ndiv F = 0, curl F ≠ 0')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Radial field (has divergence, no curl)
ax2 = fig.add_subplot(122)
U = X
V = Y

ax2.quiver(X, Y, U, V, color=COLORS['red'], alpha=0.6)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Radial Field G = [x, y, 0]\ndiv G > 0, curl G = 0')
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

### 📊 3D Curl Visualization

Let's visualize curl in 3D to better understand the rotation:

```python
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit

# Create a 3D rotational field F = [-y, x, 0.5*z]
def F_3d(x, y, z):
    """3D vector field with rotation around z-axis and upward component"""
    return np.array([-y, x, 0.5*z])  # np.array() converts Python list/tuple to efficient numpy array

# Compute curl at grid points
x = np.linspace(-1, 1, 5)  # np.linspace() creates evenly spaced array between start and end
y = np.linspace(-1, 1, 5)  # np.linspace() creates evenly spaced array between start and end
z = np.linspace(-1, 1, 5)  # np.linspace() creates evenly spaced array between start and end
X, Y, Z = np.meshgrid(x, y, z)  # np.meshgrid() creates 3D coordinate grids

# Compute field values
U = -Y  # x-component of F
V = X   # y-component of F
W = 0.5*Z  # z-component of F

# Compute curl analytically for this field
# curl F = [∂W/∂y - ∂V/∂z, ∂U/∂z - ∂W/∂x, ∂V/∂x - ∂U/∂y]
# For F = [-y, x, 0.5z]: curl F = [0, 0, 1+1] = [0, 0, 2]
curl_U = np.zeros_like(X)  # x-component of curl
curl_V = np.zeros_like(Y)  # y-component of curl
curl_W = 2*np.ones_like(Z)  # z-component of curl (constant = 2)

fig = plt.figure(figsize=(14, 6))  # plt.figure() creates a new figure for plotting

# Left: The vector field
ax1 = fig.add_subplot(121, projection='3d')  # Create 3D axes
ax1.quiver(X, Y, Z, U, V, W, length=0.3, color=COLORS['blue'],
          arrow_length_ratio=0.3, alpha=0.6, linewidth=1.5)  # plt.quiver() draws 3D arrow/vector field
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.set_zlabel('z', fontsize=10)
ax1.set_title('Vector Field F = [-y, x, 0.5z]\n(Rotating + Rising)', fontsize=11, fontweight='bold')

# Right: The curl field
ax2 = fig.add_subplot(122, projection='3d')  # Create 3D axes
ax2.quiver(X, Y, Z, curl_U, curl_V, curl_W, length=0.4, color=COLORS['red'],
          arrow_length_ratio=0.4, alpha=0.8, linewidth=2)  # plt.quiver() draws 3D arrow/vector field
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.set_zlabel('z', fontsize=10)
ax2.set_title('Curl Field: ∇ × F = [0, 0, 2]\n(Points along rotation axis)', fontsize=11, fontweight='bold')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Key insight: Curl points along the axis of rotation!")
print("For F = [-y, x, 0.5z], rotation is around z-axis → curl points in +z direction")
```

**What you should see:**
- **Left:** Blue arrows showing circular motion in xy-plane + upward motion in z
- **Right:** Red arrows all pointing straight up (+z), showing rotation axis

This demonstrates that **curl is a pseudovector** that points perpendicular to the plane of rotation!

### 🎯 Practice Question

**Q:** For the vector field **F** = [x², xy, 0], compute div **F** and curl **F**.

<details><summary>Hint 1</summary>
div F = ∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z
</details>

<details><summary>Hint 2</summary>
∂(x²)/∂x = 2x, ∂(xy)/∂y = x, ∂(0)/∂z = 0
</details>

<details><summary>Answer</summary>
div **F** = ∂(x²)/∂x + ∂(xy)/∂y + ∂(0)/∂z = 2x + x + 0 = 3x

For curl:
- curl_x = ∂F₃/∂y - ∂F₂/∂z = 0 - 0 = 0
- curl_y = ∂F₁/∂z - ∂F₃/∂x = 0 - 0 = 0
- curl_z = ∂F₂/∂x - ∂F₁/∂y = y - 0 = y

curl **F** = [0, 0, y]
</details>

---

## 6. Optimization and Critical Points

### 📖 Concept

To find extrema of f(x,y):

**Step 1:** Find critical points where ∇f = 0
- Solve: ∂f/∂x = 0 and ∂f/∂y = 0

**Step 2:** Use the second derivative test
- Compute Hessian matrix: H = [[f_xx, f_xy], [f_xy, f_yy]]
- Determinant: D = f_xx · f_yy - f_xy²

**Classification:**
- D > 0 and f_xx > 0: local minimum
- D > 0 and f_xx < 0: local maximum
- D < 0: saddle point
- D = 0: inconclusive

**Connection to GR:** Geodesics (straightest paths in curved spacetime) are found by minimizing/extremizing the spacetime interval.

### 💻 Code Example

```python
from scipy.optimize import minimize  # Function minimization

# Example: f(x,y) = x² + 2y² - 4x - 4y + 5
def f_optimize(xy):
    x, y = xy
    return x**2 + 2*y**2 - 4*x - 4*y + 5

def grad_f_optimize(xy):
    x, y = xy
    df_dx = 2*x - 4
    df_dy = 4*y - 4
    return np.array([df_dx, df_dy])  # np.array() converts Python list/tuple to efficient numpy array

def hessian_f(xy):
    """Compute Hessian matrix [[f_xx, f_xy], [f_xy, f_yy]]"""
    return np.array([[2, 0], [0, 4]])  # np.array() converts Python list/tuple to efficient numpy array

# Find critical point: ∇f = 0
print("Finding critical point of f(x,y) = x² + 2y² - 4x - 4y + 5")
print("\nSolve ∇f = 0:")
print("  ∂f/∂x = 2x - 4 = 0  →  x = 2")
print("  ∂f/∂y = 4y - 4 = 0  →  y = 1")
x_crit = np.array([2, 1])  # np.array() converts Python list/tuple to efficient numpy array

# Second derivative test
H = hessian_f(x_crit)
D = np.linalg.det(H)  # np.linalg.det() computes matrix determinant
f_xx = H[0,0]

print(f"\nCritical point: ({x_crit[0]}, {x_crit[1]})")
print(f"Hessian:\n{H}")
print(f"Determinant D = {D}")
print(f"f_xx = {f_xx}")

if D > 0 and f_xx > 0:
    print("→ Local MINIMUM")
elif D > 0 and f_xx < 0:
    print("→ Local MAXIMUM")
elif D < 0:
    print("→ SADDLE POINT")
else:
    print("→ Inconclusive")

print(f"\nValue at critical point: f({x_crit[0]}, {x_crit[1]}) = {f_optimize(x_crit):.3f}")

# Verify with numerical optimization
result = minimize(f_optimize, [0, 0], jac=grad_f_optimize)
print(f"\nNumerical optimization:")
print(f"  Minimum at: {result.x}")
print(f"  Value: {result.fun:.3f}")
```

### 📊 Visualization

```python
# Visualize function and critical point
x = np.linspace(-1, 5, 100)  # np.linspace() creates evenly spaced array between start and end
y = np.linspace(-2, 4, 100)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x, y)  # np.meshgrid() creates coordinate matrices from coordinate vectors
Z = X**2 + 2*Y**2 - 4*X - 4*Y + 5

fig = plt.figure(figsize=(14, 5))  # plt.figure() creates a new figure for plotting

# 3D surface
ax1 = fig.add_subplot(131, projection='3d')  projection='3d'  # Create 3D axes
ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')  # .plot_surface() draws 3D surface plot
ax1.plot([2], [1], [f_optimize([2, 1])], 'ro', markersize=10)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('Surface with Minimum')

# Contour plot
ax2 = fig.add_subplot(132)
contours = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.clabel(contours, inline=True, fontsize=8)
ax2.plot(2, 1, 'ro', markersize=10, label='Minimum (2, 1)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Contour Plot')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Gradient field showing flow to minimum
ax3 = fig.add_subplot(133)
x_grad = np.linspace(-1, 5, 20)  # np.linspace() creates evenly spaced array between start and end
y_grad = np.linspace(-2, 4, 20)  # np.linspace() creates evenly spaced array between start and end
X_grad, Y_grad = np.meshgrid(x_grad, y_grad)  # np.meshgrid() creates coordinate matrices from coordinate vectors
U = -(2*X_grad - 4)  # Negative gradient (descent direction)
V = -(4*Y_grad - 4)

ax3.quiver(X_grad, Y_grad, U, V, alpha=0.6, color=COLORS['blue'])
ax3.plot(2, 1, 'ro', markersize=12, label='Minimum')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Gradient Descent Flow')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

### 🎯 Practice Question

**Q:** Find and classify all critical points of f(x,y) = x² - y².

<details><summary>Hint 1</summary>
Solve ∇f = 0: [2x, -2y] = [0, 0]
</details>

<details><summary>Hint 2</summary>
Compute Hessian and its determinant at the critical point.
</details>

<details><summary>Answer</summary>
Critical point: (0, 0) from solving 2x = 0, -2y = 0

Hessian: H = [[2, 0], [0, -2]]
Determinant: D = (2)(-2) - 0² = -4 < 0

**Result: SADDLE POINT at (0, 0)**

This is a classic saddle point - minimum along x-axis, maximum along y-axis.
</details>

---

## 7. Lagrange Multipliers

### 📖 Concept

**Problem:** Optimize f(x,y) subject to constraint g(x,y) = c

**Lagrange's Method:** At the optimum, ∇f and ∇g are parallel:
```
∇f = λ∇g
```

**System to solve:**
1. ∂f/∂x = λ ∂g/∂x
2. ∂f/∂y = λ ∂g/∂y
3. g(x,y) = c

The parameter λ is called the **Lagrange multiplier**.

**Geometric intuition:** The maximum/minimum of f on the constraint curve g = c occurs where the level curve of f is tangent to the constraint curve.

**Connection to GR:** Lagrangian mechanics (which we'll use in Lesson 8) uses this principle. The Einstein-Hilbert action in GR is also based on variational principles.

### 💻 Code Example

```python
from scipy.optimize import minimize  # Function minimization

# Maximize f(x,y) = xy subject to x + y = 10
def f_lagrange(xy):
    x, y = xy
    return -(x * y)  # Negative because minimize finds minimum

def constraint(xy):
    x, y = xy
    return x + y - 10  # g(x,y) = 0 form

# Solve using Lagrange multipliers analytically
print("Maximize f(x,y) = xy subject to x + y = 10\n")
print("Using Lagrange multipliers:")
print("∇f = λ∇g")
print("[y, x] = λ[1, 1]")
print("→ y = λ and x = λ")
print("→ x = y")
print("\nFrom constraint x + y = 10:")
print("x + x = 10  →  x = 5")
print("Therefore: x = 5, y = 5")
print(f"Maximum value: f(5,5) = {5*5}")

# Verify numerically
from scipy.optimize import minimize  # Function minimization

cons = {'type': 'eq', 'fun': constraint}
result = minimize(f_lagrange, [1, 1], constraints=cons)

print(f"\nNumerical verification:")
print(f"  x = {result.x[0]:.3f}, y = {result.x[1]:.3f}")
print(f"  f(x,y) = {-result.fun:.3f}")  # Negative because we minimized -f
```

### 📊 Visualization

```python
# Visualize constrained optimization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Contour plot with constraint
x = np.linspace(0, 12, 200)  # np.linspace() creates evenly spaced array between start and end
y = np.linspace(0, 12, 200)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x, y)  # np.meshgrid() creates coordinate matrices from coordinate vectors
Z = X * Y

levels = [10, 25, 50, 75, 100, 150, 200]
contours = ax1.contour(X, Y, Z, levels=levels, colors=COLORS['blue'], alpha=0.5)
ax1.clabel(contours, inline=True, fontsize=10)

# Constraint line: x + y = 10
x_constraint = np.linspace(0, 10, 100)  # np.linspace() creates evenly spaced array between start and end
y_constraint = 10 - x_constraint
ax1.plot(x_constraint, y_constraint, color=COLORS['red'], linewidth=3, label='Constraint: x + y = 10')

# Optimum point
ax1.plot(5, 5, 'o', color=COLORS['orange'], markersize=15, label='Maximum (5, 5)')

# Gradients at optimum
grad_f = np.array([5, 5])  # ∇f = [y, x] at (5,5)
grad_g = np.array([1, 1])  # ∇g = [1, 1]

ax1.arrow(5, 5, grad_f[0]*0.3, grad_f[1]*0.3, head_width=0.3,
          color=COLORS['blue'], linewidth=2, label='∇f')
ax1.arrow(5, 5, grad_g[0]*0.5, grad_g[1]*0.5, head_width=0.3,
          color=COLORS['red'], linewidth=2, label='∇g', linestyle='--')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Lagrange Multipliers: Maximize xy subject to x + y = 10\n∇f ∥ ∇g at optimum')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 12)
ax1.axis('equal')

# 3D view
ax2 = fig.add_subplot(122, projection='3d')  projection='3d'  # Create 3D axes
ax2.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')  # .plot_surface() draws 3D surface plot

# Constraint curve on surface
t = np.linspace(0, 10, 100)  # np.linspace() creates evenly spaced array between start and end
x_c = t
y_c = 10 - t
z_c = x_c * y_c
ax2.plot(x_c, y_c, z_c, color=COLORS['red'], linewidth=3, label='Constraint curve')
ax2.plot([5], [5], [25], 'o', color=COLORS['orange'], markersize=10, label='Maximum')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y) = xy')
ax2.set_title('3D View: Maximum on Constraint Curve')
ax2.legend()

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

### 🔬 Explore

Try these constrained optimization problems:
1. Minimize x² + y² subject to x + 2y = 5
2. Maximize x²y subject to x² + y² = 3
3. Find closest point on line 3x + 4y = 25 to origin

### 🎯 Practice Question

**Q:** Use Lagrange multipliers to find the maximum of f(x,y) = x + 2y subject to x² + y² = 5.

<details><summary>Hint 1</summary>
Set up: ∇f = λ∇g where g(x,y) = x² + y² - 5
</details>

<details><summary>Hint 2</summary>
[1, 2] = λ[2x, 2y] gives: 1 = 2λx and 2 = 2λy
</details>

<details><summary>Answer</summary>
From ∇f = λ∇g:
- 1 = 2λx  →  x = 1/(2λ)
- 2 = 2λy  →  y = 1/λ  →  y = 2x

Substitute into constraint:
x² + (2x)² = 5  →  5x² = 5  →  x = ±1

Two critical points: (1, 2) and (-1, -2)
- f(1, 2) = 1 + 2(2) = 5 (maximum)
- f(-1, -2) = -1 + 2(-2) = -5 (minimum)

**Maximum: f = 5 at (1, 2)**
</details>

---

## Summary and Next Steps

You've completed multivariable calculus! Key concepts:

✓ Partial derivatives and the gradient vector
✓ Directional derivatives in arbitrary directions
✓ Chain rule for composite functions
✓ Divergence and curl of vector fields
✓ Optimization with second derivative test
✓ Constrained optimization with Lagrange multipliers

**Connection to General Relativity:**
- Gradient → Covariant derivative in curved spacetime
- Directional derivatives → Derivatives along worldlines
- Divergence → Covariant divergence of stress-energy tensor
- Optimization → Geodesic equations (extremal paths)
- Lagrange multipliers → Lagrangian mechanics foundation

**Next Lesson:** Differential Equations (essential for physics equations in GR)

---

**Need Help?** Use the AI assistant:
```python
from utils.ai_assistant import AIAssistant
assistant = AIAssistant()
assistant.set_lesson_context("Lesson 2", "Multivariable Calculus",
                             ["gradient", "divergence", "optimization"])
assistant.ask("Why is the gradient perpendicular to level curves?")
```
