# Lesson 4: Curves and Surfaces

**Topics:** Parametric curves, arc length, curvature, torsion, Frenet-Serret frame, parametric surfaces, first and second fundamental forms

**Prerequisites:** Linear algebra (Lesson 1), multivariable calculus (Lesson 2), differential equations (Lesson 3)

**Time:** ~5-6 hours

---

## Table of Contents

1. [Parametric Curves in 3D](#1-parametric-curves-in-3d)
2. [Arc Length and Reparametrization](#2-arc-length-and-reparametrization)
3. [Curvature and Osculating Circle](#3-curvature-and-osculating-circle)
4. [Torsion and the Frenet-Serret Frame](#4-torsion-and-the-frenet-serret-frame)
5. [Parametric Surfaces](#5-parametric-surfaces)
6. [First Fundamental Form](#6-first-fundamental-form)
7. [Second Fundamental Form and Gaussian Curvature](#7-second-fundamental-form-and-gaussian-curvature)

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

## 1. Parametric Curves in 3D

### 📖 Concept

A **parametric curve** in 3D is a smooth path through space described by a parameter t:

```
r(t) = (x(t), y(t), z(t))
```

Unlike functions y = f(x), parametric curves can loop, spiral, and move freely in any direction.

**Key Examples:**
- **Line:** r(t) = r₀ + tv (position + velocity × time)
- **Circle:** r(t) = (R cos(t), R sin(t), 0)
- **Helix:** r(t) = (R cos(t), R sin(t), ht) - spirals upward

**The velocity vector** is the derivative:
```
v(t) = r'(t) = (x'(t), y'(t), z'(t))
```

This points in the direction of motion, and its magnitude |v(t)| is the speed.

**Why this matters for GR:** In General Relativity, particles follow curved paths (geodesics) through spacetime. Understanding how to describe and measure curves is fundamental to understanding gravity as curved geometry.

---

### 💻 Code Example: Defining Curves

Copy this into your Python terminal:

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Define a helix
def helix(t):
    """Parametric helix: spirals upward"""
    R = 1.0  # radius
    h = 0.3  # vertical spacing
    x = R * np.cos(t)  # np.cos() computes cosine (element-wise for arrays)
    y = R * np.sin(t)  # np.sin() computes sine (element-wise for arrays)
    z = h * t
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

# Sample the curve
t_vals = np.linspace(0, 4*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
curve_points = np.array([helix(t) for t in t_vals])  # np.array() converts Python list/tuple to efficient numpy array

print(f"Curve defined from t=0 to t={4*np.pi:.2f}")
print(f"Number of sample points: {len(t_vals)}")
print(f"Starting point: {helix(0)}")
print(f"Ending point: {helix(4*np.pi)}")
```

**Expected output:**
```
Curve defined from t=0 to t=12.57
Number of sample points: 200
Starting point: [1. 0. 0.]
Ending point: [ 1.00000000e+00 -2.44929360e-16  3.76991118e+00]
```

---

### 📊 Visualization: Helix with Velocity Vectors

```python
fig = plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Left: The helix curve
ax1 = fig.add_subplot(121, projection='3d')  projection='3d'  # Create 3D axes
ax1.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2],
         color=COLORS['blue'], linewidth=2, label='Helix')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Parametric Helix: r(t) = (cos t, sin t, 0.3t)')
ax1.legend()

# Right: Helix with velocity vectors
ax2 = fig.add_subplot(122, projection='3d')  projection='3d'  # Create 3D axes
ax2.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2],
         color=COLORS['blue'], linewidth=1, alpha=0.3)

# Show velocity vectors at several points
def helix_velocity(t):
    """Velocity vector: derivative of position"""
    R = 1.0
    h = 0.3
    vx = -R * np.sin(t)  # np.sin() computes sine (element-wise for arrays)
    vy = R * np.cos(t)  # np.cos() computes cosine (element-wise for arrays)
    vz = h
    return np.array([vx, vy, vz])  # np.array() converts Python list/tuple to efficient numpy array

# Plot velocity at several points
sample_times = np.linspace(0, 4*np.pi, 8)  # np.linspace() creates evenly spaced array between start and end
for t in sample_times:
    pos = helix(t)
    vel = helix_velocity(t)
    # Scale for visibility
    vel_scaled = vel * 0.5
    ax2.quiver(pos[0], pos[1], pos[2],
              vel_scaled[0], vel_scaled[1], vel_scaled[2],
              color=COLORS['orange'], arrow_length_ratio=0.3, linewidth=2)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Helix with Velocity Vectors r\'(t)')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

**What you should see:**
- **Left:** A smooth spiral rising upward
- **Right:** Orange velocity vectors tangent to the curve, showing direction of motion

---

### 🔬 Explore on Your Own

Try these different curves:

**1. Circle in 3D:**
```python
def circle_3d(t):
    return np.array([2*np.cos(t), 2*np.sin(t), 1.5])  # Circle at height z=1.5

t_vals = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
points = np.array([circle_3d(t) for t in t_vals])  # np.array() converts Python list/tuple to efficient numpy array

fig = plt.figure(figsize=(8, 8))  # plt.figure() creates a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  projection='3d'  # Create 3D axes
ax.plot(points[:, 0], points[:, 1], points[:, 2], color=COLORS['green'], linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()  # plt.show() displays the figure window
```

**2. Trefoil knot:**
```python
def trefoil(t):
    x = np.sin(t) + 2*np.sin(2*t)  # np.sin() computes sine (element-wise for arrays)
    y = np.cos(t) - 2*np.cos(2*t)  # np.cos() computes cosine (element-wise for arrays)
    z = -np.sin(3*t)  # np.sin() computes sine (element-wise for arrays)
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

t_vals = np.linspace(0, 2*np.pi, 500)  # np.linspace() creates evenly spaced array between start and end
points = np.array([trefoil(t) for t in t_vals])  # np.array() converts Python list/tuple to efficient numpy array

fig = plt.figure(figsize=(8, 8))  # plt.figure() creates a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  projection='3d'  # Create 3D axes
ax.plot(points[:, 0], points[:, 1], points[:, 2], color=COLORS['purple'], linewidth=2)
ax.set_title('Trefoil Knot')
plt.show()  # plt.show() displays the figure window
```

---

### 🎯 Practice Question #1

**Q:** For the curve r(t) = (t, t², t³), what is the velocity vector at t = 1?

<details>
<summary>💡 Hint 1</summary>

The velocity is the derivative: v(t) = r'(t) = (dx/dt, dy/dt, dz/dt)
</details>

<details>
<summary>💡 Hint 2</summary>

Differentiate each component:
- dx/dt = 1
- dy/dt = 2t
- dz/dt = 3t²
</details>

<details>
<summary>✅ Answer</summary>

v(1) = (1, 2(1), 3(1)²) = **(1, 2, 3)**

```python
def curve(t):
    return np.array([t, t**2, t**3])  # np.array() converts Python list/tuple to efficient numpy array

def velocity(t):
    return np.array([1, 2*t, 3*t**2])  # np.array() converts Python list/tuple to efficient numpy array

t = 1
print(f"Position at t=1: {curve(t)}")
print(f"Velocity at t=1: {velocity(t)}")
# Position at t=1: [1 1 1]
# Velocity at t=1: [1 2 3]
```
</details>

---

## 2. Arc Length and Reparametrization

### 📖 Concept

The **arc length** measures the actual distance traveled along a curve from t = a to t = b:

```
L = ∫ₐᵇ |r'(t)| dt = ∫ₐᵇ √(x'² + y'² + z'²) dt
```

This is fundamental because:
1. It's independent of how you parametrize the curve
2. It measures intrinsic geometric distance
3. In GR, proper time is the arc length of a worldline!

**Arc length parametrization** means using distance s as the parameter instead of t. When r(s) is parametrized by arc length, |r'(s)| = 1 (unit speed).

**Key formula:** If you know s(t) = arc length from start, then:
```
ds/dt = |r'(t)|  (speed)
```

**Why this matters for GR:** Proper time τ in relativity is the arc length of a particle's worldline through spacetime. The metric tensor g_μν generalizes this concept to curved spacetime.

---

### 💻 Code Example: Computing Arc Length

```python
from scipy.integrate import quad  # Numerical integration (quadrature)

# Define helix again
def helix(t):
    R = 1.0
    h = 0.3
    return np.array([R*np.cos(t), R*np.sin(t), h*t])  # np.array() converts Python list/tuple to efficient numpy array

def helix_velocity(t):
    R = 1.0
    h = 0.3
    return np.array([-R*np.sin(t), R*np.cos(t), h])  # np.array() converts Python list/tuple to efficient numpy array

# Speed = magnitude of velocity
def speed(t):
    v = helix_velocity(t)
    return np.linalg.norm(v)  # np.linalg.norm() computes vector magnitude (Euclidean norm)

# Compute arc length from t=0 to t=2π (one full turn)
arc_length, error = quad(speed, 0, 2*np.pi)  # quad() performs numerical integration using adaptive quadrature

print(f"Speed at any time: {speed(0):.4f} (constant for helix)")
print(f"Arc length of one turn: {arc_length:.4f}")
print(f"Integration error: {error:.2e}")

# Compare to analytical formula for helix: L = √(R² + h²) × (2π)
R, h = 1.0, 0.3
analytical = np.sqrt(R**2 + h**2) * 2*np.pi  # np.sqrt() computes square root
print(f"Analytical formula: {analytical:.4f}")
print(f"Match: {np.isclose(arc_length, analytical)}")  # np.isclose() tests if values are approximately equal (handles floating point)
```

**Expected output:**
```
Speed at any time: 1.0440 (constant for helix)
Arc length of one turn: 6.5592
Integration error: 7.28e-14
Analytical formula: 6.5592
Match: True
```

---

### 📊 Visualization: Arc Length vs Parameter

```python
# Compute arc length as function of t
t_max = 4*np.pi
t_samples = np.linspace(0, t_max, 100)  # np.linspace() creates evenly spaced array between start and end
arc_lengths = []

for t_end in t_samples:
    s, _ = quad(speed, 0, t_end)  # quad() performs numerical integration using adaptive quadrature
    arc_lengths.append(s)

arc_lengths = np.array(arc_lengths)  # np.array() converts Python list/tuple to efficient numpy array

# Plot
plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(t_samples, arc_lengths, color=COLORS['blue'], linewidth=2)  # plt.plot() draws line plot
plt.xlabel('Parameter t', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Arc length s(t)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Arc Length Along Helix', fontsize=14)  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

# Mark complete rotations
for n in range(1, 5):
    t_mark = n * 2*np.pi
    if t_mark <= t_max:
        s_mark = np.sqrt(R**2 + h**2) * t_mark  # np.sqrt() computes square root
        plt.plot(t_mark, s_mark, 'o', color=COLORS['orange'], markersize=8)  # plt.plot() draws line plot
        plt.text(t_mark, s_mark, f'  {n} turn{"s" if n > 1 else ""}',  # plt.text() adds text annotation at specified coordinates
                fontsize=10, verticalalignment='bottom')

plt.show()  # plt.show() displays the figure window
```

**What you should see:** A linear relationship s(t) = (constant) × t, because the helix has constant speed.

---

### 🎯 Practice Question #2

**Q:** A particle moves along r(t) = (3t, 4t, 0) from t = 0 to t = 1. What is the arc length?

<details>
<summary>💡 Hint 1</summary>

First find the velocity vector r'(t), then compute its magnitude |r'(t)|.
</details>

<details>
<summary>💡 Hint 2</summary>

If speed is constant, then L = speed × time = |r'(t)| × (t_end - t_start)
</details>

<details>
<summary>✅ Answer</summary>

r'(t) = (3, 4, 0)

Speed = |r'(t)| = √(3² + 4² + 0²) = √25 = 5

Arc length L = 5 × (1 - 0) = **5**

```python
def r(t):
    return np.array([3*t, 4*t, 0])  # np.array() converts Python list/tuple to efficient numpy array

def v(t):
    return np.array([3, 4, 0])  # np.array() converts Python list/tuple to efficient numpy array

speed = np.linalg.norm(v(0))  # Constant speed
arc_length = speed * 1  # From t=0 to t=1

print(f"Speed: {speed}")
print(f"Arc length: {arc_length}")
# Speed: 5.0
# Arc length: 5.0
```
</details>

---

## 3. Curvature and Osculating Circle

### 📖 Concept

**Curvature** κ (kappa) measures how sharply a curve bends. It's the reciprocal of the radius of the best-fitting circle:

```
κ = 1/R
```

where R is the radius of the **osculating circle** (the circle that "kisses" the curve at that point).

**Formulas for curvature:**

1. **General formula:**
```
κ(t) = |r'(t) × r''(t)| / |r'(t)|³
```

2. **For arc-length parametrization** (where |r'(s)| = 1):
```
κ(s) = |r''(s)|
```

**Physical interpretation:**
- κ = 0: Straight line (no bending)
- κ = constant: Circle of radius R = 1/κ
- κ large: Sharp turn (small radius)
- κ small: Gentle turn (large radius)

**Why this matters for GR:** Curvature of spacetime is described by the Riemann curvature tensor. Understanding curvature of curves prepares us for understanding curvature of 4D spacetime manifolds.

---

### 💻 Code Example: Computing Curvature

```python
# Circle of radius 2
def circle(t):
    R = 2.0
    return np.array([R*np.cos(t), R*np.sin(t), 0])  # np.array() converts Python list/tuple to efficient numpy array

def circle_velocity(t):
    R = 2.0
    return np.array([-R*np.sin(t), R*np.cos(t), 0])  # np.array() converts Python list/tuple to efficient numpy array

def circle_acceleration(t):
    R = 2.0
    return np.array([-R*np.cos(t), -R*np.sin(t), 0])  # np.array() converts Python list/tuple to efficient numpy array

# Compute curvature at t = 0
t = 0
r_prime = circle_velocity(t)
r_double_prime = circle_acceleration(t)

# κ = |r' × r''| / |r'|³
cross = np.cross(r_prime, r_double_prime)  # np.cross() computes cross product (3D vectors only)
curvature = np.linalg.norm(cross) / np.linalg.norm(r_prime)**3  # np.linalg.norm() computes vector magnitude (Euclidean norm)

print(f"Circle radius R = 2.0")
print(f"Velocity r'(0) = {r_prime}")
print(f"Acceleration r''(0) = {r_double_prime}")
print(f"Curvature κ = {curvature:.4f}")
print(f"Expected κ = 1/R = {1/2.0}")
print(f"Osculating circle radius = 1/κ = {1/curvature:.4f}")
```

**Expected output:**
```
Circle radius R = 2.0
Velocity r'(0) = [ 0.  2.  0.]
Acceleration r''(0) = [-2.  0.  0.]
Curvature κ = 0.5000
Expected κ = 1/R = 0.5
Osculating circle radius = 1/κ = 2.0000
```

---

### 📊 Visualization: Curvature and Osculating Circles

```python
# Create a curve with varying curvature
def wavy_curve(t):
    x = t
    y = np.sin(2*t)  # np.sin() computes sine (element-wise for arrays)
    z = 0
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

def wavy_velocity(t):
    x = 1
    y = 2*np.cos(2*t)  # np.cos() computes cosine (element-wise for arrays)
    z = 0
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

def wavy_acceleration(t):
    x = 0
    y = -4*np.sin(2*t)  # np.sin() computes sine (element-wise for arrays)
    z = 0
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

# Function to compute curvature
def compute_curvature(t):
    r_prime = wavy_velocity(t)
    r_double_prime = wavy_acceleration(t)
    cross = np.cross(r_prime, r_double_prime)  # np.cross() computes cross product (3D vectors only)
    return np.linalg.norm(cross) / np.linalg.norm(r_prime)**3  # np.linalg.norm() computes vector magnitude (Euclidean norm)

# Plot the curve
t_vals = np.linspace(0, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
curve_pts = np.array([wavy_curve(t) for t in t_vals])  # np.array() converts Python list/tuple to efficient numpy array

fig = plt.figure(figsize=(14, 5))  # plt.figure() creates a new figure for plotting

# Left: The curve
ax1 = fig.add_subplot(121)
ax1.plot(curve_pts[:, 0], curve_pts[:, 1], color=COLORS['blue'], linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Curve: r(t) = (t, sin(2t), 0)')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Add osculating circles at peaks (max curvature points)
peak_times = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
for t_peak in peak_times:
    kappa = compute_curvature(t_peak)
    radius = 1 / kappa if kappa > 1e-6 else float('inf')
    center_pos = wavy_curve(t_peak)

    # Draw osculating circle
    theta_circle = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
    # Determine center offset (perpendicular to velocity)
    vel = wavy_velocity(t_peak)
    vel_norm = vel / np.linalg.norm(vel)  # np.linalg.norm() computes vector magnitude (Euclidean norm)
    perp = np.array([-vel_norm[1], vel_norm[0], 0])  # Rotate 90°

    # Check which direction (toward center of curvature)
    acc = wavy_acceleration(t_peak)
    if np.dot(perp, acc) < 0:  # np.dot() computes dot product of two arrays
        perp = -perp

    center = center_pos[:2] + perp[:2] * radius
    circle_x = center[0] + radius * np.cos(theta_circle)  # np.cos() computes cosine (element-wise for arrays)
    circle_y = center[1] + radius * np.sin(theta_circle)  # np.sin() computes sine (element-wise for arrays)

    ax1.plot(circle_x, circle_y, '--', color=COLORS['orange'],
            linewidth=1, alpha=0.6)
    ax1.plot(center_pos[0], center_pos[1], 'o', color=COLORS['orange'], markersize=6)

# Right: Curvature as function of t
ax2 = fig.add_subplot(122)
curvatures = [compute_curvature(t) for t in t_vals]
ax2.plot(t_vals, curvatures, color=COLORS['green'], linewidth=2)
ax2.set_xlabel('Parameter t')
ax2.set_ylabel('Curvature κ(t)')
ax2.set_title('Curvature Along Curve')
ax2.grid(True, alpha=0.3)

# Mark peaks
for t_peak in peak_times:
    kappa = compute_curvature(t_peak)
    ax2.plot(t_peak, kappa, 'o', color=COLORS['orange'], markersize=8)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

**What you should see:**
- **Left:** Orange dashed circles fitting snugly to the curve at points of maximum bending
- **Right:** Curvature peaks where the curve bends most sharply

---

### 🎯 Practice Question #3

**Q:** What is the curvature of a straight line?

<details>
<summary>💡 Hint</summary>

For a line r(t) = r₀ + tv, what is the acceleration r''(t)?
</details>

<details>
<summary>✅ Answer</summary>

**κ = 0** (zero curvature)

A straight line has r''(t) = 0 (zero acceleration), so the numerator |r' × r''| = 0, giving κ = 0.

This makes sense: a straight line doesn't bend, so its curvature is zero. The osculating circle has infinite radius.

```python
# Straight line
def line(t):
    return np.array([t, 2*t, 3*t])  # np.array() converts Python list/tuple to efficient numpy array

def line_vel(t):
    return np.array([1, 2, 3])  # np.array() converts Python list/tuple to efficient numpy array

def line_acc(t):
    return np.array([0, 0, 0])  # np.array() converts Python list/tuple to efficient numpy array

cross = np.cross(line_vel(0), line_acc(0))  # np.cross() computes cross product (3D vectors only)
kappa = np.linalg.norm(cross) / np.linalg.norm(line_vel(0))**3  # np.linalg.norm() computes vector magnitude (Euclidean norm)
print(f"Curvature of line: {kappa}")  # 0.0
```
</details>

---

## 4. Torsion and the Frenet-Serret Frame

### 📖 Concept

While curvature measures bending in a plane, **torsion** τ (tau) measures how much a curve twists out of that plane.

**The Frenet-Serret frame** consists of three orthonormal vectors at each point:

1. **T** = Tangent = r'(s) / |r'(s)| (direction of motion)
2. **N** = Normal = T'(s) / |T'(s)| (direction of turning)
3. **B** = Binormal = T × N (perpendicular to both)

These form a moving coordinate system that follows the curve.

**Frenet-Serret formulas** describe how the frame rotates:
```
T' = κN           (tangent turns toward normal, rate = curvature)
N' = -κT + τB     (normal rotates in plane and twists)
B' = -τN          (binormal twists, rate = torsion)
```

**Torsion formula:**
```
τ = (r' × r'') · r''' / |r' × r''|²
```

**Physical examples:**
- κ ≠ 0, τ = 0: Planar curve (like a circle)
- κ ≠ 0, τ ≠ 0: Twisted curve (like a helix)
- κ = 0: Straight line

**Why this matters for GR:** The Frenet-Serret frame is analogous to parallel transport of vectors along curves in curved spacetime. Torsion connects to the antisymmetric part of connections in differential geometry.

---

### 💻 Code Example: Frenet-Serret Frame for Helix

```python
# Helix has constant curvature and torsion
def helix(t):
    R = 1.0
    h = 0.3
    return np.array([R*np.cos(t), R*np.sin(t), h*t])  # np.array() converts Python list/tuple to efficient numpy array

def helix_vel(t):
    R = 1.0
    h = 0.3
    return np.array([-R*np.sin(t), R*np.cos(t), h])  # np.array() converts Python list/tuple to efficient numpy array

def helix_acc(t):
    R = 1.0
    return np.array([-R*np.cos(t), -R*np.sin(t), 0])  # np.array() converts Python list/tuple to efficient numpy array

def helix_jerk(t):
    R = 1.0
    return np.array([R*np.sin(t), -R*np.cos(t), 0])  # np.array() converts Python list/tuple to efficient numpy array

# Compute Frenet-Serret frame at t = 0
t = 0
r_p = helix_vel(t)
r_pp = helix_acc(t)
r_ppp = helix_jerk(t)

# Tangent vector
T = r_p / np.linalg.norm(r_p)  # np.linalg.norm() computes vector magnitude (Euclidean norm)

# Curvature
cross_1 = np.cross(r_p, r_pp)  # np.cross() computes cross product (3D vectors only)
kappa = np.linalg.norm(cross_1) / np.linalg.norm(r_p)**3  # np.linalg.norm() computes vector magnitude (Euclidean norm)

# Normal vector
T_prime = r_pp / np.linalg.norm(r_p)  # Approximation
N = T_prime / np.linalg.norm(T_prime) if np.linalg.norm(T_prime) > 1e-10 else T_prime  # np.linalg.norm() computes vector magnitude (Euclidean norm)

# Binormal vector
B = np.cross(T, N)  # np.cross() computes cross product (3D vectors only)
B = B / np.linalg.norm(B)  # np.linalg.norm() computes vector magnitude (Euclidean norm)

# Torsion
torsion = np.dot(cross_1, r_ppp) / np.linalg.norm(cross_1)**2  # np.dot() computes dot product of two arrays

print("Frenet-Serret Frame at t=0:")
print(f"Tangent T = {T}")
print(f"Normal N = {N}")
print(f"Binormal B = {B}")
print(f"\nCurvature κ = {kappa:.4f}")
print(f"Torsion τ = {torsion:.4f}")

# Verify orthonormality
print(f"\nOrthonormality check:")
print(f"T·T = {np.dot(T,T):.4f} (should be 1)")  # np.dot() computes dot product of two arrays
print(f"N·N = {np.dot(N,N):.4f} (should be 1)")  # np.dot() computes dot product of two arrays
print(f"B·B = {np.dot(B,B):.4f} (should be 1)")  # np.dot() computes dot product of two arrays
print(f"T·N = {np.dot(T,N):.4f} (should be 0)")  # np.dot() computes dot product of two arrays
print(f"T·B = {np.dot(T,B):.4f} (should be 0)")  # np.dot() computes dot product of two arrays
print(f"N·B = {np.dot(N,B):.4f} (should be 0)")  # np.dot() computes dot product of two arrays
```

---

### 📊 Visualization: Frenet-Serret Frame Along Helix

```python
fig = plt.figure(figsize=(12, 10))  # plt.figure() creates a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  projection='3d'  # Create 3D axes

# Draw the helix
t_vals = np.linspace(0, 4*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
curve_pts = np.array([helix(t) for t in t_vals])  # np.array() converts Python list/tuple to efficient numpy array
ax.plot(curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2],
        color=COLORS['blue'], linewidth=2, alpha=0.3, label='Helix')

# Draw Frenet-Serret frame at several points
def compute_frame(t):
    """Compute T, N, B at parameter t"""
    r_p = helix_vel(t)
    r_pp = helix_acc(t)

    T = r_p / np.linalg.norm(r_p)  # np.linalg.norm() computes vector magnitude (Euclidean norm)

    cross_1 = np.cross(r_p, r_pp)  # np.cross() computes cross product (3D vectors only)
    kappa = np.linalg.norm(cross_1) / np.linalg.norm(r_p)**3  # np.linalg.norm() computes vector magnitude (Euclidean norm)

    # Compute N from acceleration
    acc_perp = r_pp - np.dot(r_pp, T) * T  # Remove tangent component
    N = acc_perp / np.linalg.norm(acc_perp) if np.linalg.norm(acc_perp) > 1e-10 else np.array([0,0,1])  # np.array() converts Python list/tuple to efficient numpy array

    B = np.cross(T, N)  # np.cross() computes cross product (3D vectors only)

    return T, N, B

sample_times = np.linspace(0, 4*np.pi, 8)  # np.linspace() creates evenly spaced array between start and end
for t in sample_times:
    pos = helix(t)
    T, N, B = compute_frame(t)

    scale = 0.5  # Scale for visibility

    # Draw frame vectors
    ax.quiver(pos[0], pos[1], pos[2], T[0]*scale, T[1]*scale, T[2]*scale,
             color=COLORS['green'], arrow_length_ratio=0.3, linewidth=2)
    ax.quiver(pos[0], pos[1], pos[2], N[0]*scale, N[1]*scale, N[2]*scale,
             color=COLORS['red'], arrow_length_ratio=0.3, linewidth=2)
    ax.quiver(pos[0], pos[1], pos[2], B[0]*scale, B[1]*scale, B[2]*scale,
             color=COLORS['purple'], arrow_length_ratio=0.3, linewidth=2)

# Add legend manually
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLORS['blue'], linewidth=2, label='Helix'),
    Line2D([0], [0], color=COLORS['green'], linewidth=2, label='Tangent T'),
    Line2D([0], [0], color=COLORS['red'], linewidth=2, label='Normal N'),
    Line2D([0], [0], color=COLORS['purple'], linewidth=2, label='Binormal B')
]
ax.legend(handles=legend_elements, loc='upper right')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Frenet-Serret Frame Along Helix')
plt.show()  # plt.show() displays the figure window
```

**What you should see:** Green (tangent), red (normal), and purple (binormal) vectors forming orthogonal triads that rotate smoothly along the helix.

---

### 🎯 Practice Question #4

**Q:** For a planar curve (lying entirely in the xy-plane), what is the torsion?

<details>
<summary>💡 Hint</summary>

If the curve is planar, does it twist out of the plane?
</details>

<details>
<summary>✅ Answer</summary>

**τ = 0** (zero torsion)

A planar curve never leaves its plane, so it has no twisting. The binormal vector B stays constant, pointing perpendicular to the plane.

For a curve in the xy-plane, r'''(t) has no z-component contribution that would create torsion, so (r' × r'') · r''' in the plane gives zero.

```python
# Circle in xy-plane
def circle_planar(t):
    return np.array([np.cos(t), np.sin(t), 0])  # np.array() converts Python list/tuple to efficient numpy array

# All derivatives stay in xy-plane
# The z-component is always 0, so torsion = 0
```
</details>

---

## 5. Parametric Surfaces

### 📖 Concept

A **parametric surface** is a 2D surface in 3D space, parametrized by two variables u and v:

```
r(u, v) = (x(u, v), y(u, v), z(u, v))
```

**Common examples:**
- **Plane:** r(u, v) = r₀ + u·a + v·b (point + two directions)
- **Sphere:** r(u, v) = (R sin u cos v, R sin u sin v, R cos u)
- **Cylinder:** r(u, v) = (R cos v, R sin v, u)
- **Torus:** r(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u)

**Tangent vectors:** Partial derivatives give vectors tangent to the surface:
- r_u = ∂r/∂u (tangent in u-direction)
- r_v = ∂r/∂v (tangent in v-direction)

**Normal vector:** Cross product of tangent vectors:
```
N = r_u × r_v  (perpendicular to surface)
```

**Why this matters for GR:** Spacetime is a 4D surface (manifold). Understanding 2D surfaces in 3D prepares us for understanding the intrinsic and extrinsic geometry of higher-dimensional curved spaces.

---

### 💻 Code Example: Defining a Sphere

```python
# Parametric sphere
def sphere(u, v, R=1.0):
    """
    Sphere of radius R
    u: polar angle (0 to π)
    v: azimuthal angle (0 to 2π)
    """
    x = R * np.sin(u) * np.cos(v)  # np.sin() computes sine (element-wise for arrays)
    y = R * np.sin(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    z = R * np.cos(u)  # np.cos() computes cosine (element-wise for arrays)
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

# Create mesh
u_vals = np.linspace(0, np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
v_vals = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
U, V = np.meshgrid(u_vals, v_vals)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# Evaluate surface
X = np.sin(U) * np.cos(V)  # np.sin() computes sine (element-wise for arrays)
Y = np.sin(U) * np.sin(V)  # np.sin() computes sine (element-wise for arrays)
Z = np.cos(U)  # np.cos() computes cosine (element-wise for arrays)

print(f"Surface parametrized by u ∈ [0, π], v ∈ [0, 2π]")
print(f"Sample point at u=π/2, v=0: {sphere(np.pi/2, 0)}")
print(f"Sample point at u=π/2, v=π/2: {sphere(np.pi/2, np.pi/2)}")
```

---

### 📊 Visualization: Sphere with Tangent Vectors

```python
fig = plt.figure(figsize=(14, 6))  # plt.figure() creates a new figure for plotting

# Left: The sphere
ax1 = fig.add_subplot(121, projection='3d')  projection='3d'  # Create 3D axes
ax1.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', edgecolor='none')  # .plot_surface() draws 3D surface plot
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Unit Sphere: r(u,v)')

# Right: Sphere with tangent vectors
ax2 = fig.add_subplot(122, projection='3d')  projection='3d'  # Create 3D axes
ax2.plot_surface(X, Y, Z, alpha=0.3, color=COLORS['cyan'], edgecolor='none')  # .plot_surface() draws 3D surface plot

# Compute tangent vectors at a point
def sphere_r_u(u, v, R=1.0):
    """Partial derivative with respect to u"""
    x = R * np.cos(u) * np.cos(v)  # np.cos() computes cosine (element-wise for arrays)
    y = R * np.cos(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    z = -R * np.sin(u)  # np.sin() computes sine (element-wise for arrays)
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

def sphere_r_v(u, v, R=1.0):
    """Partial derivative with respect to v"""
    x = -R * np.sin(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    y = R * np.sin(u) * np.cos(v)  # np.sin() computes sine (element-wise for arrays)
    z = 0
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

# Show tangent vectors at several points
u_samples = [np.pi/4, np.pi/2, 3*np.pi/4]
v_samples = [0, np.pi/2, np.pi, 3*np.pi/2]

for u in u_samples:
    for v in v_samples[::2]:  # Skip some to avoid clutter
        pos = sphere(u, v)
        r_u = sphere_r_u(u, v) * 0.3  # Scale for visibility
        r_v = sphere_r_v(u, v) * 0.3

        ax2.quiver(pos[0], pos[1], pos[2], r_u[0], r_u[1], r_u[2],
                  color=COLORS['orange'], arrow_length_ratio=0.2, linewidth=2)
        ax2.quiver(pos[0], pos[1], pos[2], r_v[0], r_v[1], r_v[2],
                  color=COLORS['green'], arrow_length_ratio=0.2, linewidth=2)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Tangent Vectors r_u (orange) and r_v (green)')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🔬 Explore: Other Surfaces

**1. Torus (donut shape):**
```python
def torus(u, v, R=2.0, r=0.5):
    """
    R: major radius (center to tube center)
    r: minor radius (tube radius)
    """
    x = (R + r*np.cos(u)) * np.cos(v)  # np.cos() computes cosine (element-wise for arrays)
    y = (R + r*np.cos(u)) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    z = r * np.sin(u)  # np.sin() computes sine (element-wise for arrays)
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

u_vals = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
v_vals = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
U, V = np.meshgrid(u_vals, v_vals)  # np.meshgrid() creates coordinate matrices from coordinate vectors

X = (2 + 0.5*np.cos(U)) * np.cos(V)  # np.cos() computes cosine (element-wise for arrays)
Y = (2 + 0.5*np.cos(U)) * np.sin(V)  # np.sin() computes sine (element-wise for arrays)
Z = 0.5 * np.sin(U)  # np.sin() computes sine (element-wise for arrays)

fig = plt.figure(figsize=(10, 8))  # plt.figure() creates a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  projection='3d'  # Create 3D axes
ax.plot_surface(X, Y, Z, alpha=0.8, cmap='plasma')  # .plot_surface() draws 3D surface plot
ax.set_title('Torus')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()  # plt.show() displays the figure window
```

**2. Hyperboloid of one sheet:**
```python
u_vals = np.linspace(-2, 2, 30)  # np.linspace() creates evenly spaced array between start and end
v_vals = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
U, V = np.meshgrid(u_vals, v_vals)  # np.meshgrid() creates coordinate matrices from coordinate vectors

a, b, c = 1, 1, 1
X = a * np.sqrt(1 + U**2) * np.cos(V)  # np.cos() computes cosine (element-wise for arrays)
Y = b * np.sqrt(1 + U**2) * np.sin(V)  # np.sin() computes sine (element-wise for arrays)
Z = c * U

fig = plt.figure(figsize=(10, 8))  # plt.figure() creates a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  projection='3d'  # Create 3D axes
ax.plot_surface(X, Y, Z, alpha=0.8, cmap='coolwarm')  # .plot_surface() draws 3D surface plot
ax.set_title('Hyperboloid of One Sheet')
plt.show()  # plt.show() displays the figure window
```

---

## 6. First Fundamental Form

### 📖 Concept

The **first fundamental form** encodes the intrinsic geometry of a surface - distances and angles measured along the surface itself, without reference to the surrounding 3D space.

It's a matrix of dot products of tangent vectors:

```
I = [ E  F ]    where  E = r_u · r_u
    [ F  G ]           F = r_u · r_v
                       G = r_v · r_v
```

**What it tells us:**
- **E, G:** Measure how much the surface stretches in u and v directions
- **F:** Measures skewness (0 if coordinate curves are perpendicular)
- Determinant det(I) = EG - F² relates to area element

**Arc length on surface:**
```
ds² = E du² + 2F du dv + G dv²
```

This is the **metric** of the surface!

**Why this matters for GR:** The first fundamental form IS the metric tensor g_μν for a 2D surface. In GR, the spacetime metric g_μν is a 4×4 version of this, encoding all geometric information about spacetime.

---

### 💻 Code Example: First Fundamental Form of Sphere

```python
# Sphere tangent vectors
def sphere_r_u(u, v, R=1.0):
    x = R * np.cos(u) * np.cos(v)  # np.cos() computes cosine (element-wise for arrays)
    y = R * np.cos(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    z = -R * np.sin(u)  # np.sin() computes sine (element-wise for arrays)
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

def sphere_r_v(u, v, R=1.0):
    x = -R * np.sin(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    y = R * np.sin(u) * np.cos(v)  # np.sin() computes sine (element-wise for arrays)
    z = 0
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

# Compute first fundamental form at u = π/4, v = π/6
u, v = np.pi/4, np.pi/6
R = 1.0

r_u = sphere_r_u(u, v, R)
r_v = sphere_r_v(u, v, R)

E = np.dot(r_u, r_u)  # np.dot() computes dot product of two arrays
F = np.dot(r_u, r_v)  # np.dot() computes dot product of two arrays
G = np.dot(r_v, r_v)  # np.dot() computes dot product of two arrays

print(f"Point on sphere at u={u:.3f}, v={v:.3f}")
print(f"r_u = {r_u}")
print(f"r_v = {r_v}")
print(f"\nFirst Fundamental Form:")
print(f"E = r_u · r_u = {E:.4f}")
print(f"F = r_u · r_v = {F:.4f}")
print(f"G = r_v · r_v = {G:.4f}")
print(f"\nMetric tensor:")
print(f"I = [{E:.4f}  {F:.4f}]")
print(f"    [{F:.4f}  {G:.4f}]")
print(f"\ndet(I) = EG - F² = {E*G - F**2:.4f}")

# For sphere: E = R², F = 0, G = R²sin²(u)
print(f"\nExpected for sphere at u={u:.3f}:")
print(f"E = R² = {R**2:.4f}")
print(f"F = 0")
print(f"G = R²sin²(u) = {R**2 * np.sin(u)**2:.4f}")  # np.sin() computes sine (element-wise for arrays)
```

---

### 📊 Visualization: Metric Components on Sphere

```python
# Compute E, F, G across the sphere surface
u_vals = np.linspace(0.1, np.pi-0.1, 30)  # np.linspace() creates evenly spaced array between start and end
v_vals = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
U_grid, V_grid = np.meshgrid(u_vals, v_vals)  # np.meshgrid() creates coordinate matrices from coordinate vectors

E_vals = np.ones_like(U_grid)  # E = 1 for unit sphere
G_vals = np.sin(U_grid)**2      # G = sin²(u)

fig = plt.figure(figsize=(14, 5))  # plt.figure() creates a new figure for plotting

# Left: E component (constant for sphere)
ax1 = fig.add_subplot(121, projection='3d')  projection='3d'  # Create 3D axes
surf1 = ax1.plot_surface(U_grid, V_grid, E_vals, cmap='viridis', alpha=0.8)  # .plot_surface() draws 3D surface plot
ax1.set_xlabel('u (polar)')
ax1.set_ylabel('v (azimuthal)')
ax1.set_zlabel('E')
ax1.set_title('E = r_u · r_u (constant = R² = 1)')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Right: G component (varies with latitude)
ax2 = fig.add_subplot(122, projection='3d')  projection='3d'  # Create 3D axes
surf2 = ax2.plot_surface(U_grid, V_grid, G_vals, cmap='plasma', alpha=0.8)  # .plot_surface() draws 3D surface plot
ax2.set_xlabel('u (polar)')
ax2.set_ylabel('v (azimuthal)')
ax2.set_zlabel('G')
ax2.set_title('G = r_v · r_v = R²sin²(u)')
fig.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Notice: G → 0 at poles (u=0, π) where circles of latitude shrink to points")
```

---

### 🎯 Practice Question #5

**Q:** For a flat plane parametrized by r(u, v) = (u, v, 0), what is the first fundamental form?

<details>
<summary>💡 Hint</summary>

Compute r_u and r_v, then find E = r_u · r_u, F = r_u · r_v, G = r_v · r_v.
</details>

<details>
<summary>✅ Answer</summary>

r_u = ∂r/∂u = (1, 0, 0)
r_v = ∂r/∂v = (0, 1, 0)

E = r_u · r_u = 1
F = r_u · r_v = 0
G = r_v · r_v = 1

First fundamental form: **I = [1  0]**
                             **[0  1]**

This is the identity matrix, which means ds² = du² + dv² - the standard Euclidean metric! A flat plane has the simplest possible geometry.

```python
r_u = np.array([1, 0, 0])  # np.array() converts Python list/tuple to efficient numpy array
r_v = np.array([0, 1, 0])  # np.array() converts Python list/tuple to efficient numpy array

E = np.dot(r_u, r_u)  # np.dot() computes dot product of two arrays
F = np.dot(r_u, r_v)  # np.dot() computes dot product of two arrays
G = np.dot(r_v, r_v)  # np.dot() computes dot product of two arrays

print(f"E = {E}, F = {F}, G = {G}")
# E = 1, F = 0, G = 1
```
</details>

---

## 7. Second Fundamental Form and Gaussian Curvature

### 📖 Concept

While the first fundamental form describes intrinsic geometry (measurable on the surface), the **second fundamental form** describes extrinsic geometry - how the surface curves in the surrounding 3D space.

The second fundamental form is:

```
II = [ L  M ]    where  L = r_uu · n
     [ M  N ]           M = r_uv · n
                        N = r_vv · n
```

where n = (r_u × r_v)/|r_u × r_v| is the unit normal vector.

**Gaussian curvature** K measures the intrinsic curvature:

```
K = (LN - M²) / (EG - F²) = det(II) / det(I)
```

**Key values:**
- K > 0: Elliptic (sphere-like, bowl-shaped)
- K = 0: Parabolic (flat, or cylindrical)
- K < 0: Hyperbolic (saddle-shaped)

**Gauss's Theorema Egregium:** K can be computed entirely from the first fundamental form! This means Gaussian curvature is intrinsic - a 2D being living on the surface could measure it without knowing about the 3D embedding.

**Why this matters for GR:** Spacetime curvature is intrinsic. The Riemann curvature tensor generalizes Gaussian curvature to higher dimensions. This is how gravity works - matter curves spacetime, and this curvature is measurable from within spacetime itself.

---

### 💻 Code Example: Gaussian Curvature of Sphere

```python
# For a sphere, let's compute K numerically
def sphere(u, v, R=1.0):
    x = R * np.sin(u) * np.cos(v)  # np.sin() computes sine (element-wise for arrays)
    y = R * np.sin(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    z = R * np.cos(u)  # np.cos() computes cosine (element-wise for arrays)
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

def sphere_r_u(u, v, R=1.0):
    x = R * np.cos(u) * np.cos(v)  # np.cos() computes cosine (element-wise for arrays)
    y = R * np.cos(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    z = -R * np.sin(u)  # np.sin() computes sine (element-wise for arrays)
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

def sphere_r_v(u, v, R=1.0):
    x = -R * np.sin(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    y = R * np.sin(u) * np.cos(v)  # np.sin() computes sine (element-wise for arrays)
    z = 0
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

def sphere_r_uu(u, v, R=1.0):
    x = -R * np.sin(u) * np.cos(v)  # np.sin() computes sine (element-wise for arrays)
    y = -R * np.sin(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    z = -R * np.cos(u)  # np.cos() computes cosine (element-wise for arrays)
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

def sphere_r_uv(u, v, R=1.0):
    x = -R * np.cos(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    y = R * np.cos(u) * np.cos(v)  # np.cos() computes cosine (element-wise for arrays)
    z = 0
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

def sphere_r_vv(u, v, R=1.0):
    x = -R * np.sin(u) * np.cos(v)  # np.sin() computes sine (element-wise for arrays)
    y = -R * np.sin(u) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    z = 0
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

# Compute at a point
u, v = np.pi/3, np.pi/4
R = 2.0

r_u = sphere_r_u(u, v, R)
r_v = sphere_r_v(u, v, R)
r_uu = sphere_r_uu(u, v, R)
r_uv = sphere_r_uv(u, v, R)
r_vv = sphere_r_vv(u, v, R)

# Normal vector
normal_vec = np.cross(r_u, r_v)  # np.cross() computes cross product (3D vectors only)
n = normal_vec / np.linalg.norm(normal_vec)  # np.linalg.norm() computes vector magnitude (Euclidean norm)

# First fundamental form
E = np.dot(r_u, r_u)  # np.dot() computes dot product of two arrays
F = np.dot(r_u, r_v)  # np.dot() computes dot product of two arrays
G = np.dot(r_v, r_v)  # np.dot() computes dot product of two arrays

# Second fundamental form
L = np.dot(r_uu, n)  # np.dot() computes dot product of two arrays
M = np.dot(r_uv, n)  # np.dot() computes dot product of two arrays
N = np.dot(r_vv, n)  # np.dot() computes dot product of two arrays

# Gaussian curvature
K = (L*N - M**2) / (E*G - F**2)

print(f"Sphere of radius R = {R}")
print(f"Point at u={u:.3f}, v={v:.3f}")
print(f"\nFirst fundamental form:")
print(f"I = [{E:.4f}  {F:.4f}]")
print(f"    [{F:.4f}  {G:.4f}]")
print(f"\nSecond fundamental form:")
print(f"II = [{L:.4f}  {M:.4f}]")
print(f"     [{M:.4f}  {N:.4f}]")
print(f"\nGaussian curvature K = {K:.6f}")
print(f"Expected K = 1/R² = {1/R**2:.6f}")
print(f"Match: {np.isclose(K, 1/R**2)}")  # np.isclose() tests if values are approximately equal (handles floating point)
```

**Expected output:**
```
Sphere of radius R = 2.0
Gaussian curvature K = 0.250000
Expected K = 1/R² = 0.250000
Match: True
```

---

### 📊 Visualization: Surfaces with Different Curvatures

```python
fig = plt.figure(figsize=(16, 5))  # plt.figure() creates a new figure for plotting

# Sphere (K > 0: positive curvature)
ax1 = fig.add_subplot(131, projection='3d')  projection='3d'  # Create 3D axes
u = np.linspace(0, np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
v = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
U, V = np.meshgrid(u, v)  # np.meshgrid() creates coordinate matrices from coordinate vectors
X1 = np.sin(U) * np.cos(V)  # np.sin() computes sine (element-wise for arrays)
Y1 = np.sin(U) * np.sin(V)  # np.sin() computes sine (element-wise for arrays)
Z1 = np.cos(U)  # np.cos() computes cosine (element-wise for arrays)
ax1.plot_surface(X1, Y1, Z1, alpha=0.8, cmap='viridis')  # .plot_surface() draws 3D surface plot
ax1.set_title('Sphere\nK = 1/R² > 0\n(Elliptic)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Cylinder (K = 0: zero curvature)
ax2 = fig.add_subplot(132, projection='3d')  projection='3d'  # Create 3D axes
u = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
v = np.linspace(-2, 2, 30)  # np.linspace() creates evenly spaced array between start and end
U, V = np.meshgrid(u, v)  # np.meshgrid() creates coordinate matrices from coordinate vectors
X2 = np.cos(U)  # np.cos() computes cosine (element-wise for arrays)
Y2 = np.sin(U)  # np.sin() computes sine (element-wise for arrays)
Z2 = V
ax2.plot_surface(X2, Y2, Z2, alpha=0.8, cmap='plasma')  # .plot_surface() draws 3D surface plot
ax2.set_title('Cylinder\nK = 0\n(Parabolic)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# Saddle (K < 0: negative curvature)
ax3 = fig.add_subplot(133, projection='3d')  projection='3d'  # Create 3D axes
u = np.linspace(-2, 2, 30)  # np.linspace() creates evenly spaced array between start and end
v = np.linspace(-2, 2, 30)  # np.linspace() creates evenly spaced array between start and end
U, V = np.meshgrid(u, v)  # np.meshgrid() creates coordinate matrices from coordinate vectors
Z3 = U**2 - V**2  # Hyperbolic paraboloid
ax3.plot_surface(U, V, Z3, alpha=0.8, cmap='coolwarm')  # .plot_surface() draws 3D surface plot
ax3.set_title('Saddle\nK < 0\n(Hyperbolic)')
ax3.set_xlabel('u')
ax3.set_ylabel('v')
ax3.set_zlabel('z')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🔬 Explore: Mean Curvature

Besides Gaussian curvature K, there's also **mean curvature** H:

```python
# Mean curvature H = (EN - 2FM + GL) / (2(EG - F²))
# For sphere:
u, v = np.pi/3, np.pi/4
R = 2.0

# Use same E, F, G, L, M, N from previous computation
r_u = sphere_r_u(u, v, R)
r_v = sphere_r_v(u, v, R)
r_uu = sphere_r_uu(u, v, R)
r_uv = sphere_r_uv(u, v, R)
r_vv = sphere_r_vv(u, v, R)

normal_vec = np.cross(r_u, r_v)  # np.cross() computes cross product (3D vectors only)
n = normal_vec / np.linalg.norm(normal_vec)  # np.linalg.norm() computes vector magnitude (Euclidean norm)

E = np.dot(r_u, r_u)  # np.dot() computes dot product of two arrays
F = np.dot(r_u, r_v)  # np.dot() computes dot product of two arrays
G = np.dot(r_v, r_v)  # np.dot() computes dot product of two arrays
L = np.dot(r_uu, n)  # np.dot() computes dot product of two arrays
M = np.dot(r_uv, n)  # np.dot() computes dot product of two arrays
N = np.dot(r_vv, n)  # np.dot() computes dot product of two arrays

H = (E*N - 2*F*M + G*L) / (2*(E*G - F**2))

print(f"Mean curvature H = {H:.6f}")
print(f"Expected H = 1/R = {1/R:.6f}")
print(f"Gaussian curvature K = {(L*N - M**2)/(E*G - F**2):.6f}")
print(f"Expected K = 1/R² = {1/R**2:.6f}")
```

**Note:** For a sphere, both principal curvatures equal 1/R, so:
- Mean curvature H = 1/R
- Gaussian curvature K = 1/R²

---

### 🎯 Practice Question #6

**Q:** What is the Gaussian curvature of a flat plane?

<details>
<summary>💡 Hint</summary>

A plane doesn't curve at all. What should K be?
</details>

<details>
<summary>✅ Answer</summary>

**K = 0** (zero curvature)

A flat plane has no curvature. All second derivatives r_uu, r_uv, r_vv are zero, so L = M = N = 0, giving K = 0.

This makes sense geometrically: you can roll a flat sheet without stretching it (like making a cylinder), but you can't do that with a sphere. Zero Gaussian curvature means "locally flat."

In GR: Flat Minkowski spacetime has zero Riemann curvature tensor, just like a plane has zero Gaussian curvature.
</details>

---

## Practice Questions

After working through all sections, test your understanding:

### Curves

1. **Parametric curves:** Define a curve r(t) = (cos(2t), sin(2t), t) and compute r'(t) and |r'(t)|.

2. **Arc length:** Find the arc length of r(t) = (3t, 4t, 0) from t = 0 to t = 5.

3. **Curvature:** What is the curvature of a circle of radius 3?

4. **Torsion:** Show that the curve r(t) = (cos t, sin t, 0) has zero torsion.

### Surfaces

5. **Parametric surface:** Write parametric equations for a cylinder of radius 2 along the z-axis.

6. **First fundamental form:** For r(u,v) = (u, v, u² + v²), compute E, F, G at (0, 0).

7. **Gaussian curvature:** What sign of K corresponds to saddle-shaped surfaces?

8. **Theorema Egregium:** Why is Gaussian curvature called "intrinsic"?

---

## Summary and Next Steps

**Key Concepts Mastered:**
- Parametric curves and velocity vectors
- Arc length and geometric distance
- Curvature (bending) and torsion (twisting)
- Frenet-Serret frame
- Parametric surfaces and tangent spaces
- First fundamental form (metric)
- Second fundamental form and Gaussian curvature

**Connection to GR:**
These concepts generalize to 4D spacetime:
- Curves → Worldlines of particles
- Arc length → Proper time
- First fundamental form → Metric tensor g_μν
- Gaussian curvature → Riemann curvature tensor R_μνρσ

**Ready for next steps:**
- Lesson 5: Manifolds (abstract coordinate-free geometry)
- Lesson 6: Tensors (the language of GR)
- Lesson 7: Riemannian geometry (curvature in n dimensions)

---

**Continue to:** → [Lesson 5: Manifolds](../05_manifolds/LESSON.md)
