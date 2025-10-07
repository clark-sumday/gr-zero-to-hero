# Lesson 7: Riemannian Geometry

**Topics:** Metric Tensors, Christoffel Symbols, Covariant Derivatives, Riemann Curvature, Ricci Tensor, Geodesics
**Prerequisites:** Lessons 1-6 (Linear Algebra, Calculus, Tensors, Manifolds)
**Time:** ~6-8 hours

---

## Table of Contents

1. [The Metric Tensor](#1-the-metric-tensor)
2. [Distances and Angles in Curved Spaces](#2-distances-and-angles-in-curved-spaces)
3. [Christoffel Symbols and Connections](#3-christoffel-symbols-and-connections)
4. [Covariant Derivatives](#4-covariant-derivatives)
5. [The Riemann Curvature Tensor](#5-the-riemann-curvature-tensor)
6. [Ricci Tensor and Scalar Curvature](#6-ricci-tensor-and-scalar-curvature)
7. [Geodesics: Straight Lines in Curved Space](#7-geodesics-straight-lines-in-curved-space)
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

## 1. The Metric Tensor

### 📖 Concept

The **metric tensor** g_μν is the fundamental object in Riemannian geometry. It tells you:
- How to measure distances between nearby points
- How to compute angles between vectors
- The entire geometric structure of your space

In flat Euclidean space (ℝⁿ), the metric is just the identity matrix. But on curved manifolds like spheres, hyperboloids, or **spacetime in General Relativity**, the metric encodes all the curvature information.

**Definition:** For a smooth manifold M with coordinates x^μ, the metric tensor is a symmetric (0,2)-tensor field:
```
ds² = g_μν dx^μ dx^ν
```

This gives the **infinitesimal distance** (line element) between nearby points.

**Key Properties:**
- **Symmetric:** g_μν = g_νμ
- **Non-degenerate:** det(g) ≠ 0 (so g has an inverse g^μν)
- **Signature:** In GR, signature is (-,+,+,+) or (+,-,-,-)
- **Coordinate-dependent:** Components change under coordinate transformations

**Why this matters for GR:** The metric tensor **IS** spacetime geometry. Einstein's field equations tell us how matter curves the metric, which then tells matter how to move!

---

### 💻 Code Example: Euclidean Metric in Different Coordinates

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import sympy as sp  # SymPy for symbolic mathematics
from sympy import  # Import symbolic math functions

# Set up symbolic coordinates
x, y, r, theta = symbols('x y r theta', real=True, positive=True)  # symbols() creates symbolic variables

# Euclidean metric in Cartesian coordinates
g_cartesian = Matrix([  # Matrix() creates symbolic matrix
    [1, 0],
    [0, 1]
])

print("Euclidean metric in Cartesian (x,y) coordinates:")
print(g_cartesian)
print(f"\nLine element: ds² = dx² + dy²")

# Euclidean metric in polar coordinates
# x = r cos(θ), y = r sin(θ)
g_polar = Matrix([  # Matrix() creates symbolic matrix
    [1, 0],
    [0, r**2]
])

print("\nEuclidean metric in polar (r,θ) coordinates:")
print(g_polar)
print(f"\nLine element: ds² = dr² + r² dθ²")
print("\nNotice: Same geometry, different coordinate representation!")
```

**Expected output:**
```
Euclidean metric in Cartesian (x,y) coordinates:
Matrix([[1, 0], [0, 1]])

Line element: ds² = dx² + dy²

Euclidean metric in polar (r,θ) coordinates:
Matrix([[1, 0], [0, r**2]])

Line element: ds² = dr² + r² dθ²

Notice: Same geometry, different coordinate representation!
```

---

### 📊 Visualization: Metric on a Sphere

The 2-sphere has intrinsic curvature. Let's visualize its metric:

```python
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Metric on 2-sphere: ds² = R²(dθ² + sin²θ dφ²)
# where θ is polar angle, φ is azimuthal angle, R is radius

R = 1.0  # Unit sphere

# Create sphere mesh
theta_vals = np.linspace(0, np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
phi_vals = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
Theta, Phi = np.meshgrid(theta_vals, phi_vals)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# Convert to Cartesian for visualization
X = R * np.sin(Theta) * np.cos(Phi)  # np.sin() computes sine (element-wise for arrays)
Y = R * np.sin(Theta) * np.sin(Phi)  # np.sin() computes sine (element-wise for arrays)
Z = R * np.cos(Theta)  # np.cos() computes cosine (element-wise for arrays)

fig = plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Left: The sphere
ax1 = fig.add_subplot(121, projection='3d')  projection='3d'  # Create 3D axes
ax1.plot_surface(X, Y, Z, alpha=0.3, color=COLORS['blue'])  # .plot_surface() draws 3D surface plot
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title("2-Sphere (S²)")

# Draw some geodesics (great circles)
t = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
# Equator
ax1.plot(R*np.cos(t), R*np.sin(t), 0*t, color=COLORS['orange'], linewidth=2, label='Geodesic 1')  # np.sin() computes sine (element-wise for arrays)
# Meridian
ax1.plot(R*np.sin(t), 0*t, R*np.cos(t), color=COLORS['green'], linewidth=2, label='Geodesic 2')  # np.sin() computes sine (element-wise for arrays)
ax1.legend()

# Right: Metric components
ax2 = fig.add_subplot(122)
theta_plot = np.linspace(0.1, np.pi-0.1, 100)  # np.linspace() creates evenly spaced array between start and end

g_theta_theta = R**2 * np.ones_like(theta_plot)
g_phi_phi = R**2 * np.sin(theta_plot)**2  # np.sin() computes sine (element-wise for arrays)

ax2.plot(theta_plot, g_theta_theta, color=COLORS['blue'], linewidth=2, label='$g_{θθ} = R²$')
ax2.plot(theta_plot, g_phi_phi, color=COLORS['orange'], linewidth=2, label='$g_{φφ} = R²sin²θ$')
ax2.set_xlabel('Polar angle θ')
ax2.set_ylabel('Metric component value')
ax2.set_title('Metric Components on 2-Sphere')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=0, color='k', linewidth=0.5)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

**What you should see:**
- **Left:** A sphere with two geodesics (great circles) - the "straight lines" on a sphere
- **Right:** How metric components vary with position (g_φφ = 0 at poles!)

---

### 🎯 Practice Question #1

**Q:** What is the line element for a 2-sphere of radius R in (θ, φ) coordinates?

<details>
<summary>💡 Hint 1</summary>

The metric tensor is g = diag(R², R²sin²θ).
</details>

<details>
<summary>💡 Hint 2</summary>

Use ds² = g_μν dx^μ dx^ν with coordinates (θ, φ).
</details>

<details>
<summary>✅ Answer</summary>

**ds² = R²dθ² + R²sin²θ dφ²**

This is the intrinsic geometry of a sphere - no reference to the embedding 3D space!

```python
import sympy as sp  # SymPy for symbolic mathematics
theta, phi, R = sp.symbols('theta phi R', real=True, positive=True)  # symbols() creates symbolic variables

g = sp.Matrix([  # Matrix() creates symbolic matrix
    [R**2, 0],
    [0, R**2 * sp.sin(theta)**2]
])

print("Metric tensor on 2-sphere:")
print(g)
```
</details>

---

## 2. Distances and Angles in Curved Spaces

### 📖 Concept

The metric tensor allows us to compute:

**1. Distance between nearby points:**
```
ds² = g_μν dx^μ dx^ν
```

**2. Length of a curve γ(t) from t=a to t=b:**
```
L = ∫ₐᵇ √(g_μν (dx^μ/dt)(dx^ν/dt)) dt
```

**3. Angle between two vectors u^μ and v^ν:**
```
cos(θ) = (g_μν u^μ v^ν) / (√(g_αβ u^α u^β) √(g_γδ v^γ v^δ))
```

**Physical interpretation:** The metric generalizes the Euclidean dot product to curved spaces. In flat space, g_μν = δ_μν (Kronecker delta). In curved space, the metric tells us how the geometry deviates from flat.

---

### 💻 Code Example: Computing Distances on a Sphere

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations

def distance_on_sphere(theta1, phi1, theta2, phi2, R=1.0):
    """
    Compute geodesic distance between two points on a sphere.
    Uses the spherical law of cosines.

    Parameters:
    theta1, phi1: First point (polar, azimuthal angles)
    theta2, phi2: Second point
    R: Sphere radius
    """
    # Convert to Cartesian
    x1 = R * np.sin(theta1) * np.cos(phi1)  # np.sin() computes sine (element-wise for arrays)
    y1 = R * np.sin(theta1) * np.sin(phi1)  # np.sin() computes sine (element-wise for arrays)
    z1 = R * np.cos(theta1)  # np.cos() computes cosine (element-wise for arrays)

    x2 = R * np.sin(theta2) * np.cos(phi2)  # np.sin() computes sine (element-wise for arrays)
    y2 = R * np.sin(theta2) * np.sin(phi2)  # np.sin() computes sine (element-wise for arrays)
    z2 = R * np.cos(theta2)  # np.cos() computes cosine (element-wise for arrays)

    # Dot product
    dot = x1*x2 + y1*y2 + z1*z2

    # Angle between position vectors
    angle = np.arccos(np.clip(dot / R**2, -1, 1))

    # Arc length = R * angle
    return R * angle

# Example: Distance from North pole to equator
theta1, phi1 = 0, 0           # North pole
theta2, phi2 = np.pi/2, 0     # Point on equator

d = distance_on_sphere(theta1, phi1, theta2, phi2, R=1.0)
print(f"Distance from North pole to equator: {d:.4f} radians")
print(f"Expected (quarter circle): {np.pi/2:.4f} radians")

# Example 2: Distance between two points on equator
theta1, phi1 = np.pi/2, 0
theta2, phi2 = np.pi/2, np.pi/4  # 45° apart

d = distance_on_sphere(theta1, phi1, theta2, phi2, R=1.0)
print(f"\nDistance between two equatorial points 45° apart: {d:.4f} radians")
print(f"Expected: {np.pi/4:.4f} radians")
```

---

### 📊 Visualization: Curved vs Flat Distance

```python
# Compare distances in flat space vs on sphere
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Points on equator
n_points = 50
phi_vals = np.linspace(0, np.pi, n_points)  # np.linspace() creates evenly spaced array between start and end
R = 1.0

# Flat Euclidean distance (straight line through 3D space)
flat_distances = []
# Curved geodesic distance (along sphere surface)
curved_distances = []

for phi in phi_vals:
    theta1, phi1 = np.pi/2, 0
    theta2, phi2 = np.pi/2, phi

    # Flat distance (3D Euclidean)
    x1, y1, z1 = R * np.sin(theta1) * np.cos(phi1), R * np.sin(theta1) * np.sin(phi1), R * np.cos(theta1)  # np.sin() computes sine (element-wise for arrays)
    x2, y2, z2 = R * np.sin(theta2) * np.cos(phi2), R * np.sin(theta2) * np.sin(phi2), R * np.cos(theta2)  # np.sin() computes sine (element-wise for arrays)
    flat_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)  # np.sqrt() computes square root
    flat_distances.append(flat_dist)

    # Curved distance (geodesic on sphere)
    curved_dist = distance_on_sphere(theta1, phi1, theta2, phi2, R)
    curved_distances.append(curved_dist)

# Left plot: Distances vs angle
ax1.plot(np.degrees(phi_vals), flat_distances, color=COLORS['blue'],
         linewidth=2, label='Euclidean (chord)')
ax1.plot(np.degrees(phi_vals), curved_distances, color=COLORS['orange'],
         linewidth=2, label='Geodesic (arc)')
ax1.set_xlabel('Angle separation (degrees)')
ax1.set_ylabel('Distance')
ax1.set_title('Flat vs Curved Distance')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: Ratio (shows curvature effect)
ratio = np.array(curved_distances) / np.array(flat_distances)  # np.array() converts Python list/tuple to efficient numpy array
ax2.plot(np.degrees(phi_vals), ratio, color=COLORS['green'], linewidth=2)
ax2.set_xlabel('Angle separation (degrees)')
ax2.set_ylabel('Geodesic / Euclidean')
ax2.set_title('Curvature Effect on Distance')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1, color=COLORS['red'], linestyle='--', label='Flat space ratio')
ax2.legend()

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🔬 Explore: Earth Distances

Try computing real distances on Earth (approximate as sphere with R = 6371 km):

```python
# City coordinates (latitude, longitude)
# Convert latitude to polar angle: θ = π/2 - latitude

def latlon_to_theta_phi(lat_deg, lon_deg):
    """Convert latitude/longitude to spherical coordinates."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    theta = np.pi/2 - lat_rad  # Polar angle
    phi = lon_rad               # Azimuthal angle
    return theta, phi

# New York City: 40.7°N, 74.0°W
ny_theta, ny_phi = latlon_to_theta_phi(40.7, -74.0)

# London: 51.5°N, 0.1°W
london_theta, london_phi = latlon_to_theta_phi(51.5, -0.1)

# Tokyo: 35.7°N, 139.7°E
tokyo_theta, tokyo_phi = latlon_to_theta_phi(35.7, 139.7)

R_earth = 6371  # km

d_ny_london = distance_on_sphere(ny_theta, ny_phi, london_theta, london_phi, R_earth)
d_ny_tokyo = distance_on_sphere(ny_theta, ny_phi, tokyo_theta, tokyo_phi, R_earth)

print(f"New York to London: {d_ny_london:.0f} km")
print(f"New York to Tokyo: {d_ny_tokyo:.0f} km")
```

---

## 3. Christoffel Symbols and Connections

### 📖 Concept

On a curved manifold, vectors at different points lie in different tangent spaces. To compare vectors or take derivatives, we need a **connection** - a rule for "connecting" nearby tangent spaces.

The **Christoffel symbols** Γ^λ_μν define the Levi-Civita connection (the unique torsion-free, metric-compatible connection):

```
Γ^λ_μν = (1/2) g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
```

**Physical interpretation:**
- Γ^λ_μν tells you how basis vectors change as you move through the manifold
- In flat space (Cartesian coords), all Γ^λ_μν = 0
- Non-zero Christoffel symbols ⟹ curved space OR non-Cartesian coordinates

**Key properties:**
- Symmetric in lower indices: Γ^λ_μν = Γ^λ_νμ
- NOT a tensor (transforms non-linearly)
- Vanishes in locally inertial coordinates (equivalence principle!)

**Why this matters for GR:** Christoffel symbols appear in the geodesic equation and covariant derivative. They encode how to do calculus on curved spacetime!

---

### 💻 Code Example: Computing Christoffel Symbols

```python
import sympy as sp  # SymPy for symbolic mathematics

def christoffel_symbols(g, coords):  # symbols() creates symbolic variables
    """
    Compute Christoffel symbols from metric tensor.

    Parameters:
    g: Metric tensor (sympy Matrix)
    coords: List of coordinate symbols

    Returns:
    Dictionary of non-zero Christoffel symbols
    """
    n = len(coords)

    # Compute inverse metric
    g_inv = g.inv()

    # Compute Christoffel symbols
    christoffel = {}

    for lam in range(n):
        for mu in range(n):
            for nu in range(mu, n):  # Use symmetry
                Gamma = 0
                for sigma in range(n):
                    term1 = sp.diff(g[nu, sigma], coords[mu])  # diff() computes symbolic derivative
                    term2 = sp.diff(g[mu, sigma], coords[nu])  # diff() computes symbolic derivative
                    term3 = sp.diff(g[mu, nu], coords[sigma])  # diff() computes symbolic derivative
                    Gamma += sp.Rational(1, 2) * g_inv[lam, sigma] * (term1 + term2 - term3)

                Gamma = sp.simplify(Gamma)  # simplify() algebraically simplifies expression
                if Gamma != 0:
                    christoffel[(lam, mu, nu)] = Gamma
                    if mu != nu:  # Symmetry
                        christoffel[(lam, nu, mu)] = Gamma

    return christoffel

# Example: 2-sphere metric
theta, phi, R = sp.symbols('theta phi R', real=True, positive=True)  # symbols() creates symbolic variables
coords = [theta, phi]

g_sphere = sp.Matrix([  # Matrix() creates symbolic matrix
    [R**2, 0],
    [0, R**2 * sp.sin(theta)**2]
])

print("Metric on 2-sphere:")
print(g_sphere)

Gamma = christoffel_symbols(g_sphere, coords)  # symbols() creates symbolic variables

print("\nNon-zero Christoffel symbols:")
for key, value in sorted(Gamma.items()):
    lam, mu, nu = key
    coord_names = ['θ', 'φ']
    print(f"Γ^{coord_names[lam]}_{coord_names[mu]}{coord_names[nu]} = {value}")
```

**Expected output:**
```
Non-zero Christoffel symbols:
Γ^θ_φφ = -sin(θ)cos(θ)
Γ^φ_θφ = cos(θ)/sin(θ) = cot(θ)
Γ^φ_φθ = cot(θ)
```

---

### 📊 Visualization: Parallel Transport on Sphere

Christoffel symbols describe how vectors change during **parallel transport**:

```python
# Demonstrate parallel transport around a triangle on sphere
fig = plt.figure(figsize=(10, 8))  # plt.figure() creates a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  projection='3d'  # Create 3D axes

R = 1.0

# Draw sphere
u = np.linspace(0, 2*np.pi, 50)  # np.linspace() creates evenly spaced array between start and end
v = np.linspace(0, np.pi, 50)  # np.linspace() creates evenly spaced array between start and end
x_sphere = R * np.outer(np.cos(u), np.sin(v))  # np.sin() computes sine (element-wise for arrays)
y_sphere = R * np.outer(np.sin(u), np.sin(v))  # np.sin() computes sine (element-wise for arrays)
z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))  # np.cos() computes cosine (element-wise for arrays)
ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color=COLORS['blue'])  # .plot_surface() draws 3D surface plot

# Triangle vertices (in spherical coords)
# North pole, point on equator, another point on equator
vertices_spherical = [
    (0.1, 0),           # Near north pole
    (np.pi/2, 0),       # Equator, longitude 0
    (np.pi/2, np.pi/2)  # Equator, longitude 90°
]

# Convert to Cartesian and plot triangle
triangle_x, triangle_y, triangle_z = [], [], []
for theta, phi in vertices_spherical:
    x = R * np.sin(theta) * np.cos(phi)  # np.sin() computes sine (element-wise for arrays)
    y = R * np.sin(theta) * np.sin(phi)  # np.sin() computes sine (element-wise for arrays)
    z = R * np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
    triangle_x.append(x)
    triangle_y.append(y)
    triangle_z.append(z)

triangle_x.append(triangle_x[0])  # Close triangle
triangle_y.append(triangle_y[0])
triangle_z.append(triangle_z[0])

ax.plot(triangle_x, triangle_y, triangle_z, color=COLORS['orange'],
        linewidth=3, marker='o', markersize=8, label='Geodesic triangle')

# Draw vectors at each vertex showing parallel transport
# Start with vector pointing east at vertex 0
initial_direction = np.array([1, 0, 0])  # np.array() converts Python list/tuple to efficient numpy array

# Vertex 0: Near north pole
v0_start = np.array([triangle_x[0], triangle_y[0], triangle_z[0]])  # np.array() converts Python list/tuple to efficient numpy array
v0_dir = initial_direction - np.dot(initial_direction, v0_start) * v0_start  # np.dot() computes dot product of two arrays
v0_dir = 0.3 * v0_dir / np.linalg.norm(v0_dir)  # np.linalg.norm() computes vector magnitude (Euclidean norm)

# Vertex 1: After transport along meridian (equator, φ=0)
# Transport along meridian rotates vector by -Δθ
v1_start = np.array([triangle_x[1], triangle_y[1], triangle_z[1]])  # np.array() converts Python list/tuple to efficient numpy array
v1_dir = np.array([0, 0, 0.3])  # Points north after following geodesic down

# Vertex 2: After transport along equator (φ: 0 → π/2)
# Vector stays pointing north on equator
v2_start = np.array([triangle_x[2], triangle_y[2], triangle_z[2]])  # np.array() converts Python list/tuple to efficient numpy array
v2_dir = np.array([-0.3, 0, 0])  # Still points north, but coordinate system rotated

# Draw all three vectors
ax.quiver(v0_start[0], v0_start[1], v0_start[2],
          v0_dir[0], v0_dir[1], v0_dir[2],
          color=COLORS['green'], arrow_length_ratio=0.3, linewidth=2.5,
          label='Vector at vertex 0 (start)')

ax.quiver(v1_start[0], v1_start[1], v1_start[2],
          v1_dir[0], v1_dir[1], v1_dir[2],
          color=COLORS['orange'], arrow_length_ratio=0.3, linewidth=2.5,
          label='Vector at vertex 1 (transported)')

ax.quiver(v2_start[0], v2_start[1], v2_start[2],
          v2_dir[0], v2_dir[1], v2_dir[2],
          color=COLORS['red'], arrow_length_ratio=0.3, linewidth=2.5,
          label='Vector at vertex 2 (transported)')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Parallel Transport on 2-Sphere\n(Vector rotates even along geodesics!)')
ax.legend()
plt.show()  # plt.show() displays the figure window

print("\nKey insight: After parallel transport around a closed loop,")
print("the vector returns rotated! This rotation angle measures curvature.")
```

---

### 🎯 Practice Question #2

**Q:** In flat Euclidean space with Cartesian coordinates, what are all the Christoffel symbols?

<details>
<summary>💡 Hint 1</summary>

In Cartesian coordinates, the metric is g_μν = δ_μν (constant).
</details>

<details>
<summary>💡 Hint 2</summary>

The Christoffel symbols depend on derivatives of the metric: ∂_μ g_νλ.
</details>

<details>
<summary>✅ Answer</summary>

**All Christoffel symbols are zero:** Γ^λ_μν = 0

This is because g_μν = δ_μν (constant), so all derivatives ∂_μ g_νλ = 0.

```python
import sympy as sp  # SymPy for symbolic mathematics
x, y, z = sp.symbols('x y z', real=True)  # symbols() creates symbolic variables
coords = [x, y, z]

g_flat = sp.Matrix([  # Matrix() creates symbolic matrix
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

Gamma = christoffel_symbols(g_flat, coords)  # symbols() creates symbolic variables
print(f"Christoffel symbols in flat space: {Gamma}")  # Empty dict!
```

This is why Newtonian physics works in flat space - no need for connection!
</details>

---

## 4. Covariant Derivatives

### 📖 Concept

The ordinary partial derivative ∂_μ doesn't respect the geometry of curved spaces - it's not a tensor operation. We need the **covariant derivative** ∇_μ.

**For a vector field V^λ:**
```
∇_μ V^λ = ∂_μ V^λ + Γ^λ_μν V^ν
```

**For a covector field ω_λ:**
```
∇_μ ω_λ = ∂_μ ω_λ - Γ^ν_μλ ω_ν
```

**Key properties:**
- ∇_μ IS a tensor (transforms properly)
- Reduces to ∂_μ in flat space (Cartesian coords)
- Satisfies Leibniz rule: ∇_μ(T ⊗ S) = (∇_μ T) ⊗ S + T ⊗ (∇_μ S)
- **Metric compatibility:** ∇_μ g_νλ = 0 (the metric is covariantly constant)

**Physical interpretation:** The covariant derivative compares a vector at point p with a vector at a nearby point q, accounting for how the coordinate system itself changes.

**Why this matters for GR:** All laws of physics in curved spacetime use covariant derivatives instead of partial derivatives. This ensures coordinate independence!

---

### 💻 Code Example: Covariant Derivative

```python
def covariant_derivative_vector(V, Gamma, coords, direction_idx):
    """
    Compute covariant derivative of a contravariant vector.

    Parameters:
    V: Vector field (list of sympy expressions)
    Gamma: Christoffel symbols (dict)
    coords: Coordinate symbols
    direction_idx: Which coordinate to differentiate w.r.t.

    Returns:
    Covariant derivative components
    """
    n = len(coords)
    nabla_V = []

    for lam in range(n):
        # Start with partial derivative
        component = sp.diff(V[lam], coords[direction_idx])  # diff() computes symbolic derivative

        # Add Christoffel connection term
        for nu in range(n):
            key = (lam, direction_idx, nu)
            if key in Gamma:
                component += Gamma[key] * V[nu]

        nabla_V.append(sp.simplify(component))  # simplify() algebraically simplifies expression

    return nabla_V

# Example: Vector field on 2-sphere
# Let's take V = (1, 0) - pointing in θ direction everywhere

theta, phi, R = sp.symbols('theta phi R', real=True, positive=True)  # symbols() creates symbolic variables
coords = [theta, phi]

# Recompute Christoffel symbols
g_sphere = sp.Matrix([  # Matrix() creates symbolic matrix
    [R**2, 0],
    [0, R**2 * sp.sin(theta)**2]
])

Gamma = christoffel_symbols(g_sphere, coords)  # symbols() creates symbolic variables

# Vector field: V^θ = 1, V^φ = 0
V = [sp.Integer(1), sp.Integer(0)]

print("Vector field V = (1, 0) on 2-sphere")
print("\nCovariant derivative in θ direction (∇_θ V):")
nabla_theta_V = covariant_derivative_vector(V, Gamma, coords, 0)
print(f"(∇_θ V)^θ = {nabla_theta_V[0]}")
print(f"(∇_θ V)^φ = {nabla_theta_V[1]}")

print("\nCovariant derivative in φ direction (∇_φ V):")
nabla_phi_V = covariant_derivative_vector(V, Gamma, coords, 1)
print(f"(∇_φ V)^θ = {nabla_phi_V[0]}")
print(f"(∇_φ V)^φ = {nabla_phi_V[1]}")
```

---

### 🔬 Explore: Parallel Transport Equation

A vector field V^μ is **parallel transported** along a curve if ∇_u V = 0, where u is the tangent to the curve.

```python
# For a curve γ(t) with tangent u^μ = dx^μ/dt,
# parallel transport means: u^μ ∇_μ V^λ = 0
# Expanding: dx^μ/dt (∂_μ V^λ + Γ^λ_μν V^ν) = 0

print("Parallel transport equation along curve with tangent u^μ:")
print("u^μ ∇_μ V^λ = 0")
print("\nExpanded form:")
print("dV^λ/dt + Γ^λ_μν (dx^μ/dt) V^ν = 0")
print("\nThis is a system of ODEs for V^λ(t)!")
```

---

## 5. The Riemann Curvature Tensor

### 📖 Concept

The **Riemann curvature tensor** R^ρ_σμν measures the failure of second covariant derivatives to commute:

```
[∇_μ, ∇_ν] V^ρ = R^ρ_σμν V^σ
```

**Explicit formula in terms of Christoffel symbols:**
```
R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ
```

**⚠️ Sign Convention Note:** Different textbooks use different sign conventions for the Riemann tensor! This course uses the **(-,+,+,+) metric signature** and the convention where R^ρ_σμν = ∂_μ Γ^ρ_νσ - ... (positive sign on first term). Some books (like Misner-Thorne-Wheeler) use the opposite sign. The physics is the same—just be consistent within calculations!

**Key properties:**
- Rank (1,3) tensor with 4 indices
- **Antisymmetric** in last two indices: R^ρ_σμν = -R^ρ_σνμ
- **Cyclic identity:** R^ρ_σμν + R^ρ_μνσ + R^ρ_νσμ = 0
- **Bianchi identity:** ∇_λ R^ρ_σμν + ∇_μ R^ρ_σνλ + ∇_ν R^ρ_σλμ = 0
- In n dimensions: at most n²(n²-1)/12 independent components

**Physical interpretation:**
- Measures intrinsic curvature (independent of embedding)
- R = 0 ⟺ space is flat (locally Euclidean)
- Describes tidal forces in GR
- Controls parallel transport around closed loops

**Why this matters for GR:** The Riemann tensor IS curvature. It tells you the gravitational field strength and appears in Einstein's field equations!

---

### 💻 Code Example: Computing Riemann Tensor

```python
def riemann_tensor(Gamma, coords):
    """
    Compute Riemann curvature tensor from Christoffel symbols.

    Returns:
    Dictionary of non-zero components
    """
    n = len(coords)
    R = {}

    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(mu+1, n):  # Use antisymmetry
                    R_comp = 0

                    # ∂_μ Γ^ρ_νσ term
                    key1 = (rho, nu, sigma)
                    if key1 in Gamma:
                        R_comp += sp.diff(Gamma[key1], coords[mu])  # diff() computes symbolic derivative

                    # -∂_ν Γ^ρ_μσ term
                    key2 = (rho, mu, sigma)
                    if key2 in Gamma:
                        R_comp -= sp.diff(Gamma[key2], coords[nu])  # diff() computes symbolic derivative

                    # Γ^ρ_μλ Γ^λ_νσ term
                    for lam in range(n):
                        key3 = (rho, mu, lam)
                        key4 = (lam, nu, sigma)
                        if key3 in Gamma and key4 in Gamma:
                            R_comp += Gamma[key3] * Gamma[key4]

                    # -Γ^ρ_νλ Γ^λ_μσ term
                    for lam in range(n):
                        key5 = (rho, nu, lam)
                        key6 = (lam, mu, sigma)
                        if key5 in Gamma and key6 in Gamma:
                            R_comp -= Gamma[key5] * Gamma[key6]

                    R_comp = sp.simplify(R_comp)  # simplify() algebraically simplifies expression
                    if R_comp != 0:
                        R[(rho, sigma, mu, nu)] = R_comp
                        R[(rho, sigma, nu, mu)] = -R_comp  # Antisymmetry

    return R

# Compute for 2-sphere
print("Computing Riemann tensor for 2-sphere...")
print("(This may take a moment...)\n")

R = riemann_tensor(Gamma, coords)

print("Non-zero Riemann tensor components:")
coord_names = ['θ', 'φ']
for key, value in sorted(R.items()):
    rho, sigma, mu, nu = key
    if mu < nu:  # Only print one of each antisymmetric pair
        print(f"R^{coord_names[rho]}_{coord_names[sigma]}{coord_names[mu]}{coord_names[nu]} = {value}")
```

**Expected output:**
```
R^θ_φθφ = sin²(θ)
R^φ_θθφ = 1
```

---

### 📊 Visualization: Curvature of 2-Sphere

```python
# The Gaussian curvature of a 2-sphere is K = 1/R²
# We can visualize how curvature is constant everywhere on the sphere

fig = plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Left: Sphere with curvature representation
ax1 = fig.add_subplot(121, projection='3d')  projection='3d'  # Create 3D axes

u = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
v = np.linspace(0, np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
R_val = 1.0

x = R_val * np.outer(np.cos(u), np.sin(v))  # np.sin() computes sine (element-wise for arrays)
y = R_val * np.outer(np.sin(u), np.sin(v))  # np.sin() computes sine (element-wise for arrays)
z = R_val * np.outer(np.ones(np.size(u)), np.cos(v))  # np.cos() computes cosine (element-wise for arrays)

# Color by curvature (constant = 1/R² = 1 for unit sphere)
K = 1.0 / R_val**2
curvature_color = np.ones_like(z) * K

surf = ax1.plot_surface(x, y, z, facecolors=plt.cm.viridis(curvature_color/K),  # .plot_surface() draws 3D surface plot
                        alpha=0.8, rstride=1, cstride=1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title(f'2-Sphere: K = 1/R² = {K:.2f}')

# Right: Compare different geometries
ax2 = fig.add_subplot(122)

geometries = ['Sphere\n(K > 0)', 'Plane\n(K = 0)', 'Hyperboloid\n(K < 0)']
curvatures = [1.0, 0.0, -1.0]
colors_list = [COLORS['blue'], COLORS['gray'], COLORS['red']]

bars = ax2.bar(geometries, curvatures, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
ax2.axhline(y=0, color='k', linewidth=1)
ax2.set_ylabel('Gaussian Curvature K')
ax2.set_title('Curvature Sign Determines Geometry')
ax2.grid(True, alpha=0.3, axis='y')

# Add annotations
ax2.text(0, 0.5, 'Positive:\nTriangles have\nangle sum > π',
         ha='center', fontsize=9)
ax2.text(1, 0.1, 'Zero:\nEuclidean\ngeometry',
         ha='center', fontsize=9)
ax2.text(2, -0.5, 'Negative:\nTriangles have\nangle sum < π',
         ha='center', fontsize=9)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🎯 Practice Question #3

**Q:** What does R^ρ_σμν = 0 everywhere tell you about the space?

<details>
<summary>💡 Hint</summary>

Think about what happens to parallel transported vectors and geodesic deviation.
</details>

<details>
<summary>✅ Answer</summary>

**The space is flat** (locally Euclidean).

If R^ρ_σμν = 0 everywhere, then:
- Covariant derivatives commute: [∇_μ, ∇_ν] = 0
- Parallel transport is path-independent
- We can find global Cartesian coordinates where g_μν = δ_μν
- No tidal forces (in GR context)

This is why Newtonian gravity (flat spacetime approximation) works well for weak fields!
</details>

---

## 6. Ricci Tensor and Scalar Curvature

### 📖 Concept

The Riemann tensor has too many components for most purposes. We contract it to get simpler curvature measures:

**1. Ricci tensor** (trace over first and third indices):
```
R_μν = R^λ_μλν = g^λσ R_σμλν
```

**2. Ricci scalar** (trace of Ricci tensor):
```
R = g^μν R_μν
```

**Key properties:**
- Ricci tensor is **symmetric**: R_μν = R_νμ
- In n dimensions: Ricci has n(n+1)/2 independent components
- Ricci scalar is a single number at each point
- **Einstein tensor:** G_μν = R_μν - (1/2)g_μν R

**Why this matters for GR:** The Einstein field equations are:
```
G_μν = 8πG/c⁴ T_μν
```
where G_μν (made from Ricci tensor) describes spacetime curvature and T_μν is the stress-energy tensor (matter/energy). This is THE fundamental equation of General Relativity!

---

### 💻 Code Example: Ricci Tensor and Scalar

```python
def ricci_tensor(R, g, n):
    """
    Compute Ricci tensor from Riemann tensor.

    Parameters:
    R: Riemann tensor (dict)
    g: Metric tensor (Matrix)
    n: Dimension

    Returns:
    Ricci tensor (Matrix)
    """
    g_inv = g.inv()
    Ric = sp.zeros(n, n)

    for mu in range(n):
        for nu in range(n):
            for lam in range(n):
                for sigma in range(n):
                    key = (sigma, mu, lam, nu)
                    if key in R:
                        Ric[mu, nu] += g_inv[lam, sigma] * R[key]

    return sp.simplify(Ric)  # simplify() algebraically simplifies expression

def ricci_scalar(Ric, g, n):
    """Compute Ricci scalar from Ricci tensor."""
    g_inv = g.inv()
    R_scalar = 0

    for mu in range(n):
        for nu in range(n):
            R_scalar += g_inv[mu, nu] * Ric[mu, nu]

    return sp.simplify(R_scalar)  # simplify() algebraically simplifies expression

# Compute for 2-sphere
n = 2
Ric = ricci_tensor(R, g_sphere, n)

print("Ricci tensor for 2-sphere:")
print(Ric)

R_scalar = ricci_scalar(Ric, g_sphere, n)
print(f"\nRicci scalar R = {R_scalar}")
print(f"\nSimplified: R = {sp.simplify(R_scalar/R**2)} × (1/R²)")  # simplify() algebraically simplifies expression
print("\nFor unit sphere (R=1): R = 2")
print("Gaussian curvature: K = R/2 = 1/R²")
```

**Expected output:**
```
Ricci tensor for 2-sphere:
Matrix([[1, 0], [0, sin²(θ)]])

Ricci scalar R = 2/R²

For unit sphere (R=1): R = 2
Gaussian curvature: K = R/2 = 1/R²
```

---

### 📊 Visualization: Ricci Curvature Flow

```python
# Visualize how spaces with different Ricci curvatures evolve

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

t_vals = np.linspace(0, 2, 100)  # np.linspace() creates evenly spaced array between start and end

# Positive Ricci: contracts (like sphere)
r_positive = np.exp(-t_vals)  # np.exp() computes exponential e^x
axes[0].plot(t_vals, r_positive, color=COLORS['blue'], linewidth=3)
axes[0].fill_between(t_vals, 0, r_positive, alpha=0.3, color=COLORS['blue'])
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Radius')
axes[0].set_title('Positive Ricci: R > 0\n(Sphere contracts)')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1.2])

# Zero Ricci: stays flat
r_zero = np.ones_like(t_vals)
axes[1].plot(t_vals, r_zero, color=COLORS['gray'], linewidth=3)
axes[1].fill_between(t_vals, 0, r_zero, alpha=0.3, color=COLORS['gray'])
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Radius')
axes[1].set_title('Zero Ricci: R = 0\n(Flat space unchanging)')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1.2])

# Negative Ricci: expands (like hyperbolic space)
r_negative = np.exp(t_vals/2)  # np.exp() computes exponential e^x
axes[2].plot(t_vals, r_negative, color=COLORS['red'], linewidth=3)
axes[2].fill_between(t_vals, 0, r_negative, alpha=0.3, color=COLORS['red'])
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Radius')
axes[2].set_title('Negative Ricci: R < 0\n(Hyperbolic expands)')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([0, 3])

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Ricci flow is used in:")
print("- Poincaré conjecture proof (Perelman)")
print("- Cosmology (expanding universe)")
print("- General Relativity (spacetime evolution)")
```

---

## 7. Geodesics: Straight Lines in Curved Space

### 📖 Concept

A **geodesic** is the curved-space generalization of a straight line. It's the path a free particle follows (in GR: how objects move under gravity alone).

**Definition:** A curve γ(t) with tangent u^μ = dx^μ/dt is a geodesic if:
```
∇_u u = 0
```
or equivalently:
```
u^μ ∇_μ u^ν = 0
```

**Geodesic equation** (expanded form):
```
d²x^λ/dt² + Γ^λ_μν (dx^μ/dt)(dx^ν/dt) = 0
```

**Key properties:**
- Geodesics are locally the shortest paths (or longest for timelike curves in GR)
- Determined by 2nd order ODEs (need position and velocity as initial conditions)
- **Affinely parameterized:** t is proportional to proper time/distance
- In flat space: geodesics are straight lines

**Why this matters for GR:** Geodesics are the worldlines of freely falling particles. Planets orbit the sun by following geodesics in curved spacetime!

---

### 💻 Code Example: Geodesic Equation

```python
def geodesic_equations(Gamma, coords):
    """
    Generate geodesic equations from Christoffel symbols.

    Returns:
    List of symbolic equations for d²x^λ/dt²
    """
    n = len(coords)
    t = sp.symbols('t', real=True)  # symbols() creates symbolic variables

    # Create functions x^μ(t)
    x = [sp.Function(f'x{i}')(t) for i in range(n)]

    geodesic_eqs = []

    for lam in range(n):
        # Start with d²x^λ/dt²
        eq = sp.diff(x[lam], t, 2)  # diff() computes symbolic derivative

        # Add Christoffel terms: Γ^λ_μν dx^μ/dt dx^ν/dt
        for mu in range(n):
            for nu in range(n):
                key = (lam, mu, nu)
                if key in Gamma:
                    # Substitute coordinate functions
                    Gamma_sub = Gamma[key]
                    for i in range(n):
                        Gamma_sub = Gamma_sub.subs(coords[i], x[i])

                    eq += Gamma_sub * sp.diff(x[mu], t) * sp.diff(x[nu], t)  # diff() computes symbolic derivative

        geodesic_eqs.append(sp.simplify(eq))  # simplify() algebraically simplifies expression

    return geodesic_eqs, x

# Generate geodesic equations for 2-sphere
print("Geodesic equations on 2-sphere:\n")

geod_eqs, x_funcs = geodesic_equations(Gamma, coords)

print("Equation for θ(t):")
print(f"d²θ/dt² + ... = 0")
print(f"Full form: {geod_eqs[0]} = 0")

print("\nEquation for φ(t):")
print(f"d²φ/dt² + ... = 0")
print(f"Full form: {geod_eqs[1]} = 0")

print("\nThese are the equations great circles must satisfy!")
```

---

### 📊 Visualization: Geodesics on 2-Sphere

```python
from scipy.integrate import odeint  # ODE solver for initial value problems

def sphere_geodesic_ode(y, t, R=1.0):
    """
    ODE system for geodesics on 2-sphere.
    y = [θ, φ, dθ/dt, dφ/dt]
    """
    theta, phi, theta_dot, phi_dot = y

    # Avoid singularities at poles
    if abs(np.sin(theta)) < 1e-10:  # np.sin() computes sine (element-wise for arrays)
        theta = 1e-10

    # Geodesic equations
    theta_ddot = np.sin(theta) * np.cos(theta) * phi_dot**2  # np.sin() computes sine (element-wise for arrays)
    phi_ddot = -2 * (np.cos(theta) / np.sin(theta)) * theta_dot * phi_dot  # np.sin() computes sine (element-wise for arrays)

    return [theta_dot, phi_dot, theta_ddot, phi_ddot]

# Initial conditions for different geodesics
R = 1.0
t_span = np.linspace(0, np.pi, 200)  # np.linspace() creates evenly spaced array between start and end

fig = plt.figure(figsize=(14, 6))  # plt.figure() creates a new figure for plotting

# Left: Geodesics on sphere
ax1 = fig.add_subplot(121, projection='3d')  projection='3d'  # Create 3D axes

# Draw sphere
u = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
v = np.linspace(0, np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
x = R * np.outer(np.cos(u), np.sin(v))  # np.sin() computes sine (element-wise for arrays)
y = R * np.outer(np.sin(u), np.sin(v))  # np.sin() computes sine (element-wise for arrays)
z = R * np.outer(np.ones(np.size(u)), np.cos(v))  # np.cos() computes cosine (element-wise for arrays)
ax1.plot_surface(x, y, z, alpha=0.2, color=COLORS['blue'])  # .plot_surface() draws 3D surface plot

# Solve for several geodesics (great circles)
initial_conditions = [
    [np.pi/2, 0, 0, 1],           # Equator
    [0.1, 0, 1, 0],                # Meridian
    [np.pi/2, 0, 0.7, 0.7],       # Tilted great circle
]

colors_geod = [COLORS['orange'], COLORS['green'], COLORS['purple']]

for i, y0 in enumerate(initial_conditions):
    sol = odeint(sphere_geodesic_ode, y0, t_span)  # odeint() solves ODE system with given initial conditions

    # Convert to Cartesian
    theta_sol = sol[:, 0]
    phi_sol = sol[:, 1]

    x_geod = R * np.sin(theta_sol) * np.cos(phi_sol)  # np.sin() computes sine (element-wise for arrays)
    y_geod = R * np.sin(theta_sol) * np.sin(phi_sol)  # np.sin() computes sine (element-wise for arrays)
    z_geod = R * np.cos(theta_sol)  # np.cos() computes cosine (element-wise for arrays)

    ax1.plot(x_geod, y_geod, z_geod, color=colors_geod[i], linewidth=2, label=f'Geodesic {i+1}')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Geodesics on 2-Sphere\n(Great Circles)')
ax1.legend()

# Right: Non-geodesic path (latitude circle)
ax2 = fig.add_subplot(122, projection='3d')  projection='3d'  # Create 3D axes
ax2.plot_surface(x, y, z, alpha=0.2, color=COLORS['blue'])  # .plot_surface() draws 3D surface plot

# Geodesic (great circle)
sol_geo = odeint(sphere_geodesic_ode, [np.pi/2, 0, 0, 1], t_span)  # odeint() solves ODE system with given initial conditions
theta_geo = sol_geo[:, 0]
phi_geo = sol_geo[:, 1]
x_geo = R * np.sin(theta_geo) * np.cos(phi_geo)  # np.sin() computes sine (element-wise for arrays)
y_geo = R * np.sin(theta_geo) * np.sin(phi_geo)  # np.sin() computes sine (element-wise for arrays)
z_geo = R * np.cos(theta_geo)  # np.cos() computes cosine (element-wise for arrays)
ax2.plot(x_geo, y_geo, z_geo, color=COLORS['green'], linewidth=3, label='Geodesic (equator)')

# Non-geodesic (latitude circle at 45°)
theta_lat = np.pi/4
phi_lat = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
x_lat = R * np.sin(theta_lat) * np.cos(phi_lat)  # np.sin() computes sine (element-wise for arrays)
y_lat = R * np.sin(theta_lat) * np.sin(phi_lat)  # np.sin() computes sine (element-wise for arrays)
z_lat = R * np.cos(theta_lat) * np.ones_like(phi_lat)  # np.cos() computes cosine (element-wise for arrays)
ax2.plot(x_lat, y_lat, z_lat, color=COLORS['red'], linewidth=3, linestyle='--', label='Non-geodesic (latitude)')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Geodesic vs Non-Geodesic')
ax2.legend()

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("\nKey insight: Great circles are geodesics (shortest paths).")
print("Latitude circles (except equator) are NOT geodesics!")
```

---

### 🔬 Explore: Geodesic Deviation

Nearby geodesics accelerate away from each other due to curvature:

```python
# Geodesic deviation equation: D²ξ^μ/Dt² = -R^μ_νρσ u^ν u^ρ ξ^σ
# where ξ^μ is the separation vector between nearby geodesics

print("Geodesic Deviation Equation:")
print("D²ξ^μ/Dt² = -R^μ_νρσ u^ν u^ρ ξ^σ")
print("\nPhysical meaning:")
print("- ξ^μ: separation between nearby geodesics")
print("- u^μ: velocity along geodesics")
print("- R^μ_νρσ: Riemann curvature tensor")
print("\nIn GR: This describes tidal forces!")
print("Example: Moon creates tidal bulges on Earth because nearby")
print("geodesics (different parts of Earth) accelerate differently.")
```

---

### 🎯 Practice Question #4

**Q:** What is the geodesic equation in flat space with Cartesian coordinates?

<details>
<summary>💡 Hint</summary>

Remember: all Christoffel symbols vanish in flat space with Cartesian coordinates.
</details>

<details>
<summary>✅ Answer</summary>

**d²x^λ/dt² = 0**

Since all Γ^λ_μν = 0 in flat space (Cartesian coords), the geodesic equation simplifies to:
```
d²x^λ/dt² + Γ^λ_μν (dx^μ/dt)(dx^ν/dt) = 0
d²x^λ/dt² = 0
```

Integrating: x^λ(t) = a^λ + b^λ t

This is a **straight line** with constant velocity - exactly what we expect in flat space!
</details>

---

## 8. Practice Questions

### Section 1-2: Metric Tensor

**Q1:** Write the metric tensor for 2D polar coordinates (r, θ) in flat Euclidean space.

<details>
<summary>💡 Hint</summary>

Start from ds² = dx² + dy² and use x = r cos(θ), y = r sin(θ).
</details>

<details>
<summary>✅ Answer</summary>

```
g = [[1,  0  ],
     [0,  r² ]]
```

Line element: ds² = dr² + r²dθ²
</details>

---

**Q2:** For a 2-sphere of radius R, what is g_φφ at the north pole (θ = 0)?

<details>
<summary>💡 Hint</summary>

The metric component is g_φφ = R² sin²θ.
</details>

<details>
<summary>✅ Answer</summary>

**g_φφ = 0** at θ = 0 (north pole)

This makes sense: circumference of latitude circles → 0 as we approach the pole!
</details>

---

### Section 3-4: Christoffel Symbols & Covariant Derivatives

**Q3:** Compute Γ^r_θθ for the 2D polar metric g = diag(1, r²).

<details>
<summary>💡 Hint</summary>

Use: Γ^λ_μν = (1/2) g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
</details>

<details>
<summary>✅ Answer</summary>

**Γ^r_θθ = -r**

```python
r, theta = sp.symbols('r theta', real=True, positive=True)  # symbols() creates symbolic variables
coords = [r, theta]
g = sp.Matrix([[1, 0], [0, r**2]])  # Matrix() creates symbolic matrix
Gamma = christoffel_symbols(g, coords)  # symbols() creates symbolic variables
print(Gamma[(0, 1, 1)])  # -r
```

Physical meaning: Centrifugal "force" in rotating reference frame!
</details>

---

**Q4:** What is ∇_μ g_νλ for the Levi-Civita connection?

<details>
<summary>✅ Answer</summary>

**∇_μ g_νλ = 0** (metric compatibility)

The covariant derivative of the metric always vanishes - this is a defining property of the Levi-Civita connection.
</details>

---

### Section 5-6: Curvature Tensors

**Q5:** How many independent components does the Riemann tensor have in 4D spacetime?

<details>
<summary>💡 Hint</summary>

Use the formula: n²(n²-1)/12 for n dimensions.
</details>

<details>
<summary>✅ Answer</summary>

**20 independent components**

For n=4: 4²(4²-1)/12 = 16×15/12 = 20

In GR, these 20 numbers at each spacetime point completely describe the gravitational field!
</details>

---

**Q6:** What is the Ricci scalar for a 2-sphere of radius R?

<details>
<summary>💡 Hint</summary>

Recall from earlier: R = 2/R² for the 2-sphere.
</details>

<details>
<summary>✅ Answer</summary>

**R = 2/R²**

For unit sphere (R=1): R = 2

The Gaussian curvature is K = R/2 = 1/R².
</details>

---

### Section 7: Geodesics

**Q7:** On a sphere, are latitude circles (except the equator) geodesics?

<details>
<summary>💡 Hint</summary>

Check if they satisfy the geodesic equation: ∇_u u = 0.
</details>

<details>
<summary>✅ Answer</summary>

**No, only the equator is a geodesic among latitude circles.**

Latitude circles (except equator) have non-zero geodesic curvature. Only great circles are geodesics on a sphere.

You can verify: walking along a latitude circle at constant speed requires "turning" (acceleration perpendicular to your path).
</details>

---

**Q8:** In General Relativity, what physical quantity is maximized along a timelike geodesic?

<details>
<summary>💡 Hint</summary>

Think about what a freely-falling observer measures.
</details>

<details>
<summary>✅ Answer</summary>

**Proper time τ**

A freely-falling observer (following a timelike geodesic) experiences the maximum proper time between two events. This is the "twin paradox" principle!
</details>

---

## Summary: The Geometry-Physics Dictionary

| Geometric Concept | Physical Meaning in GR |
|------------------|------------------------|
| Metric tensor g_μν | Spacetime geometry itself |
| Christoffel symbols Γ^λ_μν | Gravitational "force" (connection) |
| Covariant derivative ∇_μ | Derivative accounting for curvature |
| Riemann tensor R^ρ_σμν | Tidal forces / gravitational field |
| Ricci tensor R_μν | Matter density effect on curvature |
| Ricci scalar R | Total curvature at a point |
| Geodesics | Free-fall trajectories |
| Geodesic deviation | Tidal acceleration |

---

## Next Steps

✅ Master computing Christoffel symbols from metrics
✅ Practice covariant derivatives on vector fields
✅ Compute Riemann tensor for simple spaces
✅ Solve geodesic equations numerically
✅ **Ready for Lesson 8: Classical Mechanics** (Lagrangian formulation)

**Additional Resources:**
- Sean Carroll's "Spacetime and Geometry" (Chapter 3)
- Schutz's "A First Course in General Relativity" (Chapters 5-6)
- MIT OCW 8.962: General Relativity (Lectures 4-7)

**Need Help?** Use the AI assistant:
```python
from utils.ai_assistant import AIAssistant
assistant = AIAssistant()
assistant.set_lesson_context("Lesson 7", "Riemannian Geometry",
                            ["metric", "Christoffel", "Riemann tensor", "geodesics"])
assistant.ask("How does the Riemann tensor relate to tidal forces in GR?")
```

---

**Ready to continue?** → [Lesson 8: Classical Mechanics](../08_classical_mechanics/LESSON.md)
