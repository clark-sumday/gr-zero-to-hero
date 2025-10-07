# Lesson 5: Manifolds

**Topics:** Topological spaces, manifolds, charts and atlases, differentiable manifolds, tangent spaces, vector fields, differential forms

**Prerequisites:** Linear algebra (Lesson 1), multivariable calculus (Lesson 2), curves and surfaces (Lesson 4)

**Time:** ~5-6 hours

---

## Table of Contents

1. [What is a Manifold?](#1-what-is-a-manifold)
2. [Charts and Atlases](#2-charts-and-atlases)
3. [Differentiable Manifolds](#3-differentiable-manifolds)
4. [Tangent Vectors and Tangent Spaces](#4-tangent-vectors-and-tangent-spaces)
5. [Vector Fields](#5-vector-fields)
6. [The Cotangent Space and One-Forms](#6-the-cotangent-space-and-one-forms)
7. [Examples: Manifolds in Physics](#7-examples-manifolds-in-physics)

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

## 1. What is a Manifold?

### 📖 Concept

A **manifold** is a space that looks locally like Euclidean space ℝⁿ, but may have a different global structure.

**Intuitive examples:**
- **Circle (S¹):** Locally looks like a line, globally loops back
- **Sphere (S²):** Locally looks like a flat plane, globally curved
- **Torus:** Locally flat, globally has a donut shape
- **Earth's surface:** Feels flat locally, but is actually a sphere

**Formal definition:** An n-dimensional manifold M is a set where:
1. Each point has a neighborhood homeomorphic to an open set in ℝⁿ
2. These neighborhoods can be smoothly patched together
3. The space is Hausdorff and second-countable (technical conditions)

**Key insight:** You can do calculus on manifolds without needing an embedding in higher-dimensional space! This is crucial for GR where spacetime is a 4D manifold that isn't embedded in anything.

**Why this matters for GR:** Spacetime in General Relativity is a 4-dimensional smooth manifold. We need manifold theory to describe physics in curved spacetime without assuming it sits in a higher-dimensional flat space.

---

### 💻 Code Example: Circle as Manifold

The circle S¹ is the simplest non-trivial manifold. Let's represent it:

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Circle as a manifold: points at angle θ
class Circle:
    """The circle S¹ as a 1D manifold"""

    def __init__(self, radius=1.0):
        self.radius = radius
        self.dim = 1  # 1-dimensional manifold

    def embed(self, theta):
        """Embed S¹ into ℝ²: θ → (x, y)"""
        x = self.radius * np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
        y = self.radius * np.sin(theta)  # np.sin() computes sine (element-wise for arrays)
        return np.array([x, y])  # np.array() converts Python list/tuple to efficient numpy array

    def chart_upper(self, point):
        """Chart covering upper semicircle: (x,y) → x coordinate"""
        x, y = point
        if y < 0:
            raise ValueError("Point not in upper chart domain")
        return x

    def chart_lower(self, point):
        """Chart covering lower semicircle: (x,y) → x coordinate"""
        x, y = point
        if y > 0:
            raise ValueError("Point not in lower chart domain")
        return x

# Create circle
S1 = Circle(radius=1.0)

# Sample points
thetas = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
points = np.array([S1.embed(t) for t in thetas])  # np.array() converts Python list/tuple to efficient numpy array

print(f"Circle S¹ as a {S1.dim}-dimensional manifold")
print(f"Sample point at θ=π/4: {S1.embed(np.pi/4)}")
print(f"Sample point at θ=π: {S1.embed(np.pi)}")
```

**Expected output:**
```
Circle S¹ as a 1-dimensional manifold
Sample point at θ=π/4: [0.70710678 0.70710678]
Sample point at θ=π: [-1.  0.]
```

---

### 📊 Visualization: Circle as Manifold

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Circle in ℝ²
ax1 = axes[0]
ax1.plot(points[:, 0], points[:, 1], color=COLORS['blue'], linewidth=3)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Circle S¹ embedded in ℝ²')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Mark several points
sample_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
for theta in sample_angles:
    pt = S1.embed(theta)
    ax1.plot(pt[0], pt[1], 'o', color=COLORS['orange'], markersize=8)
    ax1.text(pt[0]*1.2, pt[1]*1.2, f'θ={theta/np.pi:.2f}π',
            fontsize=9, ha='center')

# Right: "Unrolled" circle showing it's locally like ℝ¹
ax2 = axes[1]
theta_vals = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
ax2.plot(theta_vals, np.ones_like(theta_vals), color=COLORS['blue'], linewidth=3)
ax2.scatter(sample_angles, np.ones(len(sample_angles)),  # np.ones() creates array filled with ones
           color=COLORS['orange'], s=100, zorder=5)
ax2.set_xlabel('θ (local coordinate)')
ax2.set_ylabel('')
ax2.set_title('Circle "unrolled" - locally looks like ℝ¹')
ax2.set_ylim([0.5, 1.5])
ax2.set_yticks([])
ax2.grid(True, alpha=0.3, axis='x')

# Show periodicity
ax2.axvline(0, color=COLORS['red'], linestyle='--', linewidth=2, alpha=0.6)
ax2.axvline(2*np.pi, color=COLORS['red'], linestyle='--', linewidth=2, alpha=0.6)
ax2.text(np.pi, 0.7, 'θ=0 and θ=2π\nare the same point!',
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

**What you should see:**
- **Left:** Circle embedded in 2D plane
- **Right:** The circle "unrolled" showing it's locally 1-dimensional, with endpoints identified

---

### 🔬 Explore: 2-Sphere S²

```python
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit

# Create sphere
u = np.linspace(0, np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
v = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
U, V = np.meshgrid(u, v)  # np.meshgrid() creates coordinate matrices from coordinate vectors

X = np.sin(U) * np.cos(V)  # np.sin() computes sine (element-wise for arrays)
Y = np.sin(U) * np.sin(V)  # np.sin() computes sine (element-wise for arrays)
Z = np.cos(U)  # np.cos() computes cosine (element-wise for arrays)

fig = plt.figure(figsize=(10, 8))  # plt.figure() creates a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  projection='3d'  # Create 3D axes
ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', edgecolor='none')  # .plot_surface() draws 3D surface plot
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Sphere S² - a 2-dimensional manifold in ℝ³')

# Mark some points
points_to_mark = [
    (np.pi/2, 0, 'North pole'),
    (np.pi/2, np.pi/2, 'Equator'),
    (0, 0, 'South pole'),
]

for u_pt, v_pt, label in points_to_mark:
    x = np.sin(u_pt) * np.cos(v_pt)  # np.sin() computes sine (element-wise for arrays)
    y = np.sin(u_pt) * np.sin(v_pt)  # np.sin() computes sine (element-wise for arrays)
    z = np.cos(u_pt)  # np.cos() computes cosine (element-wise for arrays)
    ax.scatter([x], [y], [z], color=COLORS['red'], s=100)
    ax.text(x*1.3, y*1.3, z*1.3, label, fontsize=10)

plt.show()  # plt.show() displays the figure window

print("S² is a 2D manifold: locally looks like ℝ², globally is a sphere")
print("You need at least 2 charts to cover the whole sphere!")
```

---

### 🎯 Practice Question #1

**Q:** What is the dimension of the torus (donut surface) as a manifold?

<details>
<summary>💡 Hint</summary>

Think about how many coordinates you need to specify a point on the torus surface.
</details>

<details>
<summary>✅ Answer</summary>

**2 dimensions**

The torus is a 2D manifold. You need two angles (u, v) to specify a point:
- u: angle around the small circle (tube)
- v: angle around the large circle

Locally, any small patch looks like a flat 2D plane, even though globally it's a donut shape embedded in ℝ³.

```python
# Torus parametrization: 2 parameters → 3D embedding
def torus(u, v, R=2, r=0.5):
    x = (R + r*np.cos(u)) * np.cos(v)  # np.cos() computes cosine (element-wise for arrays)
    y = (R + r*np.cos(u)) * np.sin(v)  # np.sin() computes sine (element-wise for arrays)
    z = r * np.sin(u)  # np.sin() computes sine (element-wise for arrays)
    return np.array([x, y, z])  # np.array() converts Python list/tuple to efficient numpy array

print(f"Torus: 2 parameters (u, v) → 3D space")
print(f"Dimension as manifold: 2")
```
</details>

---

## 2. Charts and Atlases

### 📖 Concept

Since manifolds may not have global coordinate systems (like trying to make a flat map of Earth), we use **charts** - local coordinate systems that cover pieces of the manifold.

**Chart:** A map φ: U → ℝⁿ where U ⊂ M is an open set on the manifold
- Takes points in M to coordinates in ℝⁿ
- Must be a homeomorphism (continuous, with continuous inverse)
- Provides "local coordinates" on that region

**Atlas:** A collection of charts that cover the entire manifold
- Union of all chart domains covers M
- Where charts overlap, we can relate coordinates using transition maps

**Transition map:** φ₂ ∘ φ₁⁻¹: changes coordinates from chart 1 to chart 2

**Example:** The sphere needs at least 2 charts:
- Stereographic projection from north pole (covers everywhere except north pole)
- Stereographic projection from south pole (covers everywhere except south pole)
- Together they cover the whole sphere!

**Why this matters for GR:** Different observers use different coordinate systems. The transition maps between coordinate systems are crucial for ensuring physics is coordinate-independent.

---

### 💻 Code Example: Charts on Circle

```python
# Two charts covering the circle
class CircleWithCharts:
    """Circle S¹ with explicit charts"""

    def __init__(self, radius=1.0):
        self.radius = radius

    def point_from_angle(self, theta):
        """Map angle to point on circle"""
        x = self.radius * np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
        y = self.radius * np.sin(theta)  # np.sin() computes sine (element-wise for arrays)
        return (x, y)

    def chart_right(self, theta):
        """
        Chart covering right semicircle: (-π/2, π/2)
        Maps to y-coordinate
        """
        if not (-np.pi/2 < theta < np.pi/2):
            return None  # Outside chart domain
        x, y = self.point_from_angle(theta)
        return y  # Local coordinate

    def chart_left(self, theta):
        """
        Chart covering left semicircle: (π/2, 3π/2)
        Maps to y-coordinate
        """
        if not (np.pi/2 < theta < 3*np.pi/2):
            return None
        x, y = self.point_from_angle(theta)
        return y  # Local coordinate

    def transition_map(self, y_coord, from_right_to_left=True):
        """
        Transition map in overlap region
        Both charts use y-coordinate, so transition is identity!
        """
        return y_coord

# Test the charts
S1 = CircleWithCharts(radius=1.0)

test_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

print("Testing charts on circle:")
print("-" * 50)
for theta in test_angles:
    pt = S1.point_from_angle(theta)
    coord_right = S1.chart_right(theta)
    coord_left = S1.chart_left(theta)

    print(f"θ = {theta:.3f} → point {pt}")
    print(f"  Right chart: {coord_right}")
    print(f"  Left chart:  {coord_left}")
    print()
```

---

### 📊 Visualization: Atlas Covering Circle

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Circle with chart domains
ax1 = axes[0, 0]
theta_full = np.linspace(0, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
circle_x = np.cos(theta_full)  # np.cos() computes cosine (element-wise for arrays)
circle_y = np.sin(theta_full)  # np.sin() computes sine (element-wise for arrays)
ax1.plot(circle_x, circle_y, color=COLORS['gray'], linewidth=2, linestyle='--',
        alpha=0.3, label='Full circle')

# Right chart domain (red)
theta_right = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, 100)  # np.linspace() creates evenly spaced array between start and end
right_x = np.cos(theta_right)  # np.cos() computes cosine (element-wise for arrays)
right_y = np.sin(theta_right)  # np.sin() computes sine (element-wise for arrays)
ax1.plot(right_x, right_y, color=COLORS['red'], linewidth=4, label='Right chart')

# Left chart domain (blue)
theta_left = np.linspace(np.pi/2 + 0.1, 3*np.pi/2 - 0.1, 100)  # np.linspace() creates evenly spaced array between start and end
left_x = np.cos(theta_left)  # np.cos() computes cosine (element-wise for arrays)
left_y = np.sin(theta_left)  # np.sin() computes sine (element-wise for arrays)
ax1.plot(left_x, left_y, color=COLORS['blue'], linewidth=4, label='Left chart')

# Overlap regions
overlap_upper = np.linspace(np.pi/2 - 0.5, np.pi/2 - 0.1, 30)  # np.linspace() creates evenly spaced array between start and end
overlap_lower = np.linspace(-np.pi/2 + 0.1, -np.pi/2 + 0.5, 30)  # np.linspace() creates evenly spaced array between start and end
for theta in overlap_upper:
    ax1.plot(np.cos(theta), np.sin(theta), 'o', color=COLORS['purple'],  # np.sin() computes sine (element-wise for arrays)
            markersize=4)
for theta in overlap_lower:
    ax1.plot(np.cos(theta), np.sin(theta), 'o', color=COLORS['purple'],  # np.sin() computes sine (element-wise for arrays)
            markersize=4)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Atlas: Two Charts Cover Circle')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax1.text(0.6, 0.3, 'Overlap\n(purple)', fontsize=10, ha='center')

# Top right: Right chart mapping
ax2 = axes[0, 1]
theta_vals = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, 100)  # np.linspace() creates evenly spaced array between start and end
y_coords = np.sin(theta_vals)  # np.sin() computes sine (element-wise for arrays)
ax2.plot(theta_vals, y_coords, color=COLORS['red'], linewidth=3)
ax2.set_xlabel('θ (on manifold)')
ax2.set_ylabel('y (local coordinate)')
ax2.set_title('Right Chart: θ → y = sin(θ)')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='k', linewidth=0.5)
ax2.axvline(0, color='k', linewidth=0.5)

# Bottom left: Left chart mapping
ax3 = axes[1, 0]
theta_vals_left = np.linspace(np.pi/2 + 0.1, 3*np.pi/2 - 0.1, 100)  # np.linspace() creates evenly spaced array between start and end
y_coords_left = np.sin(theta_vals_left)  # np.sin() computes sine (element-wise for arrays)
ax3.plot(theta_vals_left, y_coords_left, color=COLORS['blue'], linewidth=3)
ax3.set_xlabel('θ (on manifold)')
ax3.set_ylabel('y (local coordinate)')
ax3.set_title('Left Chart: θ → y = sin(θ)')
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='k', linewidth=0.5)

# Bottom right: Transition map
ax4 = axes[1, 1]
ax4.text(0.5, 0.5, 'Transition Map:\nφ_left ∘ φ_right⁻¹\n\ny_left = y_right\n\n(Identity map!)',
        transform=ax4.transAxes, fontsize=14, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.3))
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.axis('off')
ax4.set_title('Transition Map in Overlap')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🎯 Practice Question #2

**Q:** Why can't you cover a sphere with a single chart?

<details>
<summary>💡 Hint 1</summary>

Think about making a flat map of Earth. What happens at the poles?
</details>

<details>
<summary>💡 Hint 2</summary>

A chart must be a homeomorphism to an open set in ℝⁿ. Can you map the entire sphere (including all boundary points) to an open set?
</details>

<details>
<summary>✅ Answer</summary>

**You cannot cover a sphere with a single chart** because:

1. The sphere S² is compact (closed and bounded)
2. Any continuous bijection from S² to an open subset of ℝ² would map a compact space to a non-compact space
3. This contradicts a theorem in topology: continuous maps preserve compactness

**Intuitively:** Try to flatten a globe onto a map. Something must go wrong:
- Mercator projection: poles stretch to infinity
- Stereographic projection: one pole is missing (maps to ∞)

You need at least 2 charts to cover S². This is why map-makers use multiple projections!

**Famous theorem:** Any atlas for S² needs at least 2 charts (this relates to the "hairy ball theorem").
</details>

---

## 3. Differentiable Manifolds

### 📖 Concept

A **differentiable (smooth) manifold** is a manifold where we can do calculus - take derivatives, compute tangent vectors, etc.

**Requirements:**
1. Have an atlas (collection of charts covering M)
2. Transition maps must be smooth (infinitely differentiable: C^∞)
3. This allows us to define smooth functions on M

**Smooth function** f: M → ℝ is smooth if f ∘ φ⁻¹ is smooth in ℝⁿ coordinates for every chart φ.

**Key insight:** Smoothness is coordinate-independent! If a function is smooth in one coordinate system, it's smooth in all coordinate systems (because transition maps are smooth).

**Dimension:** The manifold has a well-defined dimension n - every chart maps to ℝⁿ with the same n.

**Examples:**
- ℝⁿ itself (trivial manifold, one global chart)
- S¹ (circle): 1D smooth manifold
- S² (sphere): 2D smooth manifold
- Torus: 2D smooth manifold
- Spacetime in GR: 4D smooth manifold

**Why this matters for GR:** The laws of physics must be expressible in any coordinate system. Smooth manifolds provide the mathematical framework for coordinate-independent physics.

---

### 💻 Code Example: Smooth Functions on Circle

```python
# Define smooth functions on the circle
class SmoothCircle:
    """Circle with smooth structure"""

    def height_function(self, theta):
        """
        Smooth function f: S¹ → ℝ
        Returns y-coordinate (height)
        """
        return np.sin(theta)  # np.sin() computes sine (element-wise for arrays)

    def height_function_derivative(self, theta):
        """
        Derivative of height function
        In chart using angle θ: df/dθ = cos(θ)
        """
        return np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)

    def squared_height(self, theta):
        """
        Another smooth function: f(θ) = sin²(θ)
        """
        return np.sin(theta)**2  # np.sin() computes sine (element-wise for arrays)

# Test smoothness
S1 = SmoothCircle()

theta_vals = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
heights = [S1.height_function(t) for t in theta_vals]
derivatives = [S1.height_function_derivative(t) for t in theta_vals]

print("Smooth function on circle: height(θ) = sin(θ)")
print(f"At θ=0: f={S1.height_function(0):.3f}, df/dθ={S1.height_function_derivative(0):.3f}")
print(f"At θ=π/2: f={S1.height_function(np.pi/2):.3f}, df/dθ={S1.height_function_derivative(np.pi/2):.3f}")
print(f"At θ=π: f={S1.height_function(np.pi):.3f}, df/dθ={S1.height_function_derivative(np.pi):.3f}")
```

---

### 📊 Visualization: Smooth Functions on Manifolds

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Height function on circle
ax1 = axes[0]
theta_plot = np.linspace(0, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
heights = np.sin(theta_plot)  # np.sin() computes sine (element-wise for arrays)
ax1.plot(theta_plot, heights, color=COLORS['blue'], linewidth=3,
        label='f(θ) = sin(θ)')
ax1.plot(theta_plot, np.cos(theta_plot), color=COLORS['orange'], linewidth=3,  # np.cos() computes cosine (element-wise for arrays)
        linestyle='--', label="f'(θ) = cos(θ)")
ax1.set_xlabel('θ (coordinate on S¹)', fontsize=12)
ax1.set_ylabel('Function value', fontsize=12)
ax1.set_title('Smooth Function on Circle and Its Derivative')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axhline(0, color='k', linewidth=0.5)

# Mark critical points
critical_points = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
for cp in critical_points:
    ax1.plot(cp, np.sin(cp), 'o', color=COLORS['red'], markersize=8)  # np.sin() computes sine (element-wise for arrays)

# Right: Visualize as function on embedded circle
ax2 = axes[1]
circle_x = np.cos(theta_plot)  # np.cos() computes cosine (element-wise for arrays)
circle_y = np.sin(theta_plot)  # np.sin() computes sine (element-wise for arrays)

# Color by function value
scatter = ax2.scatter(circle_x, circle_y, c=heights, cmap='coolwarm',
                     s=50, vmin=-1, vmax=1)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Height Function on Circle (colored by value)')
ax2.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='f(θ) = sin(θ)')  # plt.colorbar() adds color scale bar to plot

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🔬 Explore: Checking Smoothness of Transition Maps

```python
# For stereographic projection on sphere
def stereographic_from_north(x, y, z):
    """
    Chart covering sphere minus north pole
    Projects from north pole onto xy-plane
    """
    if z >= 1 - 1e-10:
        raise ValueError("North pole not in domain")
    X = x / (1 - z)
    Y = y / (1 - z)
    return (X, Y)

def stereographic_from_south(x, y, z):
    """
    Chart covering sphere minus south pole
    Projects from south pole onto xy-plane
    """
    if z <= -1 + 1e-10:
        raise ValueError("South pole not in domain")
    X = x / (1 + z)
    Y = y / (1 + z)
    return (X, Y)

# Test on equator point (in both chart domains)
equator_point = (1, 0, 0)  # Point on sphere

try:
    coords_north = stereographic_from_north(*equator_point)
    coords_south = stereographic_from_south(*equator_point)

    print("Equator point:", equator_point)
    print(f"North chart coordinates: {coords_north}")
    print(f"South chart coordinates: {coords_south}")
    print("\nBoth charts cover this point ✓")
except ValueError as e:
    print(f"Error: {e}")
```

---

## 4. Tangent Vectors and Tangent Spaces

### 📖 Concept

On a manifold, we can't just subtract points to get vectors (like in ℝⁿ). Instead, we define tangent vectors as **directional derivatives**.

**Tangent vector** at point p ∈ M is an operator that:
1. Takes a smooth function f: M → ℝ
2. Returns the directional derivative: a number
3. Satisfies linearity and Leibniz rule

**Tangent space** T_p M is the vector space of all tangent vectors at point p.
- For an n-dimensional manifold, T_p M ≅ ℝⁿ
- Different points have different tangent spaces!

**Coordinate basis:** In a chart with coordinates (x¹, x², ..., xⁿ), the partial derivatives ∂/∂x^i form a basis for T_p M.

**Key properties:**
- Tangent vectors ARE directional derivatives
- Components transform as derivatives (contravariant)
- The tangent space is where velocity vectors live

**Physical interpretation:**
- Velocity of a particle at point p is a tangent vector in T_p M
- In GR: 4-velocity is a tangent vector in tangent space of spacetime

**Why this matters for GR:** Tangent spaces are where tensors live! The metric tensor g at point p is a map T_p M × T_p M → ℝ. Understanding tangent spaces is essential for tensor calculus.

---

### 💻 Code Example: Tangent Vectors on Circle

```python
# Tangent vectors as directional derivatives
class TangentVector:
    """
    Tangent vector on circle at angle theta
    Represents d/dθ (derivative operator)
    """

    def __init__(self, theta, component):
        """
        theta: point on circle
        component: coefficient of d/dθ basis vector
        """
        self.theta = theta
        self.component = component

    def apply_to(self, function, epsilon=1e-7):
        """
        Apply directional derivative to a function
        Approximates df/dθ using finite differences
        """
        f_plus = function(self.theta + epsilon)
        f_minus = function(self.theta - epsilon)
        derivative = (f_plus - f_minus) / (2 * epsilon)
        return self.component * derivative

    def __repr__(self):
        return f"{self.component:.2f} d/dθ at θ={self.theta:.3f}"

# Test on circle
def height_function(theta):
    return np.sin(theta)  # np.sin() computes sine (element-wise for arrays)

def x_coordinate(theta):
    return np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)

# Tangent vector at θ = 0 with component = 2
v = TangentVector(theta=0, component=2.0)

print(f"Tangent vector: {v}")
print(f"Applied to f(θ)=sin(θ): {v.apply_to(height_function):.4f}")
print(f"Expected: 2 × cos(0) = 2 × 1 = 2.0")
print()
print(f"Applied to g(θ)=cos(θ): {v.apply_to(x_coordinate):.4f}")
print(f"Expected: 2 × (-sin(0)) = 2 × 0 = 0.0")
```

---

### 📊 Visualization: Tangent Spaces on Circle

```python
fig = plt.figure(figsize=(12, 10))  # plt.figure() creates a new figure for plotting

# Main plot: Circle with tangent spaces
ax = fig.add_subplot(111)

# Draw circle
theta_full = np.linspace(0, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
circle_x = np.cos(theta_full)  # np.cos() computes cosine (element-wise for arrays)
circle_y = np.sin(theta_full)  # np.sin() computes sine (element-wise for arrays)
ax.plot(circle_x, circle_y, color=COLORS['blue'], linewidth=3, label='Circle S¹')

# Draw tangent spaces at several points
sample_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]

for theta in sample_angles:
    # Point on circle
    px = np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
    py = np.sin(theta)  # np.sin() computes sine (element-wise for arrays)

    # Tangent vector direction (perpendicular to radius)
    tx = -np.sin(theta)  # np.sin() computes sine (element-wise for arrays)
    ty = np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)

    # Draw tangent line (representing tangent space T_p S¹)
    scale = 0.5
    t_vals = np.linspace(-scale, scale, 2)  # np.linspace() creates evenly spaced array between start and end
    tangent_x = px + t_vals * tx
    tangent_y = py + t_vals * ty

    ax.plot(tangent_x, tangent_y, color=COLORS['orange'], linewidth=2, alpha=0.7)

    # Draw arrow for basis vector d/dθ
    arrow_scale = 0.3
    ax.arrow(px, py, arrow_scale*tx, arrow_scale*ty,
            head_width=0.08, head_length=0.08,
            fc=COLORS['red'], ec=COLORS['red'], linewidth=2)

    # Label point
    ax.plot(px, py, 'o', color=COLORS['blue'], markersize=8)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLORS['blue'], linewidth=3, label='Manifold S¹'),
    Line2D([0], [0], color=COLORS['orange'], linewidth=2, label='Tangent space T_p S¹'),
    Line2D([0], [0], color=COLORS['red'], linewidth=2, label='Basis vector ∂/∂θ')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Tangent Spaces at Different Points on Circle', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.set_xlim([-1.8, 1.8])
ax.set_ylim([-1.8, 1.8])

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Each orange line is a tangent space T_p S¹ (1-dimensional)")
print("Red arrows show the basis vector ∂/∂θ at each point")
```

---

### 🎯 Practice Question #3

**Q:** What is the dimension of the tangent space T_p S² at a point p on the 2-sphere?

<details>
<summary>💡 Hint</summary>

The tangent space dimension equals the manifold dimension.
</details>

<details>
<summary>✅ Answer</summary>

**2 dimensions**

The tangent space T_p S² is 2-dimensional because S² is a 2-dimensional manifold. At any point on the sphere:
- The tangent space is the plane tangent to the sphere at that point
- You need 2 coordinates to specify a direction in that plane
- Basis vectors: ∂/∂u and ∂/∂v (in spherical coordinates)

Even though the sphere sits in 3D space ℝ³, it's intrinsically 2-dimensional, so its tangent space is also 2D.

```python
# On sphere with coordinates (u, v):
# T_p S² has basis {∂/∂u, ∂/∂v}
# Any tangent vector: v = a ∂/∂u + b ∂/∂v
# where a, b ∈ ℝ (2 components = 2D space)
```
</details>

---

## 5. Vector Fields

### 📖 Concept

A **vector field** assigns a tangent vector to each point on the manifold.

**Formal definition:** A vector field X on manifold M is a smooth assignment:
```
X: M → TM
```
where TM = ∪_p T_p M is the **tangent bundle** (union of all tangent spaces).

**In coordinates:** If (x¹, ..., xⁿ) are local coordinates, a vector field is:
```
X = X¹ ∂/∂x¹ + X² ∂/∂x² + ... + Xⁿ ∂/∂xⁿ
```
where X^i are smooth functions giving components.

**Physical examples:**
- Velocity field in fluid dynamics: v(x) at each point x
- Electromagnetic field (somewhat - actually needs 1-forms)
- Wind patterns on Earth's surface
- Killing vector fields (symmetries in GR)

**Operations with vector fields:**
- Add: (X + Y)_p = X_p + Y_p
- Scalar multiply: (fX)_p = f(p) X_p
- Lie bracket: [X, Y] measures non-commutativity

**Why this matters for GR:** Vector fields represent physical quantities like velocity, momentum. Killing vector fields correspond to spacetime symmetries (energy and momentum conservation).

---

### 💻 Code Example: Vector Field on Circle

```python
# Vector field on circle
class VectorFieldOnCircle:
    """Vector field X = f(θ) d/dθ on S¹"""

    def __init__(self, component_function):
        """
        component_function: function θ → coefficient
        Defines X = f(θ) d/dθ
        """
        self.f = component_function

    def at_point(self, theta):
        """Get the tangent vector at specific point"""
        return TangentVector(theta, self.f(theta))

    def flow(self, theta_0, time_steps, dt=0.01):
        """
        Follow integral curve (flow) of vector field
        Starting from theta_0, follow d(theta)/dt = f(theta)
        """
        theta = theta_0
        trajectory = [theta]

        for _ in range(time_steps):
            theta = theta + self.f(theta) * dt
            theta = theta % (2*np.pi)  # Keep in [0, 2π)
            trajectory.append(theta)

        return np.array(trajectory)  # np.array() converts Python list/tuple to efficient numpy array

# Example: constant vector field X = 1 ∂/∂θ
constant_field = VectorFieldOnCircle(lambda theta: 1.0)

# Example: position-dependent field X = cos(θ) ∂/∂θ
cosine_field = VectorFieldOnCircle(lambda theta: np.cos(theta))  # np.cos() computes cosine (element-wise for arrays)

# Test
print("Constant vector field X = 1 ∂/∂θ")
for theta in [0, np.pi/4, np.pi/2]:
    v = constant_field.at_point(theta)
    print(f"  At θ={theta:.3f}: {v}")

print("\nVariable vector field X = cos(θ) ∂/∂θ")
for theta in [0, np.pi/4, np.pi/2]:
    v = cosine_field.at_point(theta)
    print(f"  At θ={theta:.3f}: {v}")
```

---

### 📊 Visualization: Vector Fields on Circle

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Constant vector field
ax1 = axes[0]
theta_sample = np.linspace(0, 2*np.pi, 16, endpoint=False)  # np.linspace() creates evenly spaced array between start and end

for theta in theta_sample:
    px = np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
    py = np.sin(theta)  # np.sin() computes sine (element-wise for arrays)

    # Tangent direction
    tx = -np.sin(theta)  # np.sin() computes sine (element-wise for arrays)
    ty = np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)

    # Constant magnitude
    mag = 0.3
    ax1.arrow(px, py, mag*tx, mag*ty,
             head_width=0.1, head_length=0.1,
             fc=COLORS['orange'], ec=COLORS['orange'], linewidth=1.5)

# Draw circle
circle_theta = np.linspace(0, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
ax1.plot(np.cos(circle_theta), np.sin(circle_theta),  # np.sin() computes sine (element-wise for arrays)
        color=COLORS['blue'], linewidth=2)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Constant Vector Field: X = ∂/∂θ')
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-1.8, 1.8])
ax1.set_ylim([-1.8, 1.8])

# Right: Variable vector field
ax2 = axes[1]

for theta in theta_sample:
    px = np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
    py = np.sin(theta)  # np.sin() computes sine (element-wise for arrays)

    # Tangent direction
    tx = -np.sin(theta)  # np.sin() computes sine (element-wise for arrays)
    ty = np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)

    # Variable magnitude: cos(θ)
    mag = 0.5 * np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)

    if abs(mag) > 0.05:  # Only draw if significant
        ax2.arrow(px, py, mag*tx, mag*ty,
                 head_width=0.1, head_length=0.1,
                 fc=COLORS['red'], ec=COLORS['red'], linewidth=1.5)

ax2.plot(np.cos(circle_theta), np.sin(circle_theta),  # np.sin() computes sine (element-wise for arrays)
        color=COLORS['blue'], linewidth=2)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Variable Vector Field: X = cos(θ) ∂/∂θ')
ax2.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-1.8, 1.8])
ax2.set_ylim([-1.8, 1.8])

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Left: Constant field - all vectors same length")
print("Right: Variable field - vector length depends on position")
```

---

### 🔬 Explore: Integral Curves (Flow Lines)

```python
# Visualize flow of vector field
fig = plt.figure(figsize=(10, 8))  # plt.figure() creates a new figure for plotting
ax = fig.gca()

# Draw circle
circle_theta = np.linspace(0, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
ax.plot(np.cos(circle_theta), np.sin(circle_theta),  # np.sin() computes sine (element-wise for arrays)
       color=COLORS['blue'], linewidth=2, label='Circle S¹')

# Flow several curves for constant field X = 1 ∂/∂θ
constant_field = VectorFieldOnCircle(lambda theta: 1.0)

initial_angles = [0, np.pi/6, np.pi/3, np.pi/2]
colors_flow = [COLORS['orange'], COLORS['green'], COLORS['red'], COLORS['purple']]

for theta_0, color in zip(initial_angles, colors_flow):
    # Compute flow
    trajectory = constant_field.flow(theta_0, time_steps=200, dt=0.05)

    # Convert to x, y
    flow_x = np.cos(trajectory)  # np.cos() computes cosine (element-wise for arrays)
    flow_y = np.sin(trajectory)  # np.sin() computes sine (element-wise for arrays)

    # Plot with fading
    ax.plot(flow_x, flow_y, color=color, linewidth=2, alpha=0.6,
           label=f'Flow from θ={theta_0:.2f}')

    # Mark start
    ax.plot(np.cos(theta_0), np.sin(theta_0), 'o',  # np.sin() computes sine (element-wise for arrays)
           color=color, markersize=10)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Integral Curves of Vector Field X = ∂/∂θ', fontsize=14)
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.grid(True, alpha=0.3)
ax.legend()

plt.show()  # plt.show() displays the figure window

print("Integral curves wind around the circle at constant speed")
```

---

## 6. The Cotangent Space and One-Forms

### 📖 Concept

The **cotangent space** T*_p M is the dual space to the tangent space T_p M.

**One-form** (covector) at point p: A linear map α: T_p M → ℝ
- Takes a tangent vector as input
- Outputs a real number
- Linear: α(av + bw) = aα(v) + bα(w)

**Coordinate basis for T*_p M:** In coordinates (x¹, ..., xⁿ), the basis is {dx¹, ..., dxⁿ} where:
```
dx^i(∂/∂x^j) = δ^i_j  (Kronecker delta)
```

**Why two types of vectors?**
- **Tangent vectors (contravariant):** Transform like derivatives (upper indices)
- **Cotangent vectors (covariant):** Transform like gradients (lower indices)
- They transform oppositely under coordinate changes!

**The differential:** For smooth function f: M → ℝ, the differential df is a one-form:
```
df(v) = v[f]  (directional derivative of f along v)
```

**Physical examples:**
- Gradient of a function (temperature, potential energy)
- Momentum (dual to velocity)
- Electric potential (leads to E field)

**Why this matters for GR:** The metric tensor has components g_μν (two lower indices) - it takes two tangent vectors and outputs a number (the inner product). One-forms are essential for defining tensors.

---

### 💻 Code Example: One-Forms on Circle

```python
# One-form on circle
class OneFormOnCircle:
    """One-form ω = f(θ) dθ on S¹"""

    def __init__(self, component_function):
        """
        component_function: function θ → coefficient
        Defines ω = f(θ) dθ
        """
        self.f = component_function

    def evaluate(self, tangent_vector):
        """
        Apply one-form to tangent vector
        ω(v) where v = c ∂/∂θ
        Result: f(θ) × c
        """
        theta = tangent_vector.theta
        component = tangent_vector.component
        return self.f(theta) * component

    def __repr__(self):
        return f"ω = f(θ) dθ"

# Example: differential of height function
# If f(θ) = sin(θ), then df = cos(θ) dθ
def height(theta):
    return np.sin(theta)  # np.sin() computes sine (element-wise for arrays)

df = OneFormOnCircle(lambda theta: np.cos(theta))  # np.cos() computes cosine (element-wise for arrays)

# Test: df(∂/∂θ) should give cos(θ)
test_angles = [0, np.pi/4, np.pi/2, np.pi]

print("One-form df where f(θ) = sin(θ)")
print("df = cos(θ) dθ")
print()

for theta in test_angles:
    # Basis tangent vector ∂/∂θ at this point
    basis_vector = TangentVector(theta, component=1.0)

    result = df.evaluate(basis_vector)
    expected = np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)

    print(f"θ = {theta:.3f}:")
    print(f"  df(∂/∂θ) = {result:.4f}")
    print(f"  Expected cos(θ) = {expected:.4f}")
    print(f"  Match: {np.isclose(result, expected)}")  # np.isclose() tests if values are approximately equal (handles floating point)
    print()
```

---

### 📊 Visualization: One-Forms as Stacks of Level Surfaces

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Function on circle
ax1 = axes[0]
theta_vals = np.linspace(0, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end
function_vals = np.sin(theta_vals)  # np.sin() computes sine (element-wise for arrays)

ax1.plot(theta_vals, function_vals, color=COLORS['blue'], linewidth=3,
        label='f(θ) = sin(θ)')
ax1.set_xlabel('θ', fontsize=12)
ax1.set_ylabel('f(θ)', fontsize=12)
ax1.set_title('Function on Circle', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='k', linewidth=0.5)
ax1.legend()

# Mark level sets
levels = [-0.5, 0, 0.5, 1.0]
for level in levels:
    ax1.axhline(level, color=COLORS['orange'], linestyle='--', alpha=0.5)
    # Find where function equals this level
    idx = np.where(np.abs(function_vals - level) < 0.05)[0]
    if len(idx) > 0:
        ax1.plot(theta_vals[idx], function_vals[idx], 'o',
                color=COLORS['red'], markersize=6)

# Right: One-form as gradient (spacing of level sets)
ax2 = axes[1]
gradient_vals = np.cos(theta_vals)  # df/dθ = cos(θ)

ax2.plot(theta_vals, gradient_vals, color=COLORS['green'], linewidth=3,
        label='df = cos(θ) dθ')
ax2.set_xlabel('θ', fontsize=12)
ax2.set_ylabel('df/dθ', fontsize=12)
ax2.set_title('One-Form (Differential)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='k', linewidth=0.5)
ax2.legend()

# Shade regions where gradient is large
ax2.fill_between(theta_vals, 0, gradient_vals,
                where=(gradient_vals > 0), alpha=0.2, color=COLORS['green'],
                label='Steep increase')
ax2.fill_between(theta_vals, 0, gradient_vals,
                where=(gradient_vals < 0), alpha=0.2, color=COLORS['red'],
                label='Steep decrease')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Left: Level sets of function f")
print("Right: One-form df measures spacing between level sets")
print("Where df is large: level sets are closely packed (steep slope)")
```

---

### 🎯 Practice Question #4

**Q:** If v = 3 ∂/∂θ is a tangent vector and ω = 2 dθ is a one-form, what is ω(v)?

<details>
<summary>💡 Hint</summary>

One-forms eat tangent vectors and output numbers. Use dθ(∂/∂θ) = 1.
</details>

<details>
<summary>✅ Answer</summary>

ω(v) = (2 dθ)(3 ∂/∂θ) = 2 × 3 × dθ(∂/∂θ) = 2 × 3 × 1 = **6**

```python
# In code:
v = TangentVector(theta=0, component=3.0)
omega = OneFormOnCircle(lambda theta: 2.0)

result = omega.evaluate(v)
print(result)  # 6.0
```

The one-form ω linearly maps the tangent vector v to the real number 6.
</details>

---

## 7. Examples: Manifolds in Physics

### 📖 Concept

Manifolds appear throughout physics:

**1. Configuration Space (Classical Mechanics)**
- Manifold of all possible positions of a system
- Dimension = degrees of freedom
- Example: Double pendulum → S¹ × S¹ (torus)

**2. Phase Space**
- Manifold of positions AND momenta
- Dimension = 2 × (degrees of freedom)
- Example: Particle in 1D → ℝ² (position, momentum)

**3. Spacetime (Relativity)**
- Special Relativity: Minkowski space ℝ^(3,1)
- General Relativity: 4D Lorentzian manifold
- Each event is a point; tangent vectors are 4-velocities

**4. Field Configuration Space**
- Infinite-dimensional manifold!
- Each point = one configuration of the field
- Used in quantum field theory

**Why manifolds?**
- Encode constraints naturally (particle on sphere → sphere manifold)
- Coordinate-independent formulation of physics
- Essential for curved spacetime in GR

**Why this matters for GR:** Spacetime is a 4D pseudo-Riemannian manifold with metric signature (-,+,+,+) or (+,-,-,-). Matter curves this manifold, and geodesics in the curved manifold are particle trajectories.

---

### 💻 Code Example: Configuration Space of Double Pendulum

```python
# Double pendulum configuration space = T² (2-torus)
def double_pendulum_config_space(theta1, theta2):
    """
    Configuration specified by two angles
    (theta1, theta2) ∈ S¹ × S¹ = T²
    """
    return (theta1 % (2*np.pi), theta2 % (2*np.pi))

# Visualize torus as configuration space
def visualize_torus_config_space():
    """
    Torus embedded in ℝ³ represents all configurations
    Each point (theta1, theta2) maps to a point on torus
    """
    theta1_vals = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
    theta2_vals = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
    T1, T2 = np.meshgrid(theta1_vals, theta2_vals)  # np.meshgrid() creates coordinate matrices from coordinate vectors

    # Torus embedding
    R = 2.0  # Major radius
    r = 0.5  # Minor radius
    X = (R + r*np.cos(T1)) * np.cos(T2)  # np.cos() computes cosine (element-wise for arrays)
    Y = (R + r*np.cos(T1)) * np.sin(T2)  # np.sin() computes sine (element-wise for arrays)
    Z = r * np.sin(T1)  # np.sin() computes sine (element-wise for arrays)

    fig = plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

    # Left: Torus
    ax1 = fig.add_subplot(121, projection='3d')  projection='3d'  # Create 3D axes
    ax1.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')  # .plot_surface() draws 3D surface plot
    ax1.set_title('Configuration Space T² = S¹ × S¹')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # Mark some sample configurations
    sample_configs = [(0, 0), (np.pi/2, 0), (0, np.pi/2), (np.pi/2, np.pi/2)]
    for t1, t2 in sample_configs:
        x = (R + r*np.cos(t1)) * np.cos(t2)  # np.cos() computes cosine (element-wise for arrays)
        y = (R + r*np.cos(t1)) * np.sin(t2)  # np.sin() computes sine (element-wise for arrays)
        z = r * np.sin(t1)  # np.sin() computes sine (element-wise for arrays)
        ax1.scatter([x], [y], [z], color=COLORS['red'], s=100)

    # Right: Square with periodic boundaries
    ax2 = fig.add_subplot(122)
    ax2.set_xlim([0, 2*np.pi])
    ax2.set_ylim([0, 2*np.pi])

    # Plot sample configurations
    for t1, t2 in sample_configs:
        ax2.plot(t1, t2, 'o', color=COLORS['red'], markersize=12)
        ax2.text(t1, t2, f'  ({t1:.2f}, {t2:.2f})',
                fontsize=9, verticalalignment='bottom')

    # Show periodic boundaries
    ax2.plot([0, 0], [0, 2*np.pi], color=COLORS['blue'], linewidth=3)
    ax2.plot([2*np.pi, 2*np.pi], [0, 2*np.pi], color=COLORS['blue'],
            linewidth=3, linestyle='--', label='Identified edges')
    ax2.plot([0, 2*np.pi], [0, 0], color=COLORS['green'], linewidth=3)
    ax2.plot([0, 2*np.pi], [2*np.pi, 2*np.pi], color=COLORS['green'],
            linewidth=3, linestyle='--', label='Identified edges')

    ax2.set_xlabel('θ₁ (first pendulum angle)', fontsize=11)
    ax2.set_ylabel('θ₂ (second pendulum angle)', fontsize=11)
    ax2.set_title('Fundamental Domain [0, 2π] × [0, 2π]')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
    plt.show()  # plt.show() displays the figure window

visualize_torus_config_space()

print("Double pendulum has 2 degrees of freedom")
print("Configuration space = T² (torus)")
print("Each point on torus = one configuration of the system")
```

---

### 📊 Visualization: Spacetime as Manifold

```python
# 2D slice of spacetime (1 space + 1 time dimension)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Flat Minkowski spacetime
ax1 = axes[0]
t_vals = np.linspace(-2, 2, 20)  # np.linspace() creates evenly spaced array between start and end
x_vals = np.linspace(-2, 2, 20)  # np.linspace() creates evenly spaced array between start and end

# Draw grid
for t in t_vals:
    ax1.plot(x_vals, [t]*len(x_vals), color=COLORS['blue'], linewidth=0.5, alpha=0.5)
for x in x_vals:
    ax1.plot([x]*len(t_vals), t_vals, color=COLORS['blue'], linewidth=0.5, alpha=0.5)

# Draw light cones at origin
t_cone = np.linspace(0, 2, 50)  # np.linspace() creates evenly spaced array between start and end
ax1.plot(t_cone, t_cone, color=COLORS['orange'], linewidth=3, label='Light cone')
ax1.plot(-t_cone, t_cone, color=COLORS['orange'], linewidth=3)
ax1.fill_between(t_cone, -t_cone, t_cone, alpha=0.2, color=COLORS['yellow'])

# Worldline of stationary particle
ax1.plot([0]*len(t_vals), t_vals, color=COLORS['red'], linewidth=3,
        label='Stationary particle')

# Worldline of moving particle
v = 0.5  # velocity
worldline_t = np.linspace(-2, 2, 50)  # np.linspace() creates evenly spaced array between start and end
worldline_x = v * worldline_t
ax1.plot(worldline_x, worldline_t, color=COLORS['green'], linewidth=3,
        label=f'Particle v={v}c')

ax1.set_xlabel('x (space)', fontsize=12)
ax1.set_ylabel('t (time)', fontsize=12)
ax1.set_title('Flat Minkowski Spacetime M^(1,1)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Right: Curved spacetime near mass
ax2 = axes[1]

# Create curved grid (simplified illustration)
for i, t in enumerate(t_vals):
    x_curved = x_vals.copy()
    t_curved = np.ones_like(x_vals) * t

    # Warp space near x=0
    for j, x in enumerate(x_vals):
        warp_factor = 1 - 0.3 * np.exp(-(x**2 + t**2))  # np.exp() computes exponential e^x
        x_curved[j] = x * warp_factor

    ax2.plot(x_curved, t_curved, color=COLORS['blue'],
            linewidth=0.5, alpha=0.5)

for i, x in enumerate(x_vals):
    t_curved = t_vals.copy()
    x_curved = np.ones_like(t_vals) * x

    for j, t in enumerate(t_vals):
        warp_factor = 1 - 0.3 * np.exp(-(x**2 + t**2))  # np.exp() computes exponential e^x
        x_curved[j] = x * warp_factor

    ax2.plot(x_curved, t_curved, color=COLORS['blue'],
            linewidth=0.5, alpha=0.5)

# Mark "mass" location
ax2.plot(0, 0, 'o', color=COLORS['red'], markersize=20, label='Mass')

ax2.set_xlabel('x (space)', fontsize=12)
ax2.set_ylabel('t (time)', fontsize=12)
ax2.set_title('Curved Spacetime Near Mass', fontsize=14)
ax2.legend()
ax2.grid(False)
ax2.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Spacetime in GR is a 4D manifold with curvature")
print("Left: Flat spacetime (special relativity)")
print("Right: Curved spacetime (general relativity)")
```

---

### 🔬 Explore: Lie Groups as Manifolds

```python
# SO(2): rotation group = circle S¹
def rotation_matrix(theta):
    """2D rotation by angle theta"""
    return np.array([  # np.array() converts Python list/tuple to efficient numpy array
        [np.cos(theta), -np.sin(theta)],  # np.sin() computes sine (element-wise for arrays)
        [np.sin(theta), np.cos(theta)]  # np.sin() computes sine (element-wise for arrays)
    ])

# SO(2) is a 1-parameter Lie group and a manifold
print("SO(2) - Rotation Group")
print("=" * 40)

angles = [0, np.pi/4, np.pi/2, np.pi]
for theta in angles:
    R = rotation_matrix(theta)
    print(f"\nRotation by θ = {theta:.3f}:")
    print(R)

print("\n" + "=" * 40)
print("SO(2) ≅ S¹ (circle) as a manifold")
print("Dimension: 1")
print("Each point = one rotation")
print("Group operation: R(θ₁) ∘ R(θ₂) = R(θ₁ + θ₂)")
```

---

### 🎯 Practice Question #5

**Q:** What is the dimension of the configuration space manifold for a system of N particles moving freely in 3D space?

<details>
<summary>💡 Hint</summary>

How many coordinates do you need to specify all positions?
</details>

<details>
<summary>✅ Answer</summary>

**3N dimensions**

Each particle needs 3 coordinates (x, y, z) to specify its position in 3D space. For N particles, you need 3N coordinates total.

The configuration space is ℝ^(3N) - a 3N-dimensional manifold.

Example:
- 1 particle: ℝ³ (3D)
- 2 particles: ℝ⁶ (6D)
- 3 particles: ℝ⁹ (9D)

If there are constraints (like particles connected by rigid rods), the manifold has lower dimension.
</details>

---

## Practice Questions

Test your understanding:

### Manifolds and Charts

1. **Dimensions:** What is the dimension of the Klein bottle as a manifold?

2. **Charts:** Why do you need at least 6 charts to cover the 3-torus T³?

3. **Embedding:** Can every n-dimensional manifold be embedded in ℝⁿ?

### Tangent Spaces and Vector Fields

4. **Tangent space:** What is the tangent space to ℝⁿ at any point?

5. **Vector fields:** Give an example of a vector field on ℝ² that vanishes at the origin.

6. **Coordinate basis:** In coordinates (r, θ) on ℝ²\{0}, what are the basis vectors ∂/∂r and ∂/∂θ?

### One-Forms and Duals

7. **Pairing:** If df = 3dx¹ + 2dx² and v = 4∂/∂x¹ + ∂/∂x², what is df(v)?

8. **Differential:** For f(x,y) = x² + y², what is the one-form df?

---

## Summary and Next Steps

**Key Concepts Mastered:**
- Manifolds as spaces locally like ℝⁿ
- Charts and atlases for coordinate systems
- Smooth (differentiable) manifolds
- Tangent spaces and tangent vectors
- Vector fields on manifolds
- Cotangent spaces and one-forms
- Physical examples (configuration space, spacetime)

**Connection to GR:**
- Spacetime is a 4D manifold
- Tangent spaces → where 4-velocity lives
- One-forms → energy-momentum covectors
- Coordinate transformations → change of reference frame

**Ready for next steps:**
- Lesson 6: Tensors (combines vectors and one-forms)
- Lesson 7: Riemannian geometry (metric on manifolds)
- Lesson 10: General Relativity foundations

---

**Need Help?** Use the AI assistant:
```python
from utils.ai_assistant import AIAssistant
assistant = AIAssistant()
assistant.set_lesson_context("Lesson 5", "Manifolds",
                            ["tangent space", "charts", "one-forms"])
assistant.ask("Why do tangent and cotangent vectors transform differently?")
```

---

**Continue to:** → [Lesson 6: Tensors](../06_tensors/LESSON.md)
