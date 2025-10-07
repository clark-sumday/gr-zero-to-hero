# Lesson 1: Linear Algebra Foundations

**Topics:** Vectors, Matrices, Transformations, Eigenvalues/Eigenvectors
**Prerequisites:** Basic algebra, coordinate geometry
**Time:** ~2-3 hours

---

## Table of Contents

1. [Introduction to Vectors](#1-introduction-to-vectors)
2. [Linear Combinations, Span, and Basis](#2-linear-combinations-span-and-basis)
3. [Dot Product and Inner Products](#3-dot-product-and-inner-products)
4. [Cross Product and Rotation](#4-cross-product-and-rotation)
5. [Projection and Orthogonality](#5-projection-and-orthogonality)
6. [Matrices and Linear Transformations](#6-matrices-and-linear-transformations)
7. [Eigenvalues and Eigenvectors](#7-eigenvalues-and-eigenvectors)
8. [Determinants and Matrix Properties](#8-determinants-and-matrix-properties)
9. [Matrix Inverses](#9-matrix-inverses)
10. [Practice Questions](#practice-questions)

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

## 1. Introduction to Vectors

### 📖 Concept

A **vector** is a mathematical object with both magnitude and direction. In physics, vectors represent quantities like velocity, force, and displacement.

In n-dimensional space, a vector can be written as:
```
v = (v₁, v₂, ..., vₙ)
```

For General Relativity, we'll work extensively with 4-dimensional spacetime vectors. But let's start with familiar 2D and 3D vectors.

**Key Properties:**
- **Vector Addition:** `u + v = (u₁+v₁, u₂+v₂, ...)`
- **Scalar Multiplication:** `c·v = (c·v₁, c·v₂, ...)`
- **Magnitude:** `|v| = √(v₁² + v₂² + ... + vₙ²)`

---

### 💻 Code Example: Basic Operations

Copy this into your Python terminal:

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations

# Create vectors as NumPy arrays (efficient numerical data structures)
u = np.array([3, 1])  # np.array() converts Python list to numpy array
v = np.array([1, 3])  # Allows element-wise operations

print(f"u = {u}")
print(f"v = {v}")
print(f"u + v = {u + v}")  # Element-wise addition: [3+1, 1+3] = [4, 4]
print(f"2·u = {2 * u}")    # Scalar multiplication: [2*3, 2*1] = [6, 2]
print(f"|u| = {np.linalg.norm(u)}")  # np.linalg.norm() computes vector magnitude (length)
```

**Expected output:**
```
u = [3 1]
v = [1 3]
u + v = [4 4]
2·u = [6 2]
|u| = 3.16
```

---

### 📊 Visualization: Vector Addition

Now let's see what this looks like geometrically:

```python
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')  # Add project to Python path
from utils.colorblind_colors import COLORS  # Import accessible color scheme

plt.figure(figsize=(10, 5))  # Create figure with width=10, height=5 inches

# Left: Standard view (all from origin)
plt.subplot(1, 2, 1)  # Create subplot: 1 row, 2 columns, position 1
# plt.quiver() draws arrows (vector visualization)
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
           color=COLORS['blue'], width=0.006, label='u')  # Draw vector u from origin
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
           color=COLORS['orange'], width=0.006, label='v')  # Draw vector v from origin
plt.quiver(0, 0, (u+v)[0], (u+v)[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
           color=COLORS['green'], width=0.006, label='u+v')  # Draw sum vector
# plt.text() adds text labels at specified coordinates
plt.text(u[0], u[1], ' u', fontsize=12, color=COLORS['blue'])  # plt.text() adds text annotation at specified coordinates
plt.text(v[0], v[1], ' v', fontsize=12, color=COLORS['orange'])  # plt.text() adds text annotation at specified coordinates
plt.text((u+v)[0], (u+v)[1], ' u+v', fontsize=12, color=COLORS['green'])  # plt.text() adds text annotation at specified coordinates
plt.xlim(-1, 5)  # Set x-axis limits
plt.ylim(-1, 7)  # Set y-axis limits
plt.grid(True, alpha=0.3)  # Add semi-transparent grid
plt.legend()  # Show legend with vector labels
plt.title("Vectors from Origin")  # plt.title() sets plot title
plt.xlabel("x")  # plt.xlabel() sets x-axis label
plt.ylabel("y")  # plt.ylabel() sets y-axis label
plt.axhline(y=0, color='k', linewidth=0.5)  # Draw horizontal axis
plt.axvline(x=0, color='k', linewidth=0.5)  # Draw vertical axis
plt.gca().set_aspect('equal')  # Equal scaling for both axes (geometric accuracy)

# Right: Tip-to-tail construction
plt.subplot(1, 2, 2)  # Create subplot: position 2
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
           color=COLORS['blue'], width=0.006, label='u')  # Draw u from origin
# Draw v starting from tip of u (tip-to-tail method)
plt.quiver(u[0], u[1], v[0], v[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
           color=COLORS['orange'], width=0.006, label='v (from u tip)')
plt.quiver(0, 0, (u+v)[0], (u+v)[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
           color=COLORS['green'], width=0.006, label='u+v')  # Resultant vector
# plt.plot() draws lines to show parallelogram construction
plt.plot([0, v[0]], [0, v[1]], '--', color=COLORS['gray'], alpha=0.5)  # Dashed line
plt.plot([u[0], (u+v)[0]], [u[1], (u+v)[1]], '--', color=COLORS['gray'], alpha=0.5)  # plt.plot() draws line plot
plt.text(u[0]/2, u[1]/2, ' u', fontsize=12, color=COLORS['blue'])  # plt.text() adds text annotation at specified coordinates
plt.text(u[0] + v[0]/2, u[1] + v[1]/2, ' v', fontsize=12, color=COLORS['orange'])  # plt.text() adds text annotation at specified coordinates
plt.text((u+v)[0], (u+v)[1], ' u+v', fontsize=12, color=COLORS['green'])  # plt.text() adds text annotation at specified coordinates
plt.xlim(-1, 5)  # plt.xlim() sets x-axis limits
plt.ylim(-1, 7)  # plt.ylim() sets y-axis limits
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels
plt.title("Tip-to-Tail (Parallelogram) Method")  # plt.title() sets plot title
plt.xlabel("x")  # plt.xlabel() sets x-axis label
plt.ylabel("y")  # plt.ylabel() sets y-axis label
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.axvline(x=0, color='k', linewidth=0.5)  # plt.axvline() draws vertical line across plot
plt.gca().set_aspect('equal')  # plt.gca() gets current axes object

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

**What you should see:** Two side-by-side plots:
- **Left:** All vectors starting from the origin
- **Right:** The "tip-to-tail" geometric construction showing how vector addition works

---

### 🔬 Explore on Your Own

Try these experiments before moving on:

**1. Different vectors:**
```python
u = np.array([2, 1])  # np.array() converts Python list/tuple to efficient numpy array
v = np.array([-1, 3])  # np.array() converts Python list/tuple to efficient numpy array
# Re-run the visualization code above
```

**2. Scalar multiplication:**
```python
plt.figure(figsize=(8, 6))  # plt.figure() creates a new figure for plotting
for scalar in [0.5, 1, 1.5, 2]:
    scaled = scalar * u
    plt.quiver(0, 0, scaled[0], scaled[1], angles='xy',  # plt.quiver() draws arrow/vector field plot
               scale_units='xy', scale=1, label=f'{scalar}·u')
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True)  # plt.grid() adds grid lines to plot
plt.title("Scalar Multiplication")  # plt.title() sets plot title
plt.gca().set_aspect('equal')  # plt.gca() gets current axes object
plt.show()  # plt.show() displays the figure window
```

**3. 3D vectors:**
```python
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit

u3d = np.array([1, 2, 3])  # np.array() converts Python list/tuple to efficient numpy array
v3d = np.array([2, -1, 1])  # np.array() converts Python list/tuple to efficient numpy array

fig = plt.figure(figsize=(8, 8))  # plt.figure() creates a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  projection='3d'  # Create 3D axes
ax.quiver(0, 0, 0, u3d[0], u3d[1], u3d[2],
          color=COLORS['blue'], label='u', arrow_length_ratio=0.15)
ax.quiver(0, 0, 0, v3d[0], v3d[1], v3d[2],
          color=COLORS['orange'], label='v', arrow_length_ratio=0.15)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.set_title("3D Vectors")
plt.show()  # plt.show() displays the figure window
```

---

### 🎯 Quick Check

Before moving on, make sure you can:
- [ ] Add two vectors component-wise
- [ ] Calculate vector magnitude using `np.linalg.norm()`
- [ ] Explain the geometric meaning of vector addition (tip-to-tail)
- [ ] Visualize vectors in 2D and 3D

---

## 2. Linear Combinations, Span, and Basis

### 📖 Concept

A **linear combination** of vectors v₁, v₂, ..., vₙ is:
```
c₁v₁ + c₂v₂ + ... + cₙvₙ
```
where c₁, c₂, ..., cₙ are scalars (numbers).

**Example:**
If v₁ = (1, 0) and v₂ = (0, 1), then 3v₁ + 2v₂ = (3, 2)

---

### Key Definitions

**SPAN:** The span of a set of vectors is *all possible linear combinations* of those vectors.
- Span of {(1,0), (0,1)} = all of ℝ² (entire 2D plane)
- Span of {(1,0)} = the x-axis (just a line)
- Span of {(1,2), (2,4)} = a line (since they're parallel)

**LINEAR INDEPENDENCE:** A set of vectors is linearly independent if *no vector can be written as a linear combination of the others*.
- {(1,0), (0,1)} are independent
- {(1,2), (2,4)} are dependent (second = 2 × first)

**BASIS:** A linearly independent set that spans the entire space.
- Standard basis for ℝ²: {(1,0), (0,1)}
- You need exactly n vectors for a basis of ℝⁿ
- Many different bases are possible!

**Why this matters for GR:** We need to choose coordinate bases in curved spacetime, and we need to transform between different coordinate systems!

---

### 💻 Code Example: Basis Vectors

```python
# Standard basis vectors
e1 = np.array([1, 0])  # np.array() converts Python list/tuple to efficient numpy array
e2 = np.array([0, 1])  # np.array() converts Python list/tuple to efficient numpy array

# Any vector can be written as linear combination
v = np.array([3, 2])  # np.array() converts Python list/tuple to efficient numpy array
print(f"Standard basis: e₁ = {e1}, e₂ = {e2}")
print(f"Vector v = {v}")
print(f"v = 3·e₁ + 2·e₂ = {3*e1 + 2*e2}")
print(f"Check: {np.array_equal(v, 3*e1 + 2*e2)}")
```

---

### 📊 Visualization: Linear Independence vs Dependence

```python
plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Left: Linearly INDEPENDENT vectors span plane
plt.subplot(1, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
v1 = np.array([1, 0])  # np.array() converts Python list/tuple to efficient numpy array
v2 = np.array([0.5, 1])  # np.array() converts Python list/tuple to efficient numpy array

# Show grid of linear combinations
for i in np.linspace(-2, 2, 5):  # np.linspace() creates evenly spaced array between start and end
    for j in np.linspace(-2, 2, 5):  # np.linspace() creates evenly spaced array between start and end
        point = i*v1 + j*v2
        plt.plot(point[0], point[1], 'o', color='lightblue', markersize=4)  # plt.plot() draws line plot

plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
           color=COLORS['blue'], width=0.01, label='v₁')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
           color=COLORS['orange'], width=0.01, label='v₂')
plt.text(v1[0], v1[1], ' v₁', fontsize=12, color=COLORS['blue'])  # plt.text() adds text annotation at specified coordinates
plt.text(v2[0], v2[1], ' v₂', fontsize=12, color=COLORS['orange'])  # plt.text() adds text annotation at specified coordinates
plt.xlim(-3, 3)  # plt.xlim() sets x-axis limits
plt.ylim(-3, 3)  # plt.ylim() sets y-axis limits
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels
plt.title("Linearly Independent: Span = ℝ²")  # plt.title() sets plot title
plt.gca().set_aspect('equal')  # plt.gca() gets current axes object

# Right: Linearly DEPENDENT vectors span only a line
plt.subplot(1, 2, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
v1 = np.array([1, 0.5])  # np.array() converts Python list/tuple to efficient numpy array
v2 = np.array([2, 1])  # v2 = 2*v1 (dependent!)

for t in np.linspace(-3, 3, 15):  # np.linspace() creates evenly spaced array between start and end
    point = t * v1
    plt.plot(point[0], point[1], 'o', color='lightcoral', markersize=4)  # plt.plot() draws line plot

plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
           color=COLORS['blue'], width=0.01, label='v₁')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
           color=COLORS['red'], width=0.01, label='v₂ = 2v₁')
plt.text(v1[0], v1[1], ' v₁', fontsize=12, color=COLORS['blue'],  # plt.text() adds text annotation at specified coordinates
         verticalalignment='top')
plt.text(v2[0], v2[1], ' v₂', fontsize=12, color=COLORS['red'],  # plt.text() adds text annotation at specified coordinates
         verticalalignment='bottom')
plt.xlim(-3, 3)  # plt.xlim() sets x-axis limits
plt.ylim(-3, 3)  # plt.ylim() sets y-axis limits
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels
plt.title("Linearly Dependent: Span = Line")  # plt.title() sets plot title
plt.gca().set_aspect('equal')  # plt.gca() gets current axes object

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

**What you should see:**
- **Left:** Blue dots fill the entire plane - two independent vectors can create any point
- **Right:** Red dots only on a line - dependent vectors are limited

---

### 🎯 Practice Question #1

**Q:** Are the vectors v₁ = (1, 2) and v₂ = (2, 4) linearly independent?

<details>
<summary>💡 Hint 1</summary>

Check if one is a scalar multiple of the other.
</details>

<details>
<summary>💡 Hint 2</summary>

If v₂ = c·v₁ for some scalar c, they're dependent.
</details>

<details>
<summary>✅ Answer</summary>

**No**, they are linearly dependent because v₂ = 2·v₁.

You can verify this in Python:
```python
v1 = np.array([1, 2])  # np.array() converts Python list/tuple to efficient numpy array
v2 = np.array([2, 4])  # np.array() converts Python list/tuple to efficient numpy array
print(v2 / v1)  # [2. 2.] - same scalar for both components
print(np.allclose(v2, 2*v1))  # True
```
</details>

---

### 💻 Advanced: Change of Basis

This is crucial for GR! When we change coordinate systems, the vector itself doesn't change, but its components do.

```python
# Standard basis
e1 = np.array([1, 0])  # np.array() converts Python list/tuple to efficient numpy array
e2 = np.array([0, 1])  # np.array() converts Python list/tuple to efficient numpy array

# Rotated basis (30 degrees counter-clockwise)
theta = np.pi / 6  # 30 degrees
e1_prime = np.array([np.cos(theta), np.sin(theta)])  # np.array() converts Python list/tuple to efficient numpy array
e2_prime = np.array([-np.sin(theta), np.cos(theta)])  # np.array() converts Python list/tuple to efficient numpy array

# Transformation matrix: columns are new basis vectors
T = np.column_stack([e1_prime, e2_prime])  # np.column_stack() stacks 1-D arrays as columns into 2-D array
print(f"Transformation matrix T:\n{T}")

# A vector in standard basis coordinates
v_standard = np.array([2.0, 1.0])  # np.array() converts Python list/tuple to efficient numpy array

# Same vector in rotated basis coordinates
v_rotated = np.linalg.inv(T) @ v_standard  # np.linalg.inv() computes matrix inverse

print(f"\nVector in standard basis: {v_standard}")
print(f"Same vector in rotated basis: {v_rotated}")
print(f"Transform back to standard: {T @ v_rotated}")    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
```

**Visualize it:**
```python
plt.figure(figsize=(10, 8))  # plt.figure() creates a new figure for plotting

# Plot standard basis (solid lines)
plt.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
          color=COLORS['blue'], width=0.008, label='Standard basis')
plt.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
          color=COLORS['blue'], width=0.008)
plt.text(e1[0], e1[1], ' e₁', fontsize=12, color=COLORS['blue'])  # plt.text() adds text annotation at specified coordinates
plt.text(e2[0], e2[1], ' e₂', fontsize=12, color=COLORS['blue'])  # plt.text() adds text annotation at specified coordinates

# Plot rotated basis (dashed lines)
plt.quiver(0, 0, e1_prime[0], e1_prime[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
          color=COLORS['orange'], width=0.008, linestyle='--',
          label="Rotated basis (30°)")
plt.quiver(0, 0, e2_prime[0], e2_prime[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
          color=COLORS['orange'], width=0.008, linestyle='--')
plt.text(e1_prime[0], e1_prime[1], " e'₁", fontsize=12, color=COLORS['orange'])  # plt.text() adds text annotation at specified coordinates
plt.text(e2_prime[0], e2_prime[1], " e'₂", fontsize=12, color=COLORS['orange'])  # plt.text() adds text annotation at specified coordinates

# Plot the vector (same geometric object in both bases)
plt.quiver(0, 0, v_standard[0], v_standard[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
          color=COLORS['green'], width=0.01, label='Vector v')
plt.text(v_standard[0], v_standard[1],  # plt.text() adds text annotation at specified coordinates
         f' v\nStd coords: ({v_standard[0]:.1f}, {v_standard[1]:.1f})\nRot coords: ({v_rotated[0]:.2f}, {v_rotated[1]:.2f})',
         fontsize=9, color=COLORS['green'], verticalalignment='bottom')

plt.xlim(-0.5, 3)  # plt.xlim() sets x-axis limits
plt.ylim(-0.5, 2.5)  # plt.ylim() sets y-axis limits
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels
plt.title("Change of Basis: Same Vector, Different Coordinates")  # plt.title() sets plot title
plt.xlabel("x")  # plt.xlabel() sets x-axis label
plt.ylabel("y")  # plt.ylabel() sets y-axis label
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.axvline(x=0, color='k', linewidth=0.5)  # plt.axvline() draws vertical line across plot
plt.gca().set_aspect('equal')  # plt.gca() gets current axes object
plt.show()  # plt.show() displays the figure window
```

---

## 3. Dot Product and Inner Products

### 📖 Concept

The **dot product** (also called inner product or scalar product) takes two vectors and produces a *scalar* (number):

```
u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ
```

**Geometric interpretation:**
```
u · v = |u| |v| cos(θ)
```
where θ is the angle between the vectors.

### Key Properties

- If `u · v = 0`, vectors are **orthogonal** (perpendicular), so θ = 90°
- If `u · v > 0`, angle is acute (θ < 90°)
- If `u · v < 0`, angle is obtuse (θ > 90°)
- `|v|² = v · v` (magnitude squared)

**Why this matters for GR:** The dot product generalizes to the **metric tensor** g_μν in General Relativity, which defines distances, angles, and the geometry of curved spacetime itself!

---

### 💻 Code Example: Computing Dot Products

```python
u = np.array([3, 4])  # np.array() converts Python list/tuple to efficient numpy array
v = np.array([4, -3])  # np.array() converts Python list/tuple to efficient numpy array

dot_product = np.dot(u, v)  # np.dot() computes dot product of two arrays
print(f"u = {u}")
print(f"v = {v}")
print(f"u · v = {dot_product}")
print(f"|u| = {np.linalg.norm(u):.2f}")  # np.linalg.norm() computes vector magnitude (Euclidean norm)
print(f"|v| = {np.linalg.norm(v):.2f}")  # np.linalg.norm() computes vector magnitude (Euclidean norm)

# Calculate angle between vectors
cos_theta = dot_product / (np.linalg.norm(u) * np.linalg.norm(v))  # np.linalg.norm() computes vector magnitude (Euclidean norm)
theta_rad = np.arccos(cos_theta)
theta_deg = np.degrees(theta_rad)

print(f"\nAngle between u and v: {theta_deg:.2f}°")
print(f"Are they orthogonal? {np.abs(dot_product) < 1e-10}")
```

**Expected output:**
```
u = [3 4]
v = [4 -3]
u · v = 0
|u| = 5.00
|v| = 5.00

Angle between u and v: 90.00°
Are they orthogonal? True
```

Notice: The dot product is zero, so these vectors are perpendicular!

---

### 🎯 Practice Question #2

**Q:** If u = (3, 4) and v = (5, 12), what is u · v?

<details>
<summary>💡 Hint</summary>

Use the formula: u · v = u₁v₁ + u₂v₂
</details>

<details>
<summary>✅ Answer</summary>

u · v = (3)(5) + (4)(12) = 15 + 48 = **63**

```python
u = np.array([3, 4])  # np.array() converts Python list/tuple to efficient numpy array
v = np.array([5, 12])  # np.array() converts Python list/tuple to efficient numpy array
print(np.dot(u, v))  # 63
```
</details>

---

### 🔬 Explore: Orthogonal Vectors

Try finding vectors orthogonal to a given vector:

```python
# Given vector
v = np.array([3, 4])  # np.array() converts Python list/tuple to efficient numpy array

# Find orthogonal vectors (satisfy u · v = 0)
# For 2D: if v = (a, b), then (-b, a) is orthogonal
u1 = np.array([-v[1], v[0]])  # np.array() converts Python list/tuple to efficient numpy array
u2 = np.array([v[1], -v[0]])  # np.array() converts Python list/tuple to efficient numpy array

print(f"v = {v}")
print(f"u1 = {u1}, dot product: {np.dot(v, u1)}")  # np.dot() computes dot product of two arrays
print(f"u2 = {u2}, dot product: {np.dot(v, u2)}")  # np.dot() computes dot product of two arrays

# Visualize
plt.figure(figsize=(8, 8))  # plt.figure() creates a new figure for plotting
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
          color=COLORS['blue'], width=0.01, label='v')
plt.quiver(0, 0, u1[0], u1[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
          color=COLORS['orange'], width=0.01, label='u₁ ⊥ v')
plt.quiver(0, 0, u2[0], u2[1], angles='xy', scale_units='xy', scale=1,  # plt.quiver() draws arrow/vector field plot
          color=COLORS['green'], width=0.01, label='u₂ ⊥ v')
plt.xlim(-6, 6)  # plt.xlim() sets x-axis limits
plt.ylim(-6, 6)  # plt.ylim() sets y-axis limits
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels
plt.title("Orthogonal Vectors")  # plt.title() sets plot title
plt.gca().set_aspect('equal')  # plt.gca() gets current axes object
plt.show()  # plt.show() displays the figure window
```

---

## 4. Cross Product and Rotation

### 📖 Concept

The **cross product** is defined **only in 3D** and produces a vector perpendicular to both inputs:

```
u × v = |u||v|sin(θ) n̂
```

where n̂ is a unit vector perpendicular to both u and v (determined by the right-hand rule).

**Component formula (3D only):**
```
u × v = (u₂v₃ - u₃v₂, u₃v₁ - u₁v₃, u₁v₂ - u₂v₁)
```

### Key Properties

- `u × v` is perpendicular to both u and v
- `|u × v|` = area of parallelogram formed by u and v
- `u × v = -(v × u)` (anti-commutative)
- If `u × v = 0`, vectors are parallel
- Right-hand rule determines direction

### Applications in Physics

- **Angular momentum:** L = r × p
- **Torque:** τ = r × F
- **Magnetic force:** F = q(v × B)
- **Curl in vector calculus**
- **In GR:** Levi-Civita symbol εᵢⱼₖ, rotation tensors

---

### 💻 Code Example: Cross Product

```python
# Define two 3D vectors
u = np.array([1, 0, 0])  # Along x-axis
v = np.array([0, 1, 0])  # Along y-axis

# Compute cross product
cross = np.cross(u, v)  # np.cross() computes cross product (3D vectors only)

print(f"u = {u}")
print(f"v = {v}")
print(f"u × v = {cross}")
print(f"|u × v| = {np.linalg.norm(cross):.2f}")  # np.linalg.norm() computes vector magnitude (Euclidean norm)
print(f"\nVerify perpendicularity:")
print(f"u · (u × v) = {np.dot(u, cross):.10f}  (should be 0)")  # np.dot() computes dot product of two arrays
print(f"v · (u × v) = {np.dot(v, cross):.10f}  (should be 0)")  # np.dot() computes dot product of two arrays
```

**Expected output:**
```
u = [1 0 0]
v = [0 1 0]
u × v = [0 0 1]
|u × v| = 1.00

Verify perpendicularity:
u · (u × v) = 0.0000000000  (should be 0)
v · (u × v) = 0.0000000000  (should be 0)
```

Notice: x-axis × y-axis = z-axis (right-hand rule!)

---

### 📊 Visualization: Cross Product in 3D

```python
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit

fig = plt.figure(figsize=(10, 8))  # plt.figure() creates a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  projection='3d'  # Create 3D axes

# Plot vectors
ax.quiver(0, 0, 0, u[0], u[1], u[2], color=COLORS['blue'],
         arrow_length_ratio=0.2, linewidth=2, label='u')
ax.quiver(0, 0, 0, v[0], v[1], v[2], color=COLORS['orange'],
         arrow_length_ratio=0.2, linewidth=2, label='v')
ax.quiver(0, 0, 0, cross[0], cross[1], cross[2], color=COLORS['green'],
         arrow_length_ratio=0.2, linewidth=2, label='u × v')

# Add labels
ax.text(u[0], u[1], u[2], '  u', fontsize=12, color=COLORS['blue'])
ax.text(v[0], v[1], v[2], '  v', fontsize=12, color=COLORS['orange'])
ax.text(cross[0], cross[1], cross[2], '  u × v', fontsize=12, color=COLORS['green'])

# Draw parallelogram to show area
vertices = np.array([[0, 0, 0], u, u+v, v, [0, 0, 0]])  # np.array() converts Python list/tuple to efficient numpy array
ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
       color=COLORS['gray'], alpha=0.5, linestyle='--')

ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-0.5, 1.5])
ax.set_zlim([-0.5, 1.5])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.set_title("Cross Product: u × v perpendicular to both")
plt.show()  # plt.show() displays the figure window
```

---

### 🎯 Practice Question #3

**Q:** If u = (1, 0, 0) and v = (1, 1, 0), what is u × v?

<details>
<summary>💡 Hint 1</summary>

Use the formula: u × v = (u₂v₃ - u₃v₂, u₃v₁ - u₁v₃, u₁v₂ - u₂v₁)
</details>

<details>
<summary>💡 Hint 2</summary>

The result should point in the +z direction (right-hand rule).
</details>

<details>
<summary>✅ Answer</summary>

u × v = (0·0 - 0·1, 0·1 - 1·0, 1·1 - 0·1) = **(0, 0, 1)**

```python
u = np.array([1, 0, 0])  # np.array() converts Python list/tuple to efficient numpy array
v = np.array([1, 1, 0])  # np.array() converts Python list/tuple to efficient numpy array
cross = np.cross(u, v)  # np.cross() computes cross product (3D vectors only)
print(cross)  # [0 0 1]
```

The result points along the +z axis, perpendicular to both u (x-axis) and v (xy-plane)!
</details>

---

### 🔬 Explore: Area of Parallelogram

The magnitude of the cross product gives you the area:

```python
# Two vectors in 3D
u = np.array([2, 0, 0])  # np.array() converts Python list/tuple to efficient numpy array
v = np.array([1, 3, 0])  # np.array() converts Python list/tuple to efficient numpy array

cross = np.cross(u, v)  # np.cross() computes cross product (3D vectors only)
area = np.linalg.norm(cross)  # np.linalg.norm() computes vector magnitude (Euclidean norm)

print(f"u = {u}")
print(f"v = {v}")
print(f"Area of parallelogram = |u × v| = {area}")

# Verify: area should equal base × height
base = np.linalg.norm(u)  # np.linalg.norm() computes vector magnitude (Euclidean norm)
# Height is component of v perpendicular to u
height = abs(v[1])  # Since u is along x-axis
print(f"Verification: base × height = {base} × {height} = {base * height}")
```

---

## 5. Projection and Orthogonality

*[Content continues with similar structure...]*

---

## Practice Questions

After working through all sections, test your understanding:

### Section 1-2: Vectors and Basis

1. **Vector operations:** Given u = (2, -3, 1) and v = (4, 1, -2), compute:
   - u + v
   - 3u - 2v
   - |u|

2. **Linear independence:** Are these vectors linearly independent?
   - v₁ = (1, 2, 3)
   - v₂ = (2, 4, 6)
   - v₃ = (1, 0, 1)

3. **Basis:** Express v = (7, 4) as a linear combination of basis {(1, 2), (3, 1)}

### Section 3-4: Dot and Cross Products

4. **Dot product:** Given u = (1, -2, 3) and v = (2, 1, -1), find:
   - u · v
   - Angle between u and v
   - Are they orthogonal?

5. **Cross product:** Compute (2, 1, 0) × (1, 0, 3)

6. **Geometry:** What is the area of the parallelogram formed by u = (3, 0, 0) and v = (0, 4, 0)?

### Sections 6-9: Matrices and Eigenvalues

7. **Matrix multiplication:** Given A = [[2, 1], [1, 3]], find A²

8. **Determinant:** What is det([[3, 1, 2], [0, 2, 1], [1, 0, 4]])?

9. **Eigenvalues:** For matrix [[4, 1], [0, 3]], what are the eigenvalues?

10. **Inverse:** Find the inverse of [[2, 1], [5, 3]]

---

### 📝 Check Your Answers

Run the quiz script:
```bash
cd lessons/01_linear_algebra
python quiz.py
```

Or check solutions in `SOLUTIONS.md`

---

## Next Steps

✅ Complete all practice questions
✅ Explore the visualization code
✅ Try modifying examples with your own values
✅ Move to Lesson 2: Multivariable Calculus

**Additional Resources:**
- 3Blue1Brown's "Essence of Linear Algebra" (YouTube)
- MIT OCW 18.06: Linear Algebra (Gilbert Strang)
- Khan Academy: Linear Algebra

---

**Ready to continue?** → [Lesson 2: Multivariable Calculus](../02_multivariable_calculus/LESSON.md)
