# Lesson 6: Tensors

**Topics:** Tensor definition, index notation, Einstein summation convention, contravariant and covariant components, tensor operations, metric tensor, raising and lowering indices

**Prerequisites:** Linear algebra (Lesson 1), multivariable calculus (Lesson 2), curves and surfaces (Lesson 4), manifolds (Lesson 5)

**Time:** ~5-6 hours

---

## Table of Contents

1. [What is a Tensor?](#1-what-is-a-tensor)
2. [Index Notation and Einstein Summation](#2-index-notation-and-einstein-summation)
3. [Contravariant and Covariant Tensors](#3-contravariant-and-covariant-tensors)
4. [Tensor Operations](#4-tensor-operations)
5. [The Metric Tensor](#5-the-metric-tensor)
6. [Raising and Lowering Indices](#6-raising-and-lowering-indices)
7. [Examples: Tensors in Physics](#7-examples-tensors-in-physics)

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

## 1. What is a Tensor?

### 📖 Concept

A **tensor** is a multilinear map that takes vectors and one-forms as inputs and outputs a number.

**Informal definition:** A tensor of type (r, s) is an object with r upper indices (contravariant) and s lower indices (covariant):
```
T^(μ₁...μᵣ)_(ν₁...νₛ)
```

**Formal definition:** A (r, s)-tensor at point p is a multilinear map:
```
T: (T*_p M)^r × (T_p M)^s → ℝ
```
That takes r one-forms and s vectors, and returns a number.

**Examples:**
- **(0,0)-tensor:** Scalar (just a number)
- **(1,0)-tensor:** Vector (contravariant)
- **(0,1)-tensor:** One-form (covariant)
- **(0,2)-tensor:** Metric g_μν (takes two vectors → number)
- **(1,1)-tensor:** Linear transformation (takes vector → vector)
- **(2,0)-tensor:** Stress-energy in physics
- **(1,3)-tensor:** Riemann curvature (contracted form)

**Key properties:**
1. **Multilinearity:** Linear in each argument separately
2. **Coordinate transformation:** Components transform in a specific way
3. **Geometric object:** Independent of coordinates

**Why this matters for GR:** General Relativity is written entirely in the language of tensors! The metric g_μν, Riemann curvature R^ρ_σμν, stress-energy T_μν - all tensors. This ensures physics is independent of coordinate choice.

---

### 💻 Code Example: Tensors in NumPy

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Scalars (0,0)-tensors
scalar = 5.0
print(f"Scalar (0,0)-tensor: {scalar}")
print(f"Shape: single number\n")

# Vectors (1,0)-tensors - one upper index
vector = np.array([1, 2, 3])  # np.array() converts Python list/tuple to efficient numpy array
print(f"Vector (1,0)-tensor: {vector}")
print(f"Shape: {vector.shape} (3 components)\n")

# One-forms (0,1)-tensors - one lower index
one_form = np.array([4, 5, 6])  # np.array() converts Python list/tuple to efficient numpy array
print(f"One-form (0,1)-tensor: {one_form}")
print(f"Shape: {one_form.shape} (3 components)\n")

# (0,2)-tensor - two lower indices (like a metric)
metric = np.array([  # np.array() converts Python list/tuple to efficient numpy array
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
print(f"(0,2)-tensor (matrix):")
print(metric)
print(f"Shape: {metric.shape} (3×3 components)\n")

# (1,1)-tensor - one upper, one lower index (linear transformation)
linear_map = np.array([  # np.array() converts Python list/tuple to efficient numpy array
    [2, 1, 0],
    [0, 3, 1],
    [1, 0, 2]
])
print(f"(1,1)-tensor (mixed):")
print(linear_map)
print(f"Shape: {linear_map.shape}\n")

# (0,3)-tensor - three lower indices
tensor_003 = np.random.randn(3, 3, 3)
print(f"(0,3)-tensor:")
print(f"Shape: {tensor_003.shape} (27 components)")
```

**Expected output:**
```
Scalar (0,0)-tensor: 5.0
Shape: single number

Vector (1,0)-tensor: [1 2 3]
Shape: (3,) (3 components)

One-form (0,1)-tensor: [4 5 6]
Shape: (3,) (3 components)

(0,2)-tensor (matrix):
[[1 0 0]
 [0 1 0]
 [0 0 1]]
Shape: (3, 3) (3×3 components)

(1,1)-tensor (mixed):
[[2 1 0]
 [0 3 1]
 [1 0 2]]
Shape: (3, 3)

(0,3)-tensor:
Shape: (3, 3, 3) (27 components)
```

---

### 📊 Visualization: Tensor Hierarchy

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Scalar
ax1 = axes[0, 0]
ax1.text(0.5, 0.5, 'Scalar\n(0,0)-tensor\n\nT = 5',
        transform=ax1.transAxes, fontsize=20, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor=COLORS['blue'], alpha=0.3))
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.axis('off')
ax1.set_title('Rank 0: 1 component', fontsize=14, fontweight='bold')

# Top-right: Vector
ax2 = axes[0, 1]
vector_example = np.array([1, 2, 3])  # np.array() converts Python list/tuple to efficient numpy array
y_pos = np.arange(len(vector_example))
ax2.barh(y_pos, vector_example, color=COLORS['orange'])
ax2.set_yticks(y_pos)
ax2.set_yticklabels(['T¹', 'T²', 'T³'])
ax2.set_xlabel('Component value')
ax2.set_title('Rank 1: Vector (1,0)-tensor\n3 components', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Bottom-left: Matrix (0,2)-tensor
ax3 = axes[1, 0]
matrix_example = np.array([  # np.array() converts Python list/tuple to efficient numpy array
    [1, 0.5, 0.2],
    [0.5, 2, 0.3],
    [0.2, 0.3, 1.5]
])
im = ax3.imshow(matrix_example, cmap='RdBu_r', aspect='auto')
ax3.set_xticks([0, 1, 2])
ax3.set_yticks([0, 1, 2])
ax3.set_xticklabels(['1', '2', '3'])
ax3.set_yticklabels(['1', '2', '3'])
ax3.set_xlabel('ν')
ax3.set_ylabel('μ')
ax3.set_title('Rank 2: Matrix (0,2)-tensor\n9 components', fontsize=14, fontweight='bold')

# Add values to cells
for i in range(3):
    for j in range(3):
        text = ax3.text(j, i, f'{matrix_example[i, j]:.1f}',
                       ha="center", va="center", color="black", fontsize=11)

plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)  # plt.colorbar() adds color scale bar to plot

# Bottom-right: Higher-rank tensor (symbolic)
ax4 = axes[1, 1]
ax4.text(0.5, 0.7, 'Rank 3: (0,3)-tensor',
        transform=ax4.transAxes, fontsize=16, ha='center', va='center',
        fontweight='bold')
ax4.text(0.5, 0.5, 'T_μνρ\n\n3 × 3 × 3 = 27 components',
        transform=ax4.transAxes, fontsize=14, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.2))
ax4.text(0.5, 0.2, '(Cannot easily visualize!)',
        transform=ax4.transAxes, fontsize=11, ha='center', va='center',
        style='italic', color=COLORS['gray'])
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.axis('off')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Tensor rank = number of indices")
print("Number of components = dimension^rank")
print("For 3D: rank-2 tensor has 3² = 9 components")
```

---

### 🎯 Practice Question #1

**Q:** How many independent components does a (2,1)-tensor have in 4-dimensional spacetime?

<details>
<summary>💡 Hint</summary>

Count indices: 2 upper (contravariant) + 1 lower (covariant) = 3 total indices. Each index runs from 0 to 3 (four values in 4D).
</details>

<details>
<summary>✅ Answer</summary>

**64 components**

A (2,1)-tensor has 3 indices total: T^μν_ρ

Each index runs over 4 values (0, 1, 2, 3) in 4D spacetime.

Number of components = 4 × 4 × 4 = 4³ = **64**

```python
# In 4D spacetime
dimension = 4
num_upper = 2  # contravariant indices
num_lower = 1  # covariant indices
total_indices = num_upper + num_lower

num_components = dimension ** total_indices
print(f"Components: {num_components}")  # 64
```

Note: Some tensors have symmetries that reduce the number of independent components (e.g., symmetric or antisymmetric tensors).
</details>

---

## 2. Index Notation and Einstein Summation

### 📖 Concept

**Index notation** is the standard way to write tensors and tensor equations.

**Basic rules:**
1. **Upper indices (superscripts):** Contravariant components v^μ
2. **Lower indices (subscripts):** Covariant components w_μ
3. **Greek indices (μ, ν, ρ, σ):** Spacetime (0,1,2,3)
4. **Latin indices (i, j, k):** Spatial only (1,2,3)

**Einstein summation convention:** When an index appears once up and once down, sum over it!

```
v^μ w_μ = Σ_μ v^μ w_μ = v^0 w_0 + v^1 w_1 + v^2 w_2 + v^3 w_3
```

**No sum symbol needed!** Automatic summation over repeated indices.

**Examples:**
- `v^μ w_μ` - dot product (sum over μ)
- `A^μ_ν v^ν` - matrix-vector product (sum over ν)
- `g_μν v^μ v^ν` - norm squared (sum over μ and ν)
- `T^μ_μ` - trace (sum over μ)

**Rules for summation:**
- **Dummy index:** Summed over (can rename: v^μ w_μ = v^α w_α)
- **Free index:** Not summed (appears once on each side)
- **Never sum index appearing twice in same position!** (both up or both down)

**Why this matters for GR:** All GR equations use index notation. Einstein field equations: G_μν = 8πG T_μν. Compact notation for complex tensor equations!

---

### 💻 Code Example: Einstein Summation

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations

# Define vectors
v = np.array([1, 2, 3, 4])  # v^μ
w = np.array([5, 6, 7, 8])  # w_μ

# Dot product using explicit sum
dot_explicit = 0
for mu in range(4):
    dot_explicit += v[mu] * w[mu]

print("Explicit summation:")
print(f"v^μ w_μ = Σ_μ v^μ w_μ = {dot_explicit}")

# Using NumPy (automatic)
dot_numpy = np.dot(v, w)  # np.dot() computes dot product of two arrays
print(f"\nNumPy dot product: {dot_numpy}")

# Using np.einsum (Einstein summation)
dot_einsum = np.einsum('i,i->', v, w)
print(f"np.einsum('i,i->', v, w): {dot_einsum}")

print("\n" + "="*50)

# Matrix-vector multiplication: A^μ_ν v^ν
A = np.array([  # np.array() converts Python list/tuple to efficient numpy array
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Explicit summation
result_explicit = np.zeros(4)  # np.zeros() creates array filled with zeros
for mu in range(4):
    for nu in range(4):
        result_explicit[mu] += A[mu, nu] * v[nu]

print("\nMatrix-vector product A^μ_ν v^ν:")
print(f"Explicit: {result_explicit}")

# Using matrix multiplication
result_matmul = A @ v    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
print(f"NumPy @: {result_matmul}")    # @ is matrix multiplication operator (equivalent to np.dot for matrices)

# Using einsum
result_einsum = np.einsum('ij,j->i', A, v)
print(f"einsum: {result_einsum}")

print("\n" + "="*50)

# Trace: T^μ_μ (sum over repeated index)
T = np.array([  # np.array() converts Python list/tuple to efficient numpy array
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

trace_explicit = 0
for mu in range(3):
    trace_explicit += T[mu, mu]

print("\nTrace T^μ_μ:")
print(f"Explicit: {trace_explicit}")
print(f"NumPy trace: {np.trace(T)}")
print(f"einsum: {np.einsum('ii->', T)}")
```

---

### 📊 Visualization: Einstein Summation Convention

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Top-left: Dot product
ax1 = axes[0, 0]
v_example = np.array([2, 3, 1, 4])  # np.array() converts Python list/tuple to efficient numpy array
w_example = np.array([1, 2, 3, 2])  # np.array() converts Python list/tuple to efficient numpy array

x_pos = np.arange(4)
width = 0.35

ax1.bar(x_pos - width/2, v_example, width, label='v^μ', color=COLORS['blue'])
ax1.bar(x_pos + width/2, w_example, width, label='w_μ', color=COLORS['orange'])
ax1.set_xlabel('Index μ', fontsize=12)
ax1.set_ylabel('Component value', fontsize=12)
ax1.set_title('Dot Product: v^μ w_μ = Σ_μ v^μ w_μ', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(['0', '1', '2', '3'])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Show products
products = v_example * w_example
total = np.sum(products)
text_str = ' + '.join([f'{v}×{w}' for v, w in zip(v_example, w_example)])
text_str += f' = {total}'
ax1.text(0.5, 0.95, text_str, transform=ax1.transAxes,
        fontsize=10, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Top-right: Matrix-vector product
ax2 = axes[0, 1]
A_small = np.array([[2, 1], [3, 4]])  # np.array() converts Python list/tuple to efficient numpy array
v_small = np.array([5, 2])  # np.array() converts Python list/tuple to efficient numpy array

# Show matrix
im = ax2.imshow(A_small, cmap='Blues', aspect='auto', alpha=0.6)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['ν=0', 'ν=1'])
ax2.set_yticklabels(['μ=0', 'μ=1'])
ax2.set_title('Matrix-Vector: (Av)^μ = A^μ_ν v^ν', fontsize=13, fontweight='bold')

for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{A_small[i,j]}', ha="center", va="center",
                fontsize=14, fontweight='bold')

# Show computation
result = A_small @ v_small    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
text_str = f'v = [{v_small[0]}, {v_small[1]}]\n\n'
text_str += f'Result^0 = {A_small[0,0]}×{v_small[0]} + {A_small[0,1]}×{v_small[1]} = {result[0]}\n'
text_str += f'Result^1 = {A_small[1,0]}×{v_small[0]} + {A_small[1,1]}×{v_small[1]} = {result[1]}'

ax2.text(1.5, 0.5, text_str, fontsize=10, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Bottom-left: Double sum (quadratic form)
ax3 = axes[1, 0]
g = np.array([[1, 0.2], [0.2, 1]])  # Metric
v_form = np.array([3, 2])  # np.array() converts Python list/tuple to efficient numpy array

im = ax3.imshow(g, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=2)
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['ν=0', 'ν=1'])
ax3.set_yticklabels(['μ=0', 'μ=1'])
ax3.set_title('Quadratic Form: g_μν v^μ v^ν (double sum)', fontsize=13, fontweight='bold')

for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{g[i,j]:.1f}', ha="center", va="center",
                fontsize=14, fontweight='bold', color='white' if abs(g[i,j]) > 0.5 else 'black')

# Compute
result = 0
terms = []
for i in range(2):
    for j in range(2):
        term = g[i,j] * v_form[i] * v_form[j]
        result += term
        terms.append(f'({g[i,j]:.1f})×{v_form[i]}×{v_form[j]}')

text_str = f'v = [{v_form[0]}, {v_form[1]}]\n\n'
text_str += ' + '.join(terms[:2]) + '\n+ '
text_str += ' + '.join(terms[2:]) + f'\n= {result:.1f}'

ax3.text(0.5, -0.35, text_str, transform=ax3.transAxes,
        fontsize=9, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Bottom-right: Summary
ax4 = axes[1, 1]
summary_text = """
Einstein Summation Convention

Rules:
1. Repeated index (one up, one down): SUM
   Example: v^μ w_μ = Σ_μ v^μ w_μ

2. Free index: appears once on each side
   Example: T^μ_ν v^ν = w^μ
   (sum over ν, μ is free)

3. Never: both indices up OR both down
   ✗ v^μ w^μ (not a valid tensor expression!)
   ✗ v_μ w_μ (unless with metric)

Common patterns:
• v^μ w_μ - inner product
• A^μ_ν v^ν - matrix × vector
• g_μν v^μ v^ν - quadratic form
• T^μ_μ - trace
• δ^μ_ν - Kronecker delta
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=11, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.axis('off')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🔬 Explore: Using np.einsum

```python
# np.einsum is a powerful tool for tensor operations
# Syntax: np.einsum('input_indices->output_indices', arrays)

# Example 1: Outer product v^μ w^ν (no summation)
v = np.array([1, 2, 3])  # np.array() converts Python list/tuple to efficient numpy array
w = np.array([4, 5, 6])  # np.array() converts Python list/tuple to efficient numpy array

outer = np.einsum('i,j->ij', v, w)
print("Outer product v^μ w^ν:")
print(outer)
print(f"Shape: {outer.shape}\n")

# Example 2: Tensor contraction T^μν_νρ → S^μ_ρ (sum over ν)
T = np.random.randn(3, 3, 3, 3)  # T^μνλρ
S = np.einsum('ijjk->ik', T)     # Contract 2nd and 3rd indices
print(f"Contraction T^μν_νρ:")
print(f"Original shape: {T.shape}")
print(f"Result shape: {S.shape}\n")

# Example 3: Matrix multiplication chain A @ B @ C
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = np.random.randn(5, 3)

result_standard = A @ B @ C    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
result_einsum = np.einsum('ij,jk,kl->il', A, B, C)

print("Matrix chain multiplication:")
print(f"Standard: {result_standard.shape}")
print(f"einsum: {result_einsum.shape}")
print(f"Match: {np.allclose(result_standard, result_einsum)}")  # np.allclose() tests if all array elements are approximately equal
```

---

### 🎯 Practice Question #2

**Q:** What does the expression T^μν g_μν represent? (T is a (2,0)-tensor, g is the metric)

<details>
<summary>💡 Hint</summary>

Look at the repeated indices. Which indices are summed over?
</details>

<details>
<summary>✅ Answer</summary>

**This is a contraction producing a scalar (trace-like operation).**

Both μ and ν appear as repeated indices (μ up in T, down in g; same for ν).

T^μν g_μν = Σ_μ Σ_ν T^μν g_μν

This contracts the (2,0)-tensor T with the (0,2)-metric g to produce a scalar.

```python
# Example in 3D
T = np.array([[1, 2, 3],  # np.array() converts Python list/tuple to efficient numpy array
              [4, 5, 6],
              [7, 8, 9]])

g = np.array([[1, 0, 0],  # np.array() converts Python list/tuple to efficient numpy array
              [0, 1, 0],
              [0, 0, 1]])  # Euclidean metric

# Double sum
result = np.einsum('ij,ij->', T, g)
print(f"T^μν g_μν = {result}")  # Sum of diagonal elements (since g is identity)
```

In physics: if T^μν is the stress-energy tensor, this gives the trace T = T^μ_μ.
</details>

---

## 3. Contravariant and Covariant Tensors

### 📖 Concept

Tensors have two types of indices that transform differently under coordinate changes:

**Contravariant (upper indices):** Transform like basis vectors
```
v'^μ = ∂x'^μ/∂x^ν v^ν
```

**Covariant (lower indices):** Transform like gradients
```
w'_μ = ∂x^ν/∂x'^μ w_ν
```

**Why the difference?**
- **Vectors (contravariant):** Components change opposite to basis (if basis shrinks, components grow)
- **One-forms (covariant):** Components change same as basis

**Physical intuition:**
- **Position/velocity:** Contravariant (if you stretch coordinates, components shrink)
- **Gradient/momentum:** Covariant (if you stretch coordinates, gradient components grow)

**Transformation rule for general (r,s)-tensor:**
```
T'^μ₁...μᵣ_ν₁...νₛ = (∂x'^μ₁/∂x^α₁)...(∂x'^μᵣ/∂x^αᵣ) × (∂x^β₁/∂x'^ν₁)...(∂x^βₛ/∂x'^νₛ) × T^α₁...αᵣ_β₁...βₛ
```

**Key insight:** Tensor equations that hold in one coordinate system hold in ALL coordinate systems! This is why we use tensors in physics.

**Why this matters for GR:** The laws of physics must be the same for all observers (coordinate systems). Tensor equations automatically satisfy this requirement.

---

### 💻 Code Example: Coordinate Transformation

```python
# 2D example: Cartesian (x, y) to polar (r, θ)

def cartesian_to_polar(x, y):
    """Convert Cartesian to polar coordinates"""
    r = np.sqrt(x**2 + y**2)  # np.sqrt() computes square root
    theta = np.arctan2(y, x)
    return r, theta

def jacobian_cart_to_polar(x, y):
    """
    Jacobian matrix ∂(r,θ)/∂(x,y)
    [∂r/∂x   ∂r/∂y  ]
    [∂θ/∂x   ∂θ/∂y  ]
    """
    r = np.sqrt(x**2 + y**2)  # np.sqrt() computes square root
    if r < 1e-10:
        return np.eye(2)  # Avoid singularity at origin

    dr_dx = x / r
    dr_dy = y / r
    dtheta_dx = -y / r**2
    dtheta_dy = x / r**2

    return np.array([  # np.array() converts Python list/tuple to efficient numpy array
        [dr_dx, dr_dy],
        [dtheta_dx, dtheta_dy]
    ])

def inverse_jacobian_cart_to_polar(x, y):
    """
    Inverse Jacobian ∂(x,y)/∂(r,θ)
    [∂x/∂r   ∂x/∂θ  ]
    [∂y/∂r   ∂y/∂θ  ]
    """
    r, theta = cartesian_to_polar(x, y)

    dx_dr = x / r if r > 1e-10 else 1
    dx_dtheta = -y
    dy_dr = y / r if r > 1e-10 else 0
    dy_dtheta = x

    return np.array([  # np.array() converts Python list/tuple to efficient numpy array
        [dx_dr, dx_dtheta],
        [dy_dr, dy_dtheta]
    ])

# Test point
x, y = 3.0, 4.0
r, theta = cartesian_to_polar(x, y)

print(f"Point in Cartesian: (x, y) = ({x}, {y})")
print(f"Point in polar: (r, θ) = ({r:.3f}, {theta:.3f} rad)\n")

# Transform a contravariant vector
v_cartesian = np.array([1.0, 2.0])  # Vector in Cartesian coords
print(f"Contravariant vector in Cartesian: v = {v_cartesian}")

# Transform: v'^μ = (∂x'^μ/∂x^ν) v^ν
J = jacobian_cart_to_polar(x, y)
v_polar = J @ v_cartesian    # @ is matrix multiplication operator (equivalent to np.dot for matrices)

print(f"Jacobian ∂(r,θ)/∂(x,y):")
print(J)
print(f"Vector in polar coords: v' = {v_polar}")
print(f"  v^r = {v_polar[0]:.3f}")
print(f"  v^θ = {v_polar[1]:.3f}\n")

# Transform a covariant vector (one-form)
w_cartesian = np.array([5.0, 6.0])  # One-form in Cartesian
print(f"Covariant vector (one-form) in Cartesian: w = {w_cartesian}")

# Transform: w'_μ = (∂x^ν/∂x'^μ) w_ν
J_inv = inverse_jacobian_cart_to_polar(x, y)
w_polar = J_inv.T @ w_cartesian  # Note: transpose for covariant

print(f"Inverse Jacobian ∂(x,y)/∂(r,θ):")
print(J_inv)
print(f"One-form in polar coords: w' = {w_polar}")
print(f"  w_r = {w_polar[0]:.3f}")
print(f"  w_θ = {w_polar[1]:.3f}\n")

# Check: dot product is invariant!
dot_cartesian = np.dot(v_cartesian, w_cartesian)  # np.dot() computes dot product of two arrays
dot_polar = np.dot(v_polar, w_polar)  # np.dot() computes dot product of two arrays
print(f"Dot product in Cartesian: {dot_cartesian:.6f}")
print(f"Dot product in polar: {dot_polar:.6f}")
print(f"Match (invariant): {np.isclose(dot_cartesian, dot_polar)}")  # np.isclose() tests if values are approximately equal (handles floating point)
```

---

### 📊 Visualization: Contravariant vs Covariant

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Contravariant vector (transforms opposite to basis)
ax1 = axes[0]

# Original Cartesian basis
ax1.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1,
         fc=COLORS['blue'], ec=COLORS['blue'], linewidth=2, label='e_x')
ax1.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1,
         fc=COLORS['orange'], ec=COLORS['orange'], linewidth=2, label='e_y')

# Stretched basis (scale x by 2)
ax1.arrow(0, -2, 2, 0, head_width=0.1, head_length=0.1,
         fc=COLORS['blue'], ec=COLORS['blue'], linewidth=2,
         linestyle='--', alpha=0.6)
ax1.arrow(0, -2, 0, 1, head_width=0.1, head_length=0.1,
         fc=COLORS['orange'], ec=COLORS['orange'], linewidth=2,
         linestyle='--', alpha=0.6)

# Original vector v = (2, 1)
v_orig = np.array([2, 1])  # np.array() converts Python list/tuple to efficient numpy array
ax1.arrow(0, 0, v_orig[0], v_orig[1], head_width=0.15, head_length=0.15,
         fc=COLORS['green'], ec=COLORS['green'], linewidth=3, label='v (original)')

# Transformed vector components (basis stretched by 2 in x → components shrink by 1/2)
v_transformed = np.array([1, 1])  # v' = (1, 1)
ax1.arrow(0, -2, v_transformed[0]*2, v_transformed[1], head_width=0.15, head_length=0.15,
         fc=COLORS['red'], ec=COLORS['red'], linewidth=3, alpha=0.7,
         label='v (new coords)')

ax1.text(1, 1.3, 'Basis stretched →\nComponents shrink', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax1.set_xlim([-0.5, 3])
ax1.set_ylim([-2.5, 2])
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Contravariant Vector v^μ', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='k', linewidth=0.5)
ax1.axvline(0, color='k', linewidth=0.5)

# Right: Covariant vector (transforms same as basis)
ax2 = axes[1]

# Draw level surfaces of a function (one-form is gradient)
x_vals = np.linspace(-1, 3, 100)  # np.linspace() creates evenly spaced array between start and end
y_vals = np.linspace(-1, 3, 100)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x_vals, y_vals)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# Function f(x,y) = 2x + y
F = 2*X + Y

# Plot level curves
contour = ax2.contour(X, Y, F, levels=10, colors=COLORS['blue'], alpha=0.6)
ax2.clabel(contour, inline=True, fontsize=8)

# Gradient (covariant vector): df = ∂f/∂x dx + ∂f/∂y dy = 2 dx + 1 dy
gradient = np.array([2, 1])  # np.array() converts Python list/tuple to efficient numpy array
ax2.arrow(1, 1, gradient[0]*0.3, gradient[1]*0.3,
         head_width=0.1, head_length=0.1,
         fc=COLORS['red'], ec=COLORS['red'], linewidth=3,
         label='∇f (gradient)')

ax2.text(1.8, 1.5, 'Gradient perpendicular\nto level curves', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax2.set_xlim([-1, 3])
ax2.set_ylim([-1, 3])
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Covariant Vector (One-Form) ω_μ', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Contravariant: components change opposite to basis")
print("Covariant: components change with basis (like gradients)")
```

---

## 4. Tensor Operations

### 📖 Concept

**Tensor addition:** Add components (must be same type):
```
(S + T)^μ_ν = S^μ_ν + T^μ_ν
```

**Scalar multiplication:**
```
(cT)^μ_ν = c T^μ_ν
```

**Tensor product (outer product):** Combine tensors:
```
(S ⊗ T)^μν_ρσ = S^μ_ρ T^ν_σ
```
Result has rank = sum of ranks.

**Contraction:** Sum over one up and one down index:
```
T^μν_μρ = Σ_μ T^μν_μρ → S^ν_ρ
```
Result has rank decreased by 2.

**Symmetrization:**
```
T_(μν) = 1/2(T_μν + T_νμ)
```

**Antisymmetrization:**
```
T_[μν] = 1/2(T_μν - T_νμ)
```

**Common operations:**
- Trace: contraction of (1,1)-tensor T^μ_μ
- Determinant: fully antisymmetric contraction
- Divergence: ∂_μ v^μ (contraction with derivative)

**Why this matters for GR:** Einstein field equations involve contractions (Ricci tensor from Riemann), symmetrization, traces. The energy-momentum tensor is symmetric: T_μν = T_νμ.

---

### 💻 Code Example: Tensor Operations

```python
# Define some tensors
T = np.array([[1, 2, 3],  # np.array() converts Python list/tuple to efficient numpy array
              [4, 5, 6],
              [7, 8, 9]])  # (1,1)-tensor

S = np.array([[9, 8, 7],  # np.array() converts Python list/tuple to efficient numpy array
              [6, 5, 4],
              [3, 2, 1]])  # (1,1)-tensor

print("=" * 50)
print("TENSOR ADDITION")
print("=" * 50)
sum_tensor = T + S
print(f"T + S =\n{sum_tensor}\n")

print("=" * 50)
print("SCALAR MULTIPLICATION")
print("=" * 50)
scaled = 3 * T
print(f"3T =\n{scaled}\n")

print("=" * 50)
print("TENSOR PRODUCT (outer product)")
print("=" * 50)
v = np.array([1, 2, 3])  # np.array() converts Python list/tuple to efficient numpy array
w = np.array([4, 5, 6])  # np.array() converts Python list/tuple to efficient numpy array

# v^μ ⊗ w^ν → (2,0)-tensor
outer_product = np.outer(v, w)
print(f"v ⊗ w =\n{outer_product}")
print(f"Shape: {outer_product.shape} (rank 2)\n")

print("=" * 50)
print("CONTRACTION (trace)")
print("=" * 50)
trace = np.trace(T)
print(f"Trace T^μ_μ = {trace}")
print(f"Explicit: T^0_0 + T^1_1 + T^2_2 = {T[0,0]} + {T[1,1]} + {T[2,2]} = {trace}\n")

# Contraction using einsum
trace_einsum = np.einsum('ii->', T)
print(f"Using einsum: {trace_einsum}\n")

print("=" * 50)
print("SYMMETRIZATION")
print("=" * 50)
# Make a non-symmetric tensor
A = np.array([[1, 2, 3],  # np.array() converts Python list/tuple to efficient numpy array
              [4, 5, 6],
              [7, 8, 9]])

symmetric = 0.5 * (A + A.T)
print(f"Original A =\n{A}\n")
print(f"Symmetric part A_(μν) = (A_μν + A_νμ)/2 =\n{symmetric}\n")

print("=" * 50)
print("ANTISYMMETRIZATION")
print("=" * 50)
antisymmetric = 0.5 * (A - A.T)
print(f"Antisymmetric part A_[μν] = (A_μν - A_νμ)/2 =\n{antisymmetric}\n")

# Verify decomposition: A = symmetric + antisymmetric
reconstructed = symmetric + antisymmetric
print(f"Reconstruction check:")
print(f"Symmetric + Antisymmetric =\n{reconstructed}")
print(f"Matches original: {np.allclose(A, reconstructed)}\n")  # np.allclose() tests if all array elements are approximately equal

print("=" * 50)
print("HIGHER-RANK CONTRACTION")
print("=" * 50)
# (0,2,0,1)-tensor → contract 2nd and 3rd indices
tensor_4 = np.random.randn(3, 3, 3, 3)
print(f"Original shape: {tensor_4.shape}")

# Contract: T_μ^ν_ν^ρ → S_μ^ρ
contracted = np.einsum('ijjk->ik', tensor_4)
print(f"After contracting indices 1 and 2: {contracted.shape}")
```

---

### 📊 Visualization: Symmetrization and Antisymmetrization

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original tensor
A = np.array([[5, 2, 1],  # np.array() converts Python list/tuple to efficient numpy array
              [8, 3, 4],
              [6, 7, 9]])

# Symmetric part
A_sym = 0.5 * (A + A.T)

# Antisymmetric part
A_antisym = 0.5 * (A - A.T)

# Plot original
ax1 = axes[0, 0]
im1 = ax1.imshow(A, cmap='RdBu_r', vmin=-5, vmax=10)
ax1.set_title('Original A_μν', fontsize=14, fontweight='bold')
ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1, 2])
for i in range(3):
    for j in range(3):
        ax1.text(j, i, f'{A[i,j]:.0f}', ha="center", va="center",
                color='white' if abs(A[i,j]) > 5 else 'black', fontsize=12)
plt.colorbar(im1, ax=ax1)  # plt.colorbar() adds color scale bar to plot

# Plot symmetric part
ax2 = axes[0, 1]
im2 = ax2.imshow(A_sym, cmap='RdBu_r', vmin=-5, vmax=10)
ax2.set_title('Symmetric A_(μν) = (A_μν + A_νμ)/2', fontsize=14, fontweight='bold')
ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2])
for i in range(3):
    for j in range(3):
        ax2.text(j, i, f'{A_sym[i,j]:.1f}', ha="center", va="center",
                color='white' if abs(A_sym[i,j]) > 5 else 'black', fontsize=11)
plt.colorbar(im2, ax=ax2)  # plt.colorbar() adds color scale bar to plot

# Plot antisymmetric part
ax3 = axes[1, 0]
im3 = ax3.imshow(A_antisym, cmap='RdBu_r', vmin=-5, vmax=5)
ax3.set_title('Antisymmetric A_[μν] = (A_μν - A_νμ)/2', fontsize=14, fontweight='bold')
ax3.set_xticks([0, 1, 2])
ax3.set_yticks([0, 1, 2])
for i in range(3):
    for j in range(3):
        ax3.text(j, i, f'{A_antisym[i,j]:.1f}', ha="center", va="center",
                color='white' if abs(A_antisym[i,j]) > 2 else 'black', fontsize=11)
plt.colorbar(im3, ax=ax3)  # plt.colorbar() adds color scale bar to plot

# Verification
ax4 = axes[1, 1]
ax4.text(0.5, 0.7, 'Decomposition Theorem', transform=ax4.transAxes,
        fontsize=16, ha='center', va='center', fontweight='bold')
ax4.text(0.5, 0.45, 'Any tensor can be decomposed:\n\nA_μν = A_(μν) + A_[μν]\n\nSymmetric + Antisymmetric',
        transform=ax4.transAxes, fontsize=13, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.3))

properties_text = """
Properties:
• A_(μν) = A_(νμ)  (symmetric)
• A_[μν] = -A_[νμ]  (antisymmetric)
• A_[μμ] = 0  (diagonal is zero)
"""
ax4.text(0.5, 0.15, properties_text, transform=ax4.transAxes,
        fontsize=11, ha='center', va='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))

ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.axis('off')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Symmetric part: A_(μν) = A_(νμ)")
print(f"Check: {np.allclose(A_sym, A_sym.T)}")  # np.allclose() tests if all array elements are approximately equal
print("\nAntisymmetric part: A_[μν] = -A_[νμ]")
print(f"Check: {np.allclose(A_antisym, -A_antisym.T)}")  # np.allclose() tests if all array elements are approximately equal
print(f"Diagonal zeros: {np.allclose(np.diag(A_antisym), 0)}")  # np.allclose() tests if all array elements are approximately equal
```

---

### 🎯 Practice Question #3

**Q:** If T_μν is a (0,2)-tensor and we contract it to get S = T^μ_μ, what type of tensor is S?

<details>
<summary>💡 Hint</summary>

Contraction reduces the rank by 2 (removes one upper and one lower index).
</details>

<details>
<summary>✅ Answer</summary>

**S is a (0,0)-tensor, i.e., a scalar.**

Starting with T_μν (0,2)-tensor with 2 lower indices, we:
1. Raise one index using the metric: T^μ_ν
2. Contract: S = T^μ_μ = Σ_μ T^μ_μ

This removes both indices, leaving a scalar (single number).

This is the **trace** operation.

```python
T = np.array([[1, 2], [3, 4]])  # (0,2)-tensor

# To contract, we need metric to raise index
g_inv = np.eye(2)  # Inverse metric (identity for Euclidean)

# T^μ_ν = g^μρ T_ρν
T_mixed = g_inv @ T    # @ is matrix multiplication operator (equivalent to np.dot for matrices)

# Trace
S = np.trace(T_mixed)
print(f"Scalar S = T^μ_μ = {S}")  # 5
```
</details>

---

## 5. The Metric Tensor

### 📖 Concept

The **metric tensor** g_μν is the most important tensor in differential geometry and General Relativity.

**Definition:** The metric is a (0,2)-tensor that:
1. Takes two vectors and outputs their inner product
2. Is symmetric: g_μν = g_νμ
3. Is non-degenerate: det(g) ≠ 0

**In components:**
```
g_μν v^μ w^ν = inner product of v and w
```

**What it tells us:**
- **Distances:** ds² = g_μν dx^μ dx^ν
- **Angles:** cos(θ) = (g_μν v^μ w^ν) / (√(g_μν v^μ v^ν) √(g_ρσ w^ρ w^σ))
- **Volumes:** √|det(g)| dx¹...dxⁿ
- **ALL geometry** of the manifold!

**Examples:**

1. **Euclidean metric (flat space):**
```
g_μν = δ_μν = [1  0  0]
              [0  1  0]
              [0  0  1]
ds² = dx² + dy² + dz²
```

2. **Minkowski metric (flat spacetime):**
```
η_μν = [-1  0  0  0]
       [ 0  1  0  0]
       [ 0  0  1  0]
       [ 0  0  0  1]
ds² = -dt² + dx² + dy² + dz²
```

3. **Schwarzschild metric (around black hole):**
```
ds² = -(1 - 2M/r) dt² + (1 - 2M/r)⁻¹ dr² + r²(dθ² + sin²θ dφ²)
```

**Why this matters for GR:** Einstein's field equations determine how matter curves the metric g_μν. The metric encodes ALL gravitational effects!

---

### 💻 Code Example: Metrics and Distances

```python
# Euclidean metric in 3D
g_euclidean = np.eye(3)
print("Euclidean metric g_ij:")
print(g_euclidean)
print()

# Compute distance between two points
point_A = np.array([0, 0, 0])  # np.array() converts Python list/tuple to efficient numpy array
point_B = np.array([3, 4, 0])  # np.array() converts Python list/tuple to efficient numpy array

displacement = point_B - point_A
distance_squared = displacement @ g_euclidean @ displacement    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
distance = np.sqrt(distance_squared)  # np.sqrt() computes square root

print(f"Points A = {point_A}, B = {point_B}")
print(f"Displacement Δx = {displacement}")
print(f"Distance² = g_ij Δx^i Δx^j = {distance_squared}")
print(f"Distance = {distance}")
print(f"Expected: √(3² + 4²) = 5.0\n")

print("=" * 50)

# Minkowski metric (special relativity)
eta = np.diag([-1, 1, 1, 1])  # Signature (-,+,+,+)
print("Minkowski metric η_μν:")
print(eta)
print()

# Spacetime interval for two events
event_1 = np.array([0, 0, 0, 0])  # (t, x, y, z)
event_2 = np.array([5, 3, 4, 0])  # Time = 5, position = (3, 4, 0)

separation = event_2 - event_1
interval_squared = separation @ eta @ separation    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
interval = np.sqrt(np.abs(interval_squared))  # np.sqrt() computes square root

print(f"Event 1: {event_1}")
print(f"Event 2: {event_2}")
print(f"Separation Δx^μ = {separation}")
print(f"Interval² = η_μν Δx^μ Δx^ν = {interval_squared}")

if interval_squared < 0:
    print(f"Timelike separation (proper time τ = {interval:.3f})")
elif interval_squared > 0:
    print(f"Spacelike separation (proper distance)")
else:
    print(f"Lightlike (null) separation")

print("\n" + "=" * 50)

# Polar metric in 2D
def polar_metric(r, theta):
    """
    Metric in polar coordinates (r, θ)
    ds² = dr² + r² dθ²
    """
    return np.array([  # np.array() converts Python list/tuple to efficient numpy array
        [1, 0],
        [0, r**2]
    ])

r = 2.0
g_polar = polar_metric(r, 0)
print(f"Polar metric at r={r}:")
print(g_polar)
print()

# Distance along circular arc from θ=0 to θ=π/4 at fixed r=2
dtheta = np.pi / 4
arc_length = np.sqrt(g_polar[1, 1]) * dtheta  # √(r²) dθ = r dθ
print(f"Arc length from θ=0 to θ=π/4 at r={r}:")
print(f"s = r Δθ = {r} × {dtheta:.4f} = {arc_length:.4f}")
```

---

### 📊 Visualization: Metric Determines Geometry

```python
fig = plt.figure(figsize=(14, 10))  # plt.figure() creates a new figure for plotting

# Top: Euclidean metric (flat)
ax1 = plt.subplot(2, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
theta = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
r_values = [1, 2, 3]

for r in r_values:
    x = r * np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)
    y = r * np.sin(theta)  # np.sin() computes sine (element-wise for arrays)
    ax1.plot(x, y, linewidth=2, label=f'r = {r}')

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Euclidean Metric: ds² = dx² + dy²\n(Flat - circles are circular)',
             fontsize=12, fontweight='bold')
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax1.grid(True, alpha=0.3)
ax1.legend()

# Top-right: Spherical metric
ax2 = plt.subplot(2, 2, 2, projection='3d')  # plt.subplot() creates subplot in grid layout (rows, cols, position)
u = np.linspace(0, np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
v = np.linspace(0, 2*np.pi, 30)  # np.linspace() creates evenly spaced array between start and end
U, V = np.meshgrid(u, v)  # np.meshgrid() creates coordinate matrices from coordinate vectors

X = np.sin(U) * np.cos(V)  # np.sin() computes sine (element-wise for arrays)
Y = np.sin(U) * np.sin(V)  # np.sin() computes sine (element-wise for arrays)
Z = np.cos(U)  # np.cos() computes cosine (element-wise for arrays)

ax2.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')  # .plot_surface() draws 3D surface plot
ax2.set_title('Spherical Metric: ds² = R²(dθ² + sin²θ dφ²)\n(Curved surface)',
             fontsize=11, fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# Bottom-left: Hyperbolic metric
ax3 = plt.subplot(2, 2, 3)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
x_hyp = np.linspace(-0.95, 0.95, 30)  # np.linspace() creates evenly spaced array between start and end
y_hyp = np.linspace(-0.95, 0.95, 30)  # np.linspace() creates evenly spaced array between start and end
X_hyp, Y_hyp = np.meshgrid(x_hyp, y_hyp)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# Poincaré disk model (hyperbolic geometry)
# Metric: ds² = 4(dx² + dy²)/(1 - x² - y²)²

# Draw geodesics (circles perpendicular to boundary)
circle_centers = [0.5, 0, -0.5]
for center_x in circle_centers:
    if center_x == 0:
        # Straight line through origin
        ax3.plot([0, 0], [-0.95, 0.95], color=COLORS['blue'], linewidth=2)
    else:
        # Circular arc
        angles = np.linspace(-np.pi/2, np.pi/2, 100)  # np.linspace() creates evenly spaced array between start and end
        radius = 0.7
        x_circle = center_x + radius * np.cos(angles)  # np.cos() computes cosine (element-wise for arrays)
        y_circle = radius * np.sin(angles)  # np.sin() computes sine (element-wise for arrays)
        # Clip to unit disk
        mask = x_circle**2 + y_circle**2 < 0.95**2
        ax3.plot(x_circle[mask], y_circle[mask], color=COLORS['orange'], linewidth=2)

# Draw boundary circle
boundary_theta = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
ax3.plot(np.cos(boundary_theta), np.sin(boundary_theta),  # np.sin() computes sine (element-wise for arrays)
        'k--', linewidth=2, label='Boundary')

ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Hyperbolic Metric (Poincaré disk)\n(Negative curvature)',
             fontsize=12, fontweight='bold')
ax3.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-1.1, 1.1])
ax3.set_ylim([-1.1, 1.1])

# Bottom-right: Minkowski metric (spacetime diagram)
ax4 = plt.subplot(2, 2, 4)  # plt.subplot() creates subplot in grid layout (rows, cols, position)

# Draw light cones
t_vals = np.linspace(-3, 3, 100)  # np.linspace() creates evenly spaced array between start and end
ax4.plot(t_vals, t_vals, color=COLORS['orange'], linewidth=3, label='Light rays')
ax4.plot(-t_vals, t_vals, color=COLORS['orange'], linewidth=3)
ax4.fill_between(t_vals, -t_vals, t_vals, alpha=0.2, color=COLORS['yellow'],
                label='Timelike region')

# Worldline of massive particle
proper_time = np.linspace(0, 3, 50)  # np.linspace() creates evenly spaced array between start and end
x_particle = 0.6 * proper_time
t_particle = np.sqrt(proper_time**2 + x_particle**2)  # np.sqrt() computes square root
ax4.plot(x_particle, t_particle, color=COLORS['blue'], linewidth=3,
        label='Massive particle', linestyle='--')

ax4.set_xlabel('x (space)', fontsize=12)
ax4.set_ylabel('t (time)', fontsize=12)
ax4.set_title('Minkowski Metric: ds² = -dt² + dx²\n(Flat spacetime)',
             fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim([-3, 3])
ax4.set_ylim([0, 3])

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("The metric g_μν completely determines the geometry!")
```

---

### 🔬 Explore: Computing Christoffel Symbols

```python
# Christoffel symbols Γ^ρ_μν describe how basis vectors change
# Γ^ρ_μν = (1/2) g^ρσ (∂g_σμ/∂x^ν + ∂g_σν/∂x^μ - ∂g_μν/∂x^σ)

# Example: Polar coordinates (r, θ)
def metric_polar(r):
    """Metric in polar: diag(1, r²)"""
    return np.array([[1, 0],  # np.array() converts Python list/tuple to efficient numpy array
                     [0, r**2]])

def metric_inverse_polar(r):
    """Inverse metric: diag(1, 1/r²)"""
    return np.array([[1, 0],  # np.array() converts Python list/tuple to efficient numpy array
                     [0, 1/r**2]])

def christoffel_polar(r):
    """
    Christoffel symbols for polar coordinates
    Non-zero components:
    Γ^r_θθ = -r
    Γ^θ_rθ = Γ^θ_θr = 1/r
    """
    gamma = np.zeros((2, 2, 2))  # Γ^ρ_μν

    # Γ^r_θθ = -r (index: rho=0, mu=1, nu=1)
    gamma[0, 1, 1] = -r

    # Γ^θ_rθ = 1/r (index: rho=1, mu=0, nu=1)
    gamma[1, 0, 1] = 1/r

    # Γ^θ_θr = 1/r (index: rho=1, mu=1, nu=0)
    gamma[1, 1, 0] = 1/r

    return gamma

r = 2.0
gamma = christoffel_polar(r)

print(f"Christoffel symbols at r = {r}:")
print(f"Γ^r_θθ = {gamma[0, 1, 1]}")
print(f"Γ^θ_rθ = Γ^θ_θr = {gamma[1, 0, 1]}")
print("\nAll other components are zero")
print("\nThese describe how polar basis vectors change with position!")
```

---

## 6. Raising and Lowering Indices

### 📖 Concept

The metric g_μν and its inverse g^μν allow us to convert between contravariant and covariant indices.

**Lowering an index:**
```
v_μ = g_μν v^ν
```
Converts contravariant vector to covariant.

**Raising an index:**
```
v^μ = g^μν v_ν
```
Converts covariant to contravariant.

**Inverse metric:** Satisfies g^μρ g_ρν = δ^μ_ν (Kronecker delta)

**General rule:**
- Contract with g_μν to lower
- Contract with g^μν to raise

**Multiple indices:**
```
T^μ_ν = g^μρ T_ρν  (raise first index)
T_μν = g_μρ g_νσ T^ρσ  (lower both indices)
```

**Important:** The metric defines what we mean by "orthonormal" and "length"!

**Why this matters for GR:** We constantly raise and lower indices. For example:
- Ricci tensor: R_μν = R^ρ_μρν (contraction)
- Stress-energy: T^μν vs T_μν
- 4-velocity: u^μ = dx^μ/dτ, u_μ = g_μν u^ν

---

### 💻 Code Example: Raising and Lowering Indices

```python
# Euclidean space (3D)
g = np.array([[1, 0, 0],  # np.array() converts Python list/tuple to efficient numpy array
              [0, 1, 0],
              [0, 0, 1]])

g_inv = np.linalg.inv(g)  # For Euclidean, same as g

print("Euclidean metric g_ij:")
print(g)
print("\nInverse metric g^ij:")
print(g_inv)
print()

# Contravariant vector
v_upper = np.array([2, 3, 4])  # np.array() converts Python list/tuple to efficient numpy array
print(f"Contravariant vector v^i = {v_upper}")

# Lower index: v_i = g_ij v^j
v_lower = g @ v_upper    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
print(f"Covariant vector v_i = g_ij v^j = {v_lower}")

# Raise index back: v^i = g^ij v_j
v_upper_again = g_inv @ v_lower    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
print(f"Raised back v^i = g^ij v_j = {v_upper_again}")
print(f"Match: {np.allclose(v_upper, v_upper_again)}\n")  # np.allclose() tests if all array elements are approximately equal

print("=" * 50)

# Minkowski spacetime
eta = np.diag([-1, 1, 1, 1])
eta_inv = np.diag([-1, 1, 1, 1])  # Self-inverse!

print("Minkowski metric η_μν:")
print(eta)
print()

# 4-velocity (contravariant)
# For particle at rest: u^μ = (1, 0, 0, 0)
u_upper = np.array([1, 0, 0, 0])  # np.array() converts Python list/tuple to efficient numpy array
print(f"4-velocity u^μ = {u_upper}")

# Lower index
u_lower = eta @ u_upper    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
print(f"Lowered u_μ = η_μν u^ν = {u_lower}")
print("Note: u_0 = -1 (sign flip due to metric signature!)\n")

# Verify normalization: u^μ u_μ = -1 for timelike
norm = u_upper @ u_lower    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
print(f"Normalization u^μ u_μ = {norm}")
print("Expected: -1 for massive particle\n")

print("=" * 50)

# (0,2)-tensor → (1,1)-tensor → (2,0)-tensor
T_lower = np.array([[1, 2],  # np.array() converts Python list/tuple to efficient numpy array
                   [3, 4]])

g_2d = np.eye(2)
g_2d_inv = np.eye(2)

print("(0,2)-tensor T_μν:")
print(T_lower)
print()

# Raise first index: T^μ_ν = g^μρ T_ρν
T_mixed = g_2d_inv @ T_lower    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
print("(1,1)-tensor T^μ_ν = g^μρ T_ρν:")
print(T_mixed)
print()

# Raise second index: T^μν = g^νσ T^μ_σ
T_upper = T_mixed @ g_2d_inv    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
print("(2,0)-tensor T^μν = g^νσ T^μ_σ:")
print(T_upper)
print()

# Or directly: T^μν = g^μρ g^νσ T_ρσ
T_upper_direct = g_2d_inv @ T_lower @ g_2d_inv    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
print("Direct: T^μν = g^μρ g^νσ T_ρσ:")
print(T_upper_direct)
print(f"Match: {np.allclose(T_upper, T_upper_direct)}")  # np.allclose() tests if all array elements are approximately equal
```

---

### 📊 Visualization: Index Manipulation

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Metric and inverse
ax1 = axes[0, 0]
eta = np.diag([-1, 1, 1, 1])

im1 = ax1.imshow(eta, cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_title('Minkowski Metric η_μν', fontsize=14, fontweight='bold')
ax1.set_xticks([0, 1, 2, 3])
ax1.set_yticks([0, 1, 2, 3])
ax1.set_xticklabels(['0', '1', '2', '3'])
ax1.set_yticklabels(['0', '1', '2', '3'])

for i in range(4):
    for j in range(4):
        ax1.text(j, i, f'{eta[i,j]:.0f}', ha="center", va="center",
                color='white' if eta[i,j] != 0 else 'black', fontsize=14, fontweight='bold')

plt.colorbar(im1, ax=ax1)  # plt.colorbar() adds color scale bar to plot

# Top-right: Lowering index
ax2 = axes[0, 1]
ax2.text(0.5, 0.8, 'Lowering Index', transform=ax2.transAxes,
        fontsize=16, ha='center', fontweight='bold')

operation_text = """
Contravariant → Covariant

v^μ = [1, 0, 0, 0]

↓ (multiply by metric)

v_μ = η_μν v^ν

v_μ = [-1, 0, 0, 0]

Sign flip on time component!
"""

ax2.text(0.5, 0.4, operation_text, transform=ax2.transAxes,
        fontsize=12, ha='center', va='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['orange'], alpha=0.2))

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.axis('off')

# Bottom-left: Raising index
ax3 = axes[1, 0]
ax3.text(0.5, 0.8, 'Raising Index', transform=ax3.transAxes,
        fontsize=16, ha='center', fontweight='bold')

operation_text2 = """
Covariant → Contravariant

v_μ = [-1, 0, 0, 0]

↑ (multiply by inverse metric)

v^μ = η^μν v_ν

v^μ = [1, 0, 0, 0]

Back to original!
"""

ax3.text(0.5, 0.4, operation_text2, transform=ax3.transAxes,
        fontsize=12, ha='center', va='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.2))

ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.axis('off')

# Bottom-right: Summary
ax4 = axes[1, 1]
summary_text = """
Index Raising/Lowering Rules

Lowering (upper → lower):
T_...μ... = g_μν T_...^ν...

Raising (lower → upper):
T_...^μ... = g^μν T_...ν...

Metric properties:
• g_μρ g^ρν = δ_μ^ν (identity)
• Symmetric: g_μν = g_νμ
• Non-degenerate: det(g) ≠ 0

In Minkowski space:
η_μν = η^μν = diag(-1, 1, 1, 1)
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=11, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))

ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.axis('off')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("The metric g_μν is the 'musical isomorphism'")
print("It converts between vectors and one-forms")
```

---

### 🎯 Practice Question #4

**Q:** In Minkowski spacetime with metric η_μν = diag(-1,1,1,1), if u^μ = (2, 1, 0, 0), what is u_μ?

<details>
<summary>💡 Hint</summary>

Use u_μ = η_μν u^ν. Remember the metric is diagonal!
</details>

<details>
<summary>✅ Answer</summary>

u_μ = η_μν u^ν

Since η is diagonal:
- u_0 = η_00 u^0 = (-1)(2) = **-2**
- u_1 = η_11 u^1 = (1)(1) = **1**
- u_2 = η_22 u^2 = (1)(0) = **0**
- u_3 = η_33 u^3 = (1)(0) = **0**

So u_μ = **(-2, 1, 0, 0)**

```python
eta = np.diag([-1, 1, 1, 1])
u_upper = np.array([2, 1, 0, 0])  # np.array() converts Python list/tuple to efficient numpy array
u_lower = eta @ u_upper    # @ is matrix multiplication operator (equivalent to np.dot for matrices)
print(u_lower)  # [-2  1  0  0]
```

Note the sign flip on the time component due to the metric signature!
</details>

---

## 7. Examples: Tensors in Physics

### 📖 Concept

Tensors appear throughout physics:

**1. Stress-Energy Tensor T_μν (GR)**
- Describes matter and energy density
- Symmetric (0,2)-tensor
- T_00 = energy density
- T_0i = momentum density
- T_ij = stress (pressure/shear)
- Source term in Einstein field equations

**2. Electromagnetic Field Tensor F_μν**
- Antisymmetric (0,2)-tensor
- Encodes E and B fields
- F_0i = E_i/c
- F_ij = ε_ijk B_k
- Maxwell's equations: ∂_μ F^μν = J^ν

**3. Riemann Curvature Tensor R^ρ_σμν**
- (1,3)-tensor
- Describes spacetime curvature
- 20 independent components in 4D
- Encodes tidal forces

**4. Metric Tensor g_μν**
- Fundamental (0,2)-tensor
- Defines geometry
- Dynamical in GR (gravity = curved metric)

**5. Inertia Tensor I_ij (Classical Mechanics)**
- Relates angular momentum to angular velocity
- L_i = I_ij ω_j

**Why tensors?** Physics laws must be coordinate-independent. Tensor equations maintain form under coordinate transformations → same physics for all observers.

---

### 💻 Code Example: Electromagnetic Field Tensor

```python
# Electromagnetic field tensor F_μν
# F_μν is antisymmetric, encodes E and B fields

def em_field_tensor(E, B, c=1):
    """
    Construct F_μν from E and B fields
    Convention: F_0i = E_i/c, F_ij = -ε_ijk B_k

    Args:
        E: Electric field [Ex, Ey, Ez]
        B: Magnetic field [Bx, By, Bz]
        c: Speed of light (default 1 in natural units)

    Returns:
        4x4 antisymmetric tensor F_μν
    """
    F = np.zeros((4, 4))  # np.zeros() creates array filled with zeros

    # Time-space components: F_0i = E_i/c
    F[0, 1] = E[0] / c
    F[0, 2] = E[1] / c
    F[0, 3] = E[2] / c

    # Antisymmetry: F_i0 = -F_0i
    F[1, 0] = -F[0, 1]
    F[2, 0] = -F[0, 2]
    F[3, 0] = -F[0, 3]

    # Space-space components: F_ij = -ε_ijk B_k
    # F_12 = -B_z
    F[1, 2] = -B[2]
    F[2, 1] = -F[1, 2]

    # F_13 = B_y
    F[1, 3] = B[1]
    F[3, 1] = -F[1, 3]

    # F_23 = -B_x
    F[2, 3] = -B[0]
    F[3, 2] = -F[2, 3]

    return F

# Example: uniform E field in x-direction, B field in z-direction
E = np.array([3, 0, 0])  # E field along x
B = np.array([0, 0, 2])  # B field along z

F = em_field_tensor(E, B, c=1)

print("Electric field E:", E)
print("Magnetic field B:", B)
print("\nElectromagnetic field tensor F_μν:")
print(F)
print("\nVerify antisymmetry: F_μν = -F_νμ")
print(f"Check: {np.allclose(F, -F.T)}")  # np.allclose() tests if all array elements are approximately equal

# Invariants
I1 = 0.5 * np.einsum('ij,ij->', F, F)  # F_μν F^μν
print(f"\nFirst invariant: F_μν F^μν = {I1:.2f}")
print(f"Equals: 2(B² - E²/c²) = 2({np.dot(B,B):.0f} - {np.dot(E,E):.0f}) = {2*(np.dot(B,B) - np.dot(E,E)):.0f}")  # np.dot() computes dot product of two arrays
```

---

### 📊 Visualization: Stress-Energy Tensor

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Perfect fluid stress-energy tensor
ax1 = axes[0]

# T_μν = (ρ + p)u_μ u_ν + p η_μν
# For fluid at rest: u_μ = (-1, 0, 0, 0)
rho = 5  # Energy density
p = 1    # Pressure

T_fluid = np.diag([-rho, p, p, p])

im1 = ax1.imshow(T_fluid, cmap='RdBu_r', vmin=-6, vmax=6)
ax1.set_title('Perfect Fluid Stress-Energy T_μν\n(at rest)', fontsize=14, fontweight='bold')
ax1.set_xticks([0, 1, 2, 3])
ax1.set_yticks([0, 1, 2, 3])
ax1.set_xticklabels(['t', 'x', 'y', 'z'])
ax1.set_yticklabels(['t', 'x', 'y', 'z'])

for i in range(4):
    for j in range(4):
        text = ax1.text(j, i, f'{T_fluid[i,j]:.0f}', ha="center", va="center",
                       color='white' if abs(T_fluid[i,j]) > 3 else 'black',
                       fontsize=14, fontweight='bold')

plt.colorbar(im1, ax=ax1, label='Energy/Pressure')  # plt.colorbar() adds color scale bar to plot

# Add labels
ax1.text(0, -0.7, 'T_00 = -ρ\n(energy density)', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax1.text(2, 4.5, 'T_ii = p\n(pressure)', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Right: Electromagnetic stress-energy
ax2 = axes[1]

# For EM field: T_μν = (1/μ₀)[F_μα F^α_ν - (1/4)η_μν F_αβ F^αβ]
# Simplified for uniform E field

E_mag = 2
B_mag = 0
energy_density = 0.5 * (E_mag**2 + B_mag**2)
poynting = np.array([0, 0, 0])  # S = E × B
maxwell_stress = -0.5 * E_mag**2  # Negative pressure in field direction

T_em = np.diag([energy_density, maxwell_stress, maxwell_stress, maxwell_stress])

im2 = ax2.imshow(T_em, cmap='plasma', vmin=-3, vmax=3)
ax2.set_title('EM Field Stress-Energy T_μν', fontsize=14, fontweight='bold')
ax2.set_xticks([0, 1, 2, 3])
ax2.set_yticks([0, 1, 2, 3])
ax2.set_xticklabels(['t', 'x', 'y', 'z'])
ax2.set_yticklabels(['t', 'x', 'y', 'z'])

for i in range(4):
    for j in range(4):
        text = ax2.text(j, i, f'{T_em[i,j]:.1f}', ha="center", va="center",
                       color='white' if abs(T_em[i,j]) > 1.5 else 'black',
                       fontsize=14, fontweight='bold')

plt.colorbar(im2, ax=ax2, label='Energy/Stress')  # plt.colorbar() adds color scale bar to plot

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Stress-energy tensor sources gravity in Einstein's equations:")
print("G_μν = 8πG T_μν")
```

---

### 🔬 Explore: Computing Ricci Tensor from Christoffel Symbols

```python
# Ricci tensor: R_μν = ∂Γ^ρ_μν/∂x^ρ - ∂Γ^ρ_μρ/∂x^ν + Γ^ρ_ρλ Γ^λ_μν - Γ^ρ_μλ Γ^λ_ρν
# For polar coordinates (simple example)

def ricci_tensor_flat_2d():
    """
    For flat 2D space (polar coords), Ricci tensor should be zero
    R_μν = 0 (no intrinsic curvature)
    """
    # This is a simplified symbolic calculation
    # In flat space: R_μν = 0 everywhere

    R = np.zeros((2, 2))  # np.zeros() creates array filled with zeros
    return R

R_polar = ricci_tensor_flat_2d()
print("Ricci tensor in flat 2D polar coordinates:")
print(R_polar)
print("\nAll components zero → flat space (no curvature)")
print("\nFor curved spaces (like sphere), R_μν ≠ 0!")
```

---

## Practice Questions

Test your understanding:

### Basic Tensors

1. **Rank:** What is the rank of the tensor T^μνρ_σ?

2. **Components:** How many components does a (1,2)-tensor have in 3D?

3. **Summation:** Evaluate A^i_j B^j where A is a 3×3 matrix and B is a 3-vector.

### Operations

4. **Contraction:** What type of tensor results from contracting T^μν_μρ?

5. **Trace:** Compute the trace of [[2, 1], [3, 4]].

6. **Symmetrization:** Is the tensor T_μν = [[1, 2], [2, 1]] symmetric?

### Metric

7. **Raising:** In Euclidean space with g_ij = δ_ij, if v_i = (3, 4, 0), what is v^i?

8. **Minkowski:** For η_μν = diag(-1,1,1,1) and v^μ = (1, 1, 0, 0), compute v^μ v_μ.

9. **Inverse:** Verify that g_μρ g^ρν = δ_μ^ν for the 2D Euclidean metric.

---

## Summary and Next Steps

**Key Concepts Mastered:**
- Tensors as multilinear maps
- Index notation and Einstein summation
- Contravariant vs covariant indices
- Tensor operations (addition, product, contraction)
- Metric tensor g_μν
- Raising and lowering indices
- Physical examples (stress-energy, EM field)

**Connection to GR:**
- Einstein field equations: G_μν = 8πG T_μν
- Metric g_μν encodes spacetime curvature
- Riemann tensor R^ρ_σμν describes tidal forces
- All GR is tensor calculus on 4D manifolds

**Ready for:**
- Lesson 7: Riemannian Geometry (curvature tensors)
- Lesson 8: Classical Mechanics (Lagrangian formulation)
- Lesson 10: General Relativity Foundations

---

**Continue to:** → [Lesson 7: Riemannian Geometry](../07_riemannian_geometry/LESSON.md)
