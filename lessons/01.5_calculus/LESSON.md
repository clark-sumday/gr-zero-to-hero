# Lesson 1.5: Single-Variable Calculus

**Topics:** Limits, Derivatives, Integration, Fundamental Theorem, Exponentials, Logarithms

**Prerequisites:** Lesson 1 (Linear Algebra), high school algebra, basic trigonometry

**Time:** ~6-8 hours

---

## Table of Contents

1. [Functions and Limits](#1-functions-and-limits)
2. [Derivatives: The Rate of Change](#2-derivatives-the-rate-of-change)
3. [Differentiation Rules](#3-differentiation-rules)
4. [Integration: Accumulation and Area](#4-integration-accumulation-and-area)
5. [The Fundamental Theorem of Calculus](#5-the-fundamental-theorem-of-calculus)
6. [Exponentials and Logarithms](#6-exponentials-and-logarithms)
7. [Applications to Physics](#7-applications-to-physics)
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

**Why This Lesson Matters for GR:**
Calculus is the language of change and accumulation. In General Relativity, derivatives describe how spacetime curves, and integrals compute paths through curved space. This lesson builds the foundation you'll need for multivariable calculus, differential equations, and eventually Einstein's field equations.

---

## 1. Functions and Limits

### 📖 Concept

A **function** f(x) takes an input x and produces an output f(x). Think of it as a machine: you put in x, you get out f(x).

**Examples:**
- f(x) = x² (square the input)
- f(x) = 2x + 3 (double and add three)
- f(x) = sin(x) (take the sine)

A **limit** describes what happens to f(x) as x approaches some value a:

```
lim[x→a] f(x) = L
```

This means: "As x gets arbitrarily close to a, f(x) gets arbitrarily close to L."

**Key Insight:** Limits let us talk about "instantaneous" behavior - the foundation of derivatives.

**Why this matters for GR:** Limits formalize the concept of "infinitesimally small" intervals in spacetime, crucial for defining curvature at a point.

---

### 💻 Code Example: Visualizing Functions and Limits

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Define a simple function
def f(x):
    """f(x) = x² - 2x + 1"""
    return x**2 - 2*x + 1

# Create x values
x = np.linspace(-1, 3, 100)  # np.linspace() creates evenly spaced array between start and end
y = f(x)

# Plot the function
plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(x, y, color=COLORS['blue'], linewidth=2, label='f(x) = x² - 2x + 1')  # plt.plot() draws line plot

# Show limit as x → 1
x_approach = 1.0
plt.axvline(x=x_approach, color=COLORS['red'], linestyle='--', alpha=0.5, label=f'x = {x_approach}')  # plt.axvline() draws vertical line across plot
plt.plot(x_approach, f(x_approach), 'o', color=COLORS['orange'], markersize=10, label=f'f({x_approach}) = {f(x_approach)}')

plt.xlabel('x', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('f(x)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Function and Limit Visualization', fontsize=14, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend(fontsize=10)  # plt.legend() displays legend with labels
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.show()  # plt.show() displays the figure window

print(f"As x approaches {x_approach}, f(x) approaches {f(x_approach)}")
print(f"lim[x→{x_approach}] f(x) = {f(x_approach)}")
```

---

### 🔬 Explore: Numerical Limits

```python
# Compute limit numerically by approaching from both sides
def compute_limit(f, a, epsilon=1e-6):
    """Estimate limit of f(x) as x → a"""
    left = f(a - epsilon)  # Approach from left
    right = f(a + epsilon)  # Approach from right
    at_point = f(a)  # Value at the point
    return left, at_point, right

def g(x):
    """Another example: g(x) = (x² - 4)/(x - 2)"""
    if abs(x - 2) < 1e-10:
        return 4  # Limit value (we'll see why later)
    return (x**2 - 4) / (x - 2)

left, at, right = compute_limit(g, 2)
print(f"\nFor g(x) = (x² - 4)/(x - 2):")
print(f"  From left (x=1.999999): {left:.6f}")
print(f"  At x=2: {at:.6f}")
print(f"  From right (x=2.000001): {right:.6f}")
print(f"  Limit as x→2: {(left + right)/2:.6f}")
```

**What you should see:** The limit exists (≈4) even though we have to be careful at x=2.

---

### 🎯 Practice Question #1

**Q:** What is lim[x→3] (2x + 1)?

<details>
<summary>💡 Hint 1</summary>
Try substituting x = 3 directly into the expression.
</details>

<details>
<summary>💡 Hint 2</summary>
For polynomial functions, you can usually just plug in the value.
</details>

<details>
<summary>✅ Answer</summary>

lim[x→3] (2x + 1) = 2(3) + 1 = 7

This is a simple continuous function, so the limit equals the function value.
</details>

---

## 2. Derivatives: The Rate of Change

### 📖 Concept

The **derivative** of f(x) at x = a measures the **instantaneous rate of change** of f at that point.

**Definition:**
```
f'(a) = lim[h→0] [f(a + h) - f(a)] / h
```

**Geometric interpretation:** The derivative is the slope of the tangent line to f(x) at x = a.

**Physical interpretation:**
- If f(x) = position, then f'(x) = velocity
- If f(x) = velocity, then f'(x) = acceleration

**Notation:**
- f'(x) (Lagrange)
- df/dx (Leibniz) ← we'll use this most
- Df(x) (Euler)
- ḟ(x) (Newton, for time derivatives)

**Why this matters for GR:** Derivatives describe how quantities change. In GR, we use derivatives to describe how spacetime curves - the derivative of the metric tensor gives the Christoffel symbols, which encode gravitational effects.

---

### 💻 Code Example: Computing Derivatives Numerically

```python
def derivative_numerical(f, x, h=1e-5):
    """
    Compute derivative of f at x using finite difference approximation.
    f'(x) ≈ [f(x + h) - f(x)] / h
    """
    return (f(x + h) - f(x)) / h

# Example: f(x) = x²
def f(x):
    return x**2

x_val = 3.0
derivative_at_3 = derivative_numerical(f, x_val)

print(f"For f(x) = x²:")
print(f"  Numerical derivative at x = {x_val}: {derivative_at_3:.6f}")
print(f"  Analytical derivative: 2x = 2({x_val}) = {2*x_val}")
print(f"  (We expect 2x for the derivative of x²)")
```

---

### 📊 Visualization: Tangent Line (Derivative)

```python
def f(x):
    """f(x) = x² - 3x + 2"""
    return x**2 - 3*x + 2

def f_prime(x):
    """Derivative: f'(x) = 2x - 3"""
    return 2*x - 3

# Point to evaluate derivative
x0 = 2.0
y0 = f(x0)
slope = f_prime(x0)

# Create plot
x = np.linspace(-0.5, 4, 100)  # np.linspace() creates evenly spaced array between start and end
y = f(x)

# Tangent line: y - y0 = m(x - x0)
x_tangent = np.linspace(x0 - 1, x0 + 1, 100)  # np.linspace() creates evenly spaced array between start and end
y_tangent = y0 + slope * (x_tangent - x0)

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(x, y, color=COLORS['blue'], linewidth=2, label='f(x) = x² - 3x + 2')  # plt.plot() draws line plot
plt.plot(x_tangent, y_tangent, color=COLORS['red'], linewidth=2, label=f'Tangent line (slope = {slope})')  # plt.plot() draws line plot
plt.plot(x0, y0, 'o', color=COLORS['orange'], markersize=10, label=f'Point ({x0}, {y0})')

plt.xlabel('x', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('f(x)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title(f"Derivative as Tangent Line Slope\nf'({x0}) = {slope}", fontsize=14, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend(fontsize=10)  # plt.legend() displays legend with labels
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.axvline(x=0, color='k', linewidth=0.5)  # plt.axvline() draws vertical line across plot
plt.show()  # plt.show() displays the figure window

print(f"\nAt x = {x0}:")
print(f"  f({x0}) = {y0}")
print(f"  f'({x0}) = {slope}")
print(f"  This means f is changing at a rate of {slope} units per unit x")
```

**What you should see:** The red tangent line touches the blue curve at exactly one point and has the same slope as the curve at that point.

---

### 🎯 Practice Question #2

**Q:** Using the definition of derivative, find f'(x) for f(x) = 3x².

<details>
<summary>💡 Hint 1</summary>
Start with f'(x) = lim[h→0] [f(x + h) - f(x)] / h
</details>

<details>
<summary>💡 Hint 2</summary>
f(x + h) = 3(x + h)² = 3(x² + 2xh + h²) = 3x² + 6xh + 3h²
</details>

<details>
<summary>💡 Hint 3</summary>
[f(x + h) - f(x)] / h = [3x² + 6xh + 3h² - 3x²] / h = (6xh + 3h²) / h = 6x + 3h
</details>

<details>
<summary>✅ Answer</summary>

f'(x) = lim[h→0] [f(x + h) - f(x)] / h
      = lim[h→0] [3(x + h)² - 3x²] / h
      = lim[h→0] [3x² + 6xh + 3h² - 3x²] / h
      = lim[h→0] (6xh + 3h²) / h
      = lim[h→0] (6x + 3h)
      = 6x

So f'(x) = 6x.

**General pattern:** The derivative of ax^n is nax^(n-1).
</details>

---

## 3. Differentiation Rules

### 📖 Concept

Computing derivatives from the definition is tedious. Here are the **power rules** that make it easier:

**Basic Rules:**
1. **Constant Rule:** If f(x) = c, then f'(x) = 0
2. **Power Rule:** If f(x) = x^n, then f'(x) = nx^(n-1)
3. **Constant Multiple:** If f(x) = c·g(x), then f'(x) = c·g'(x)
4. **Sum Rule:** If f(x) = g(x) + h(x), then f'(x) = g'(x) + h'(x)

**Examples:**
- f(x) = 5 → f'(x) = 0
- f(x) = x³ → f'(x) = 3x²
- f(x) = 4x² → f'(x) = 8x
- f(x) = x³ + 2x² - 5x + 7 → f'(x) = 3x² + 4x - 5

**Advanced Rules:**
5. **Product Rule:** (fg)' = f'g + fg'
6. **Quotient Rule:** (f/g)' = (f'g - fg') / g²
7. **Chain Rule:** If h(x) = f(g(x)), then h'(x) = f'(g(x)) · g'(x)

**Why this matters for GR:** The chain rule is fundamental in differential geometry. When we change coordinates in curved spacetime, the chain rule tells us how derivatives transform - this leads directly to the concept of covariant derivatives!

---

### 💻 Code Example: Differentiation Rules

```python
# Verify differentiation rules numerically

def power_rule(n, x):
    """Derivative of x^n is n*x^(n-1)"""
    return n * x**(n-1)

def verify_derivative(f, f_prime_analytical, x, label):
    """Compare numerical and analytical derivatives"""
    numerical = derivative_numerical(f, x)
    analytical = f_prime_analytical(x)
    error = abs(numerical - analytical)

    print(f"{label}:")
    print(f"  Numerical:  {numerical:.8f}")
    print(f"  Analytical: {analytical:.8f}")
    print(f"  Error:      {error:.2e}\n")

# Test power rule: f(x) = x^5
verify_derivative(
    f=lambda x: x**5,
    f_prime_analytical=lambda x: 5*x**4,
    x=2.0,
    label="Power Rule: f(x) = x⁵"
)

# Test product rule: f(x) = x² · sin(x)
# f'(x) = 2x·sin(x) + x²·cos(x)
verify_derivative(
    f=lambda x: x**2 * np.sin(x),  # np.sin() computes sine (element-wise for arrays)
    f_prime_analytical=lambda x: 2*x*np.sin(x) + x**2*np.cos(x),  # np.cos() computes cosine (element-wise for arrays)
    x=1.5,
    label="Product Rule: f(x) = x²·sin(x)"
)

# Test chain rule: f(x) = (3x + 1)⁴
# f'(x) = 4(3x + 1)³ · 3 = 12(3x + 1)³
verify_derivative(
    f=lambda x: (3*x + 1)**4,
    f_prime_analytical=lambda x: 12*(3*x + 1)**3,
    x=2.0,
    label="Chain Rule: f(x) = (3x + 1)⁴"
)
```

---

### 📊 Visualization: Chain Rule

```python
# Visualize chain rule: h(x) = f(g(x))
# Example: h(x) = sin(x²)

def g(x):
    """Inner function: g(x) = x²"""
    return x**2

def f(u):
    """Outer function: f(u) = sin(u)"""
    return np.sin(u)  # np.sin() computes sine (element-wise for arrays)

def h(x):
    """Composite: h(x) = f(g(x)) = sin(x²)"""
    return f(g(x))

def h_prime(x):
    """Derivative via chain rule: h'(x) = f'(g(x)) · g'(x) = cos(x²) · 2x"""
    return np.cos(g(x)) * 2*x  # np.cos() computes cosine (element-wise for arrays)

x = np.linspace(-3, 3, 200)  # np.linspace() creates evenly spaced array between start and end

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # plt.subplots() creates figure with multiple subplots

# Left: The composite function
ax1.plot(x, h(x), color=COLORS['blue'], linewidth=2, label='h(x) = sin(x²)')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('h(x)', fontsize=12)
ax1.set_title('Composite Function: h(x) = sin(x²)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.axhline(y=0, color='k', linewidth=0.5)

# Right: Its derivative
ax2.plot(x, h_prime(x), color=COLORS['red'], linewidth=2, label="h'(x) = cos(x²)·2x")
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel("h'(x)", fontsize=12)
ax2.set_title("Derivative via Chain Rule", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.axhline(y=0, color='k', linewidth=0.5)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Chain Rule: If h(x) = f(g(x)), then h'(x) = f'(g(x)) · g'(x)")
print("For h(x) = sin(x²):")
print("  Inner: g(x) = x², so g'(x) = 2x")
print("  Outer: f(u) = sin(u), so f'(u) = cos(u)")
print("  Result: h'(x) = cos(x²) · 2x")
```

---

### 🎯 Practice Question #3

**Q:** Find the derivative of f(x) = (2x + 3)³ using the chain rule.

<details>
<summary>💡 Hint 1</summary>
Identify inner function g(x) = 2x + 3 and outer function f(u) = u³
</details>

<details>
<summary>💡 Hint 2</summary>
g'(x) = 2 and f'(u) = 3u²
</details>

<details>
<summary>✅ Answer</summary>

Using chain rule: h'(x) = f'(g(x)) · g'(x)

- Inner: g(x) = 2x + 3, so g'(x) = 2
- Outer: f(u) = u³, so f'(u) = 3u²
- Therefore: h'(x) = 3(2x + 3)² · 2 = 6(2x + 3)²

**Answer:** f'(x) = 6(2x + 3)²
</details>

---

## 4. Integration: Accumulation and Area

### 📖 Concept

**Integration** is the reverse of differentiation. It answers two related questions:

1. **Antiderivative:** What function F(x) has derivative f(x)?
   - If F'(x) = f(x), then F is an antiderivative of f
   - Notation: ∫f(x)dx = F(x) + C (indefinite integral)

2. **Area under curve:** What is the total accumulation of f(x) from x = a to x = b?
   - Notation: ∫[from a to b] f(x)dx (definite integral)

**Geometric interpretation:** The definite integral ∫[a to b] f(x)dx is the signed area between f(x) and the x-axis from x = a to x = b.

**Physical interpretation:**
- If f(x) = velocity, then ∫f(x)dx = position
- If f(x) = force, then ∫f(x)dx = work

**Basic integration rules** (reverse of differentiation):
- ∫x^n dx = x^(n+1)/(n+1) + C (for n ≠ -1)
- ∫sin(x)dx = -cos(x) + C
- ∫cos(x)dx = sin(x) + C
- ∫e^x dx = e^x + C

**Why this matters for GR:** Integration computes paths through spacetime (geodesics), total energy, and action. The Einstein-Hilbert action, which leads to Einstein's field equations, is an integral over curved spacetime!

---

### 💻 Code Example: Numerical Integration (Riemann Sums)

```python
def riemann_sum(f, a, b, n=100):
    """
    Approximate integral of f from a to b using Riemann sum.
    Divide [a,b] into n rectangles.
    """
    dx = (b - a) / n  # Width of each rectangle
    x_values = np.linspace(a, b, n)  # np.linspace() creates evenly spaced array between start and end
    y_values = f(x_values)

    # Sum up areas of rectangles: height × width
    area = np.sum(y_values) * dx  # np.sum() computes sum of array elements
    return area

# Example: integrate x² from 0 to 2
def f(x):
    return x**2

a, b = 0, 2
numerical_integral = riemann_sum(f, a, b, n=1000)

# Analytical: ∫x²dx = x³/3, so ∫[0,2] x²dx = 8/3 - 0 = 8/3
analytical_integral = (b**3 - a**3) / 3

print(f"Integral of x² from {a} to {b}:")
print(f"  Numerical (Riemann sum): {numerical_integral:.6f}")
print(f"  Analytical (x³/3):       {analytical_integral:.6f}")
print(f"  Error: {abs(numerical_integral - analytical_integral):.2e}")
```

---

### 📊 Visualization: Integration as Area Under Curve

```python
def f(x):
    """f(x) = x² - 2x + 3"""
    return x**2 - 2*x + 3

a, b = 0, 3  # Integration bounds

# Create x values and compute function
x = np.linspace(-0.5, 3.5, 200)  # np.linspace() creates evenly spaced array between start and end
y = f(x)

# Create filled region for integral
x_fill = np.linspace(a, b, 100)  # np.linspace() creates evenly spaced array between start and end
y_fill = f(x_fill)

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(x, y, color=COLORS['blue'], linewidth=2, label='f(x) = x² - 2x + 3')  # plt.plot() draws line plot
plt.fill_between(x_fill, 0, y_fill, alpha=0.3, color=COLORS['blue'], label=f'Area = ∫[{a},{b}] f(x)dx')

plt.axvline(x=a, color=COLORS['red'], linestyle='--', alpha=0.5, label=f'x = {a}')  # plt.axvline() draws vertical line across plot
plt.axvline(x=b, color=COLORS['red'], linestyle='--', alpha=0.5, label=f'x = {b}')  # plt.axvline() draws vertical line across plot

plt.xlabel('x', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('f(x)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Integration: Area Under Curve', fontsize=14, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend(fontsize=10)  # plt.legend() displays legend with labels
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.show()  # plt.show() displays the figure window

# Compute integral numerically
area = riemann_sum(f, a, b, n=1000)
print(f"\n∫[{a},{b}] (x² - 2x + 3)dx = {area:.6f}")
print(f"This is the blue shaded area!")
```

---

### 🎯 Practice Question #4

**Q:** Find ∫(3x² + 2x)dx.

<details>
<summary>💡 Hint 1</summary>
Use the power rule backwards: ∫x^n dx = x^(n+1)/(n+1) + C
</details>

<details>
<summary>💡 Hint 2</summary>
Integrate term by term: ∫(3x² + 2x)dx = ∫3x²dx + ∫2x dx
</details>

<details>
<summary>✅ Answer</summary>

∫(3x² + 2x)dx = ∫3x²dx + ∫2x dx
                = 3·(x³/3) + 2·(x²/2) + C
                = x³ + x² + C

**Answer:** x³ + x² + C

You can verify: d/dx(x³ + x² + C) = 3x² + 2x ✓
</details>

---

## 5. The Fundamental Theorem of Calculus

### 📖 Concept

The **Fundamental Theorem of Calculus** connects derivatives and integrals - it shows they are inverse operations!

**Part 1:** If F(x) = ∫[from a to x] f(t)dt, then F'(x) = f(x)
- Differentiating an integral gives back the original function

**Part 2:** If F'(x) = f(x), then ∫[from a to b] f(x)dx = F(b) - F(a)
- To compute a definite integral, find any antiderivative F and evaluate F(b) - F(a)

**This is profound:** It means we can compute areas (integrals) by finding antiderivatives!

**Why this matters for GR:** The Einstein-Hilbert action is an integral, and varying it (taking derivatives) gives Einstein's field equations. The fundamental theorem connects these operations, allowing us to derive the equations of General Relativity.

---

### 💻 Code Example: Fundamental Theorem in Action

```python
# Demonstrate Fundamental Theorem Part 2

def f(x):
    """f(x) = 2x"""
    return 2*x

def F(x):
    """Antiderivative: F(x) = x² (since F'(x) = 2x)"""
    return x**2

# Compute ∫[1,3] 2x dx two ways

# Method 1: Numerical integration (Riemann sum)
a, b = 1, 3
numerical = riemann_sum(f, a, b, n=1000)

# Method 2: Fundamental Theorem: F(b) - F(a)
analytical = F(b) - F(a)

print("Fundamental Theorem of Calculus:")
print(f"∫[{a},{b}] 2x dx = ?")
print(f"\nMethod 1 (Numerical): {numerical:.6f}")
print(f"Method 2 (FTC): F({b}) - F({a}) = {b}² - {a}² = {b**2} - {a**2} = {analytical}")
print(f"\nThey match! This is the power of the Fundamental Theorem.")
```

---

### 📊 Visualization: Accumulation Function

```python
# Show how ∫[0,x] f(t)dt accumulates area

def f(t):
    """f(t) = sin(t)"""
    return np.sin(t)  # np.sin() computes sine (element-wise for arrays)

def F(x):
    """Accumulation function: F(x) = ∫[0,x] sin(t)dt = -cos(x) + 1"""
    return -np.cos(x) + 1  # np.cos() computes cosine (element-wise for arrays)

t = np.linspace(0, 2*np.pi, 200)  # np.linspace() creates evenly spaced array between start and end

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # plt.subplots() creates figure with multiple subplots

# Top: Original function
ax1.plot(t, f(t), color=COLORS['blue'], linewidth=2, label='f(t) = sin(t)')
ax1.fill_between(t, 0, f(t), alpha=0.2, color=COLORS['blue'])
ax1.set_xlabel('t', fontsize=12)
ax1.set_ylabel('f(t)', fontsize=12)
ax1.set_title('Original Function f(t) = sin(t)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.axhline(y=0, color='k', linewidth=0.5)

# Bottom: Accumulation function (integral)
ax2.plot(t, F(t), color=COLORS['red'], linewidth=2, label='F(x) = ∫[0,x] sin(t)dt')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('F(x)', fontsize=12)
ax2.set_title('Accumulation Function F(x) = ∫[0,x] sin(t)dt', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.axhline(y=0, color='k', linewidth=0.5)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Key Insight:")
print("The slope of F(x) at any point equals f(x) at that point!")
print("This is the Fundamental Theorem: F'(x) = f(x)")
```

---

### 🎯 Practice Question #5

**Q:** Compute ∫[0,2] (x² + 1)dx using the Fundamental Theorem.

<details>
<summary>💡 Hint 1</summary>
First find an antiderivative F(x) of f(x) = x² + 1
</details>

<details>
<summary>💡 Hint 2</summary>
∫x²dx = x³/3 and ∫1 dx = x, so F(x) = x³/3 + x
</details>

<details>
<summary>✅ Answer</summary>

Step 1: Find antiderivative
F(x) = x³/3 + x  (since F'(x) = x² + 1)

Step 2: Apply Fundamental Theorem
∫[0,2] (x² + 1)dx = F(2) - F(0)
                    = (2³/3 + 2) - (0³/3 + 0)
                    = (8/3 + 2) - 0
                    = 8/3 + 6/3
                    = 14/3
                    ≈ 4.667

**Answer:** 14/3 or approximately 4.667
</details>

---

## 6. Exponentials and Logarithms

### 📖 Concept

The **exponential function** e^x and **natural logarithm** ln(x) are inverse functions central to calculus and physics.

**Key number:** e ≈ 2.71828... (Euler's number)

**Properties of e^x:**
- e^0 = 1
- e^(a+b) = e^a · e^b
- (e^x)' = e^x ← **The function is its own derivative!**
- ∫e^x dx = e^x + C

**Properties of ln(x):**
- ln(1) = 0
- ln(ab) = ln(a) + ln(b)
- ln(x^n) = n·ln(x)
- (ln(x))' = 1/x
- ∫(1/x)dx = ln|x| + C

**Why e is special:** e^x is the unique function that equals its own derivative. This makes it fundamental in differential equations describing exponential growth/decay.

**Why this matters for GR:** Exponential functions appear in:
- Schwarzschild metric: e^(2Φ/c²)
- Cosmological expansion: a(t) ~ e^(Ht)
- Solutions to wave equations in curved spacetime

---

### 💻 Code Example: Exponentials and Logarithms

```python
# Demonstrate properties of e^x and ln(x)

x = np.linspace(-2, 3, 100)  # np.linspace() creates evenly spaced array between start and end

# e^x and its derivative
y_exp = np.exp(x)  # np.exp() computes exponential e^x
y_exp_derivative = np.exp(x)  # np.exp() computes exponential e^x  # Same! e^x is its own derivative

# Plot exponential
plt.figure(figsize=(14, 5))  # plt.figure() creates a new figure for plotting

plt.subplot(1, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(x, y_exp, color=COLORS['blue'], linewidth=2, label='f(x) = e^x')  # plt.plot() draws line plot
plt.plot(x, y_exp_derivative, '--', color=COLORS['red'], linewidth=2, label="f'(x) = e^x (same!)")  # plt.plot() draws line plot
plt.xlabel('x', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('y', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Exponential: e^x is its own derivative', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend(fontsize=10)  # plt.legend() displays legend with labels
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.axvline(x=0, color='k', linewidth=0.5)  # plt.axvline() draws vertical line across plot

# ln(x) and its derivative
x_pos = np.linspace(0.1, 5, 100)  # np.linspace() creates evenly spaced array between start and end  # ln only defined for x > 0
y_ln = np.log(x_pos)  # np.log() computes natural logarithm
y_ln_derivative = 1 / x_pos  # Derivative of ln(x)

plt.subplot(1, 2, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(x_pos, y_ln, color=COLORS['blue'], linewidth=2, label='f(x) = ln(x)')  # plt.plot() draws line plot
plt.plot(x_pos, y_ln_derivative, '--', color=COLORS['red'], linewidth=2, label="f'(x) = 1/x")  # plt.plot() draws line plot
plt.xlabel('x', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('y', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Natural Log: (ln x)' + "' = 1/x", fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend(fontsize=10)  # plt.legend() displays legend with labels
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.axvline(x=0, color='k', linewidth=0.5)  # plt.axvline() draws vertical line across plot

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

# Numerical verification
x_test = 1.5
print(f"\nAt x = {x_test}:")
print(f"  e^x = {np.exp(x_test):.6f}")  # np.exp() computes exponential e^x
print(f"  d/dx(e^x) = {np.exp(x_test):.6f} (same!)")  # np.exp() computes exponential e^x
print(f"\n  ln(x) = {np.log(x_test):.6f}")  # np.log() computes natural logarithm
print(f"  d/dx(ln x) = 1/x = {1/x_test:.6f}")
```

---

### 🔬 Explore: Exponential Growth

```python
# Model exponential growth: N(t) = N₀ · e^(kt)

def exponential_growth(t, N0=1.0, k=0.5):
    """
    Exponential growth model.
    N0: initial amount
    k: growth rate
    """
    return N0 * np.exp(k * t)  # np.exp() computes exponential e^x

t = np.linspace(0, 5, 100)  # np.linspace() creates evenly spaced array between start and end

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
for k in [0.2, 0.5, 1.0, 1.5]:
    N = exponential_growth(t, N0=1.0, k=k)
    plt.plot(t, N, linewidth=2, label=f'k = {k}')  # plt.plot() draws line plot

plt.xlabel('Time t', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Amount N(t)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Exponential Growth: N(t) = e^(kt)', fontsize=14, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend(fontsize=10)  # plt.legend() displays legend with labels
plt.show()  # plt.show() displays the figure window

print("Exponential growth appears in:")
print("  - Population dynamics")
print("  - Radioactive decay (k < 0)")
print("  - Compound interest")
print("  - Cosmological expansion in de Sitter space!")
```

---

### 🎯 Practice Question #6

**Q:** Find the derivative of f(x) = x · e^x.

<details>
<summary>💡 Hint 1</summary>
Use the product rule: (fg)' = f'g + fg'
</details>

<details>
<summary>💡 Hint 2</summary>
Let f = x (so f' = 1) and g = e^x (so g' = e^x)
</details>

<details>
<summary>✅ Answer</summary>

Using product rule: (x · e^x)' = x' · e^x + x · (e^x)'
                                = 1 · e^x + x · e^x
                                = e^x + x·e^x
                                = e^x(1 + x)

**Answer:** f'(x) = e^x(1 + x)
</details>

---

## 7. Applications to Physics

### 📖 Concept

Calculus is the language of physics. Here are key applications you'll use in GR:

**1. Position, Velocity, Acceleration:**
- Position: x(t)
- Velocity: v(t) = dx/dt
- Acceleration: a(t) = dv/dt = d²x/dt²

**2. Work and Energy:**
- Work: W = ∫F(x)dx
- Kinetic Energy: KE = ½mv²
- Potential Energy: dU/dx = -F

**3. Differential Equations:**
Many physics laws are stated as differential equations:
- Newton's 2nd Law: F = ma = m(d²x/dt²)
- Einstein's Field Equations: Gμν = (8πG/c⁴)Tμν

**Why this matters for GR:**
- Geodesics are curves that extremize proper time (calculus of variations)
- Einstein's equations relate curvature (derivatives of metric) to energy
- Everything in GR involves derivatives and integrals in curved spacetime

---

### 💻 Code Example: Free Fall Under Gravity

```python
# Solve: d²x/dt² = -g (constant downward acceleration)

# Given: x(0) = h (initial height), v(0) = 0 (dropped from rest)
# Solution: x(t) = h - ½gt²

g = 9.8  # m/s² (Earth's gravity)
h = 100  # meters (initial height)

def position(t):
    """Position as function of time"""
    return h - 0.5 * g * t**2

def velocity(t):
    """Velocity: v = dx/dt = -gt"""
    return -g * t

def acceleration(t):
    """Acceleration: a = dv/dt = -g"""
    return -g * np.ones_like(t)  # np.ones_like() creates array of ones with same shape

# Time until hits ground: 0 = h - ½gt² → t = √(2h/g)
t_impact = np.sqrt(2 * h / g)  # np.sqrt() computes square root

t = np.linspace(0, t_impact, 100)  # np.linspace() creates evenly spaced array between start and end

fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # plt.subplots() creates figure with multiple subplots

# Position
axes[0].plot(t, position(t), color=COLORS['blue'], linewidth=2)
axes[0].set_xlabel('Time (s)', fontsize=11)
axes[0].set_ylabel('Height (m)', fontsize=11)
axes[0].set_title('Position x(t) = h - ½gt²', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linewidth=0.5)

# Velocity
axes[1].plot(t, velocity(t), color=COLORS['red'], linewidth=2)
axes[1].set_xlabel('Time (s)', fontsize=11)
axes[1].set_ylabel('Velocity (m/s)', fontsize=11)
axes[1].set_title('Velocity v(t) = dx/dt = -gt', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linewidth=0.5)

# Acceleration
axes[2].plot(t, acceleration(t), color=COLORS['orange'], linewidth=2)
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Acceleration (m/s²)', fontsize=11)
axes[2].set_title('Acceleration a(t) = dv/dt = -g', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=-g, color='k', linewidth=0.5, linestyle='--')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print(f"Object dropped from h = {h} m")
print(f"Hits ground at t = {t_impact:.2f} s")
print(f"Impact velocity: v = {velocity(t_impact):.2f} m/s")
print(f"\nThis uses calculus to solve Newton's F = ma!")
```

---

### 📊 Visualization: Simple Harmonic Oscillator

```python
# Solve: d²x/dt² = -ω²x (spring equation)
# Solution: x(t) = A·cos(ωt)

omega = 2.0  # Angular frequency
A = 1.0      # Amplitude

def x(t):
    """Position: x(t) = A·cos(ωt)"""
    return A * np.cos(omega * t)  # np.cos() computes cosine (element-wise for arrays)

def v(t):
    """Velocity: v(t) = dx/dt = -Aω·sin(ωt)"""
    return -A * omega * np.sin(omega * t)  # np.sin() computes sine (element-wise for arrays)

def a(t):
    """Acceleration: a(t) = dv/dt = -Aω²·cos(ωt) = -ω²x"""
    return -A * omega**2 * np.cos(omega * t)  # np.cos() computes cosine (element-wise for arrays)

t = np.linspace(0, 4*np.pi/omega, 200)  # np.linspace() creates evenly spaced array between start and end  # Two periods

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(t, x(t), color=COLORS['blue'], linewidth=2, label='Position x(t)')  # plt.plot() draws line plot
plt.plot(t, v(t), color=COLORS['red'], linewidth=2, label='Velocity v(t) = dx/dt')  # plt.plot() draws line plot
plt.plot(t, a(t), color=COLORS['orange'], linewidth=2, label='Acceleration a(t) = dv/dt')  # plt.plot() draws line plot

plt.xlabel('Time t', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Value', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Simple Harmonic Oscillator\n(Spring or Pendulum)', fontsize=14, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend(fontsize=10)  # plt.legend() displays legend with labels
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.show()  # plt.show() displays the figure window

print("Harmonic oscillator equation: d²x/dt² = -ω²x")
print("This describes:")
print("  - Mass on a spring")
print("  - Simple pendulum (small angles)")
print("  - Vibrations in molecules")
print("  - Gravitational waves! (small perturbations to flat spacetime)")
```

---

## Practice Questions

### Section 1-2: Limits and Derivatives

**Q1:** What is lim[x→2] (x² - 4)/(x - 2)?

<details><summary>Answer</summary>
This is indeterminate (0/0). Factor: (x² - 4)/(x - 2) = (x+2)(x-2)/(x-2) = x+2 for x≠2.
So lim[x→2] = 2+2 = 4.
</details>

**Q2:** Find f'(x) if f(x) = 4x³ - 3x² + 2x - 1.

<details><summary>Answer</summary>
f'(x) = 12x² - 6x + 2
</details>

### Section 3: Chain Rule

**Q3:** Find d/dx[sin(3x²)].

<details><summary>Answer</summary>
Chain rule: outer derivative × inner derivative
= cos(3x²) · 6x = 6x·cos(3x²)
</details>

### Section 4-5: Integration

**Q4:** Compute ∫(4x³ + 2x)dx.

<details><summary>Answer</summary>
∫(4x³ + 2x)dx = x⁴ + x² + C
</details>

**Q5:** Evaluate ∫[1,3] (2x + 1)dx.

<details><summary>Answer</summary>
Antiderivative: F(x) = x² + x
F(3) - F(1) = (9 + 3) - (1 + 1) = 12 - 2 = 10
</details>

### Section 6: Exponentials

**Q6:** Find d/dx[ln(x²)].

<details><summary>Answer</summary>
Method 1: Simplify first: ln(x²) = 2ln(x), so d/dx = 2/x
Method 2: Chain rule: 1/x² · 2x = 2/x
Both give: 2/x
</details>

---

## Summary and Next Steps

### What You've Learned

✅ **Limits:** Foundation of calculus, describing instantaneous behavior
✅ **Derivatives:** Rate of change, slope of tangent line
✅ **Differentiation Rules:** Power, product, quotient, chain rule
✅ **Integration:** Area under curves, antiderivatives
✅ **Fundamental Theorem:** Links derivatives and integrals
✅ **Exponentials/Logs:** e^x and ln(x) and their special properties
✅ **Physics Applications:** Position/velocity/acceleration, differential equations

### Calculus Notation Summary

- **Derivative:** f'(x), df/dx, dy/dx
- **Integral:** ∫f(x)dx (indefinite), ∫[a,b]f(x)dx (definite)
- **Limit:** lim[x→a] f(x)
- **Common symbols:** e (≈2.718), π (≈3.14159), ∞ (infinity)

### Connection to General Relativity

Everything you learned here extends to multiple dimensions:
- **Derivatives → Partial Derivatives** (∂/∂x, ∂/∂y, ∂/∂t)
- **Chain Rule → Tensor Transformations** (how quantities change under coordinate changes)
- **Integration → Path Integrals** (geodesics, action)
- **Exponentials → Metric Components** (e^(2Φ) in Schwarzschild solution)

**You're now ready for Lesson 2: Multivariable Calculus!** 🚀

---

## Need Help?

Use the AI assistant:

```python
from utils.ai_assistant import AIAssistant

assistant = AIAssistant()
assistant.set_lesson_context(
    "Lesson 1.5: Single-Variable Calculus",
    "Derivatives and Integration",
    ["limits", "derivatives", "chain rule", "integration", "fundamental theorem"]
)

# Ask questions
assistant.ask("Why is e^x its own derivative?")
assistant.ask("Can you give another example of the chain rule?")
assistant.ask("How does calculus connect to curved spacetime?")
```

---

**Next Lesson:** [Multivariable Calculus](../02_multivariable_calculus/LESSON.md) - Extending calculus to functions of multiple variables!
