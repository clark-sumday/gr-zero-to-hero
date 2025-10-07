# Lesson 9: Special Relativity

**Topics:** Lorentz Transformations, Spacetime Diagrams, Time Dilation, Length Contraction, 4-Vectors, Energy-Momentum, Causality
**Prerequisites:** Lessons 1-8 (Linear Algebra, Calculus, Classical Mechanics)
**Time:** ~6-8 hours

---

## Table of Contents

1. [The Postulates of Special Relativity](#1-the-postulates-of-special-relativity)
2. [Spacetime and the Interval](#2-spacetime-and-the-interval)
3. [Lorentz Transformations](#3-lorentz-transformations)
4. [Time Dilation and Length Contraction](#4-time-dilation-and-length-contraction)
5. [Spacetime Diagrams and Causality](#5-spacetime-diagrams-and-causality)
6. [Four-Vectors and Tensors](#6-four-vectors-and-tensors)
7. [Relativistic Energy and Momentum](#7-relativistic-energy-and-momentum)
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

## 1. The Postulates of Special Relativity

### 📖 Concept

Einstein's 1905 theory rests on two simple postulates:

**Postulate 1: Principle of Relativity**
The laws of physics are the same in all inertial reference frames.

**Postulate 2: Constancy of Light Speed**
The speed of light in vacuum (c ≈ 3×10⁸ m/s) is the same in all inertial frames, independent of the motion of the source or observer.

**Consequences** (derived, not assumed):
- Time dilation: Moving clocks run slow
- Length contraction: Moving objects contract along motion direction
- Relativity of simultaneity: Events simultaneous in one frame aren't in another
- Mass-energy equivalence: E = mc²
- Nothing with mass can reach speed c

**What's an inertial frame?** A reference frame moving at constant velocity (no acceleration) where Newton's first law holds.

**Why this matters for GR:**
- SR is the local approximation to GR (in small regions, spacetime is flat)
- GR extends SR to accelerated frames and gravity
- 4-vector formalism carries over directly to curved spacetime
- The metric in SR (Minkowski) → general metric g_μν in GR

---

### 💻 Code Example: Speed of Light Invariance

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Speed of light
c = 3e8  # m/s

# Observer velocities (as fraction of c)
v_observers = np.array([0, 0.3*c, 0.6*c, 0.9*c])  # np.array() converts Python list/tuple to efficient numpy array

# Classical prediction: speed of light depends on observer velocity
# If source emits at speed c, observer moving at v sees c ± v
classical_speeds = c - v_observers  # Observer moving toward source

# Relativistic reality: all observers measure same c
relativistic_speeds = c * np.ones_like(v_observers)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Classical (wrong!)
x_pos = np.arange(len(v_observers))
ax1.bar(x_pos, classical_speeds / c, color=COLORS['red'], alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{v/c:.1f}c' for v in v_observers])
ax1.set_xlabel('Observer velocity')
ax1.set_ylabel('Measured light speed (units of c)')
ax1.set_title('Classical Prediction\n(WRONG - velocity addition)')
ax1.axhline(y=1, color=COLORS['blue'], linestyle='--', linewidth=2, label='c')
ax1.set_ylim([0, 1.5])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Right: Relativistic (correct!)
ax2.bar(x_pos, relativistic_speeds / c, color=COLORS['green'], alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{v/c:.1f}c' for v in v_observers])
ax2.set_xlabel('Observer velocity')
ax2.set_ylabel('Measured light speed (units of c)')
ax2.set_title('Special Relativity\n(CORRECT - c is invariant)')
ax2.axhline(y=1, color=COLORS['blue'], linestyle='--', linewidth=2, label='c')
ax2.set_ylim([0, 1.5])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Special Relativity: ALL observers measure c = 3×10⁸ m/s")
print("This seems impossible with Galilean relativity, but it's experimentally verified!")
print("\nKey experiments:")
print("- Michelson-Morley (1887): No ether wind detected")
print("- Modern tests: GPS satellites (need SR + GR corrections!)")
```

---

### 🔬 Explore: Galilean vs Relativistic Velocity Addition

```python
def galilean_addition(u, v):
    """Classical velocity addition: w = u + v"""
    return u + v

def relativistic_addition(u, v, c=1.0):
    """
    Relativistic velocity addition:
    w = (u + v) / (1 + uv/c²)
    """
    return (u + v) / (1 + u*v/c**2)

# Example: Two velocities both 0.8c
u = 0.8  # units of c
v = 0.8

w_classical = galilean_addition(u, v)
w_relativistic = relativistic_addition(u, v, c=1.0)

print("Velocity Addition Example")
print("=" * 50)
print(f"Velocity 1: u = {u}c")
print(f"Velocity 2: v = {v}c")
print(f"\nClassical result: w = u + v = {w_classical}c  [WRONG! Exceeds c]")
print(f"Relativistic result: w = (u+v)/(1+uv/c²) = {w_relativistic:.3f}c  [Correct]")
print(f"\nNo matter how you add velocities, result is always < c")

# Plot velocity addition
u_vals = np.linspace(0, 0.99, 100)  # np.linspace() creates evenly spaced array between start and end
v_fixed = 0.8

w_classical_arr = u_vals + v_fixed
w_relativistic_arr = relativistic_addition(u_vals, v_fixed, c=1.0)

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(u_vals, w_classical_arr, color=COLORS['red'], linewidth=2,  # plt.plot() draws line plot
         label='Classical: w = u + v', linestyle='--')
plt.plot(u_vals, w_relativistic_arr, color=COLORS['green'], linewidth=2,  # plt.plot() draws line plot
         label='Relativistic: w = (u+v)/(1+uv/c²)')
plt.axhline(y=1, color=COLORS['blue'], linestyle=':', linewidth=2, label='Speed of light c')  # plt.axhline() draws horizontal line across plot
plt.xlabel('Velocity u (units of c)')  # plt.xlabel() sets x-axis label
plt.ylabel('Combined velocity w (units of c)')  # plt.ylabel() sets y-axis label
plt.title(f'Velocity Addition (v = {v_fixed}c fixed)')  # plt.title() sets plot title
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.xlim([0, 1])  # plt.xlim() sets x-axis limits
plt.ylim([0, 2])  # plt.ylim() sets y-axis limits
plt.show()  # plt.show() displays the figure window
```

---

## 2. Spacetime and the Interval

### 📖 Concept

In special relativity, space and time are unified into **spacetime** - a 4-dimensional manifold.

**Event:** A point in spacetime with coordinates (t, x, y, z) or (ct, x, y, z)

**The spacetime interval** between two events is:
```
Δs² = -c²Δt² + Δx² + Δy² + Δz²
```

or in natural units (c = 1):
```
Δs² = -Δt² + Δx² + Δy² + Δz²
```

**Key property:** The interval is **invariant** under Lorentz transformations!
All observers agree on Δs², even though they disagree on Δt and Δx separately.

**Classification of intervals:**

| Interval | Name | Physical Meaning |
|----------|------|------------------|
| Δs² < 0 | **Timelike** | Can be connected by particle (v < c) |
| Δs² = 0 | **Lightlike/Null** | Connected by light ray |
| Δs² > 0 | **Spacelike** | Cannot be causally connected |

**Minkowski metric** (flat spacetime):
```
η_μν = [[-1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]]
```

**⚠️ Metric Signature Convention:** This course uses the **(-,+,+,+) signature** (also called "mostly plus" or "West Coast" convention). Some books use (+,-,-,-) ("East Coast"). Both are valid—just stay consistent! We use (-,+,+,+) to match most GR textbooks (Carroll, Wald, Hartle).

---

### 💻 Code Example: Computing Spacetime Intervals

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations

def spacetime_interval(dt, dx, dy, dz, c=1.0):
    """
    Compute spacetime interval Δs².
    Uses signature (-,+,+,+).

    Returns:
    - Δs²
    - Classification: 'timelike', 'lightlike', or 'spacelike'
    """
    ds_squared = -(c * dt)**2 + dx**2 + dy**2 + dz**2

    if ds_squared < -1e-10:
        classification = 'timelike'
    elif abs(ds_squared) < 1e-10:
        classification = 'lightlike'
    else:
        classification = 'spacelike'

    return ds_squared, classification

# Examples (c = 1 for simplicity)
print("Spacetime Interval Examples")
print("=" * 60)

# Event 1: Particle moving at v = 0.5c
dt, dx, dy, dz = 2.0, 1.0, 0.0, 0.0
ds2, cls = spacetime_interval(dt, dx, dy, dz)
print(f"\nEvent pair 1: Δt = {dt}, Δx = {dx}")
print(f"Δs² = -Δt² + Δx² = {ds2:.3f}")
print(f"Classification: {cls.upper()}")
print(f"Can a particle travel between them? {cls == 'timelike'}")

# Event 2: Light ray
dt, dx = 1.0, 1.0
ds2, cls = spacetime_interval(dt, dx, 0, 0)
print(f"\nEvent pair 2: Δt = {dt}, Δx = {dx}")
print(f"Δs² = -Δt² + Δx² = {ds2:.3f}")
print(f"Classification: {cls.upper()}")
print(f"Light ray connects them? {cls == 'lightlike'}")

# Event 3: Spacelike separated
dt, dx = 0.5, 2.0
ds2, cls = spacetime_interval(dt, dx, 0, 0)
print(f"\nEvent pair 3: Δt = {dt}, Δx = {dx}")
print(f"Δs² = -Δt² + Δx² = {ds2:.3f}")
print(f"Classification: {cls.upper()}")
print(f"Causally connected? {cls != 'spacelike'}")

# Demonstrate invariance under Lorentz boost
print("\n" + "=" * 60)
print("INVARIANCE OF INTERVAL")
print("=" * 60)

def lorentz_boost_x(t, x, v, c=1.0):
    """Apply Lorentz boost in x-direction."""
    gamma = 1 / np.sqrt(1 - v**2/c**2)  # np.sqrt() computes square root
    t_prime = gamma * (t - v*x/c**2)
    x_prime = gamma * (x - v*t)
    return t_prime, x_prime

# Original events
t1, x1 = 0.0, 0.0
t2, x2 = 2.0, 1.0

dt_orig = t2 - t1
dx_orig = x2 - x1
ds2_orig, _ = spacetime_interval(dt_orig, dx_orig, 0, 0)

# Boost to moving frame (v = 0.6c)
v = 0.6
t1_prime, x1_prime = lorentz_boost_x(t1, x1, v)
t2_prime, x2_prime = lorentz_boost_x(t2, x2, v)

dt_prime = t2_prime - t1_prime
dx_prime = x2_prime - x1_prime
ds2_prime, _ = spacetime_interval(dt_prime, dx_prime, 0, 0)

print(f"Original frame: Δt = {dt_orig:.3f}, Δx = {dx_orig:.3f}")
print(f"                Δs² = {ds2_orig:.6f}")
print(f"\nBoosted frame (v = {v}c): Δt' = {dt_prime:.3f}, Δx' = {dx_prime:.3f}")
print(f"                          Δs'² = {ds2_prime:.6f}")
print(f"\nDifference: |Δs² - Δs'²| = {abs(ds2_orig - ds2_prime):.10f}")
print("The interval is INVARIANT! ✓")
```

---

### 📊 Visualization: Light Cone and Causal Structure

```python
# Visualize the light cone structure

fig = plt.figure(figsize=(14, 6))  # plt.figure() creates a new figure for plotting

# Left: 2D spacetime diagram
ax1 = fig.add_subplot(121)

# Light cone from origin
t_vals = np.linspace(0, 3, 100)  # np.linspace() creates evenly spaced array between start and end
x_future_plus = t_vals
x_future_minus = -t_vals
x_past_plus = -t_vals
x_past_minus = t_vals

ax1.fill_betweenx(t_vals, x_future_minus, x_future_plus,
                  alpha=0.3, color=COLORS['orange'], label='Future light cone')
ax1.fill_betweenx(-t_vals, x_past_minus, x_past_plus,
                  alpha=0.3, color=COLORS['blue'], label='Past light cone')

# Regions
ax1.text(0, 2, 'FUTURE\n(timelike)', ha='center', fontsize=11, weight='bold')
ax1.text(0, -2, 'PAST\n(timelike)', ha='center', fontsize=11, weight='bold')
ax1.text(2, 0, 'ELSEWHERE\n(spacelike)', ha='center', fontsize=10, style='italic')
ax1.text(-2, 0, 'ELSEWHERE\n(spacelike)', ha='center', fontsize=10, style='italic')

# Light rays
ax1.plot(t_vals, t_vals, color=COLORS['red'], linewidth=2, label='Light ray')
ax1.plot(-t_vals, t_vals, color=COLORS['red'], linewidth=2)
ax1.plot(t_vals, -t_vals, color=COLORS['red'], linewidth=2)
ax1.plot(-t_vals, -t_vals, color=COLORS['red'], linewidth=2)

# Sample worldlines
t_particle = np.linspace(-3, 3, 100)  # np.linspace() creates evenly spaced array between start and end
x_particle = 0.5 * t_particle  # v = 0.5c

ax1.plot(x_particle, t_particle, color=COLORS['green'], linewidth=2,
        linestyle='--', label='Particle (v < c)')

ax1.set_xlabel('Space x')
ax1.set_ylabel('Time t')
ax1.set_title('Light Cone Structure')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Right: 3D light cone
ax2 = fig.add_subplot(122, projection='3d')  projection='3d'  # Create 3D axes

# Create light cone mesh
theta = np.linspace(0, 2*np.pi, 50)  # np.linspace() creates evenly spaced array between start and end
t_cone = np.linspace(0, 2, 30)  # np.linspace() creates evenly spaced array between start and end
Theta, T = np.meshgrid(theta, t_cone)  # np.meshgrid() creates coordinate matrices from coordinate vectors

X = T * np.cos(Theta)  # np.cos() computes cosine (element-wise for arrays)
Y = T * np.sin(Theta)  # np.sin() computes sine (element-wise for arrays)
Z = T

# Future cone
ax2.plot_surface(X, Y, Z, alpha=0.4, color=COLORS['orange'])  # .plot_surface() draws 3D surface plot
# Past cone
ax2.plot_surface(X, Y, -Z, alpha=0.4, color=COLORS['blue'])  # .plot_surface() draws 3D surface plot

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('t')
ax2.set_title('3D Light Cone\n(Spacetime)')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("\nCausality:")
print("- Events in future light cone can be influenced by origin")
print("- Events in past light cone can influence origin")
print("- Events in 'elsewhere' are causally disconnected (no signal can reach)")
```

---

### 🎯 Practice Question #1

**Q:** Two events occur with Δt = 3 seconds and Δx = 4 light-seconds. What is the spacetime interval Δs²? Can a particle travel between them?

<details>
<summary>💡 Hint</summary>

Use Δs² = -c²Δt² + Δx² with c = 1 light-second/second.
</details>

<details>
<summary>✅ Answer</summary>

**Δs² = -c²(3)² + (4)² = -9 + 16 = 7 (light-seconds)²**

Since Δs² > 0, this is **spacelike** separated.

**No**, a particle cannot travel between them (would need v > c).

However, note that proper distance is √(Δs²) = √7 ≈ 2.65 light-seconds.
</details>

---

## 3. Lorentz Transformations

### 📖 Concept

**Lorentz transformations** relate coordinates in different inertial frames. They replace Galilean transformations and preserve the speed of light.

**Boost in x-direction** (frame S' moving at velocity v relative to S):

```
ct' = γ(ct - βx)
x'  = γ(x - vt) = γ(x - βct)
y'  = y
z'  = z
```

where:
- **β = v/c** (velocity as fraction of c)
- **γ = 1/√(1-β²)** (Lorentz factor)

**Matrix form:**
```
⎡ct'⎤   ⎡ γ  -βγ  0  0⎤ ⎡ct⎤
⎢x' ⎥ = ⎢-βγ  γ   0  0⎥ ⎢x ⎥
⎢y' ⎥   ⎢ 0   0   1  0⎥ ⎢y ⎥
⎣z' ⎦   ⎣ 0   0   0  1⎦ ⎣z ⎦
```

**Key properties:**
- Reduces to Galilean transformation when v << c (γ → 1, β → 0)
- Preserves spacetime interval: Δs'² = Δs²
- Compositions of boosts → another Lorentz transformation
- Forms a group: the Lorentz group SO(3,1)

---

### 💻 Code Example: Lorentz Transformation

```python
def lorentz_factor(v, c=1.0):
    """Compute Lorentz factor γ = 1/√(1-v²/c²)"""
    beta = v / c
    if abs(beta) >= 1:
        raise ValueError("Velocity must be less than c")
    return 1 / np.sqrt(1 - beta**2)  # np.sqrt() computes square root

def lorentz_transform(t, x, y, z, v, c=1.0):
    """
    Apply Lorentz boost in x-direction.

    Parameters:
    t, x, y, z: Event coordinates in frame S
    v: Velocity of frame S' relative to S
    c: Speed of light

    Returns:
    t', x', y', z': Transformed coordinates in frame S'
    """
    beta = v / c
    gamma = lorentz_factor(v, c)

    t_prime = gamma * (t - beta * x / c)
    x_prime = gamma * (x - v * t)
    y_prime = y
    z_prime = z

    return t_prime, x_prime, y_prime, z_prime

# Example: Transform an event
print("Lorentz Transformation Example")
print("=" * 60)

# Event in frame S
t, x, y, z = 5.0, 3.0, 0.0, 0.0  # units: seconds, light-seconds
v = 0.6  # S' moving at 0.6c relative to S
c = 1.0

gamma = lorentz_factor(v, c)
beta = v / c

print(f"Event in frame S: (t={t}, x={x}, y={y}, z={z})")
print(f"Frame S' velocity: v = {v}c")
print(f"Lorentz factor: γ = {gamma:.4f}")
print(f"Beta: β = {beta:.4f}")

# Transform
t_p, x_p, y_p, z_p = lorentz_transform(t, x, y, z, v, c)

print(f"\nEvent in frame S': (t'={t_p:.4f}, x'={x_p:.4f}, y'={y_p:.4f}, z'={z_p:.4f})")

# Verify interval invariance
ds2_S = -(c*t)**2 + x**2 + y**2 + z**2
ds2_Sp = -(c*t_p)**2 + x_p**2 + y_p**2 + z_p**2

print(f"\nInterval in S:  Δs² = {ds2_S:.6f}")
print(f"Interval in S': Δs² = {ds2_Sp:.6f}")
print(f"Difference: {abs(ds2_S - ds2_Sp):.10f} (should be ~0)")

# Matrix representation
print("\n" + "=" * 60)
print("Matrix Form")
print("=" * 60)

Lambda = np.array([  # np.array() converts Python list/tuple to efficient numpy array
    [ gamma, -beta*gamma, 0, 0],
    [-beta*gamma,  gamma, 0, 0],
    [ 0,      0,      1, 0],
    [ 0,      0,      0, 1]
])

print("Lorentz transformation matrix Λ:")
print(Lambda)

# Apply to 4-vector
x_four = np.array([c*t, x, y, z])  # np.array() converts Python list/tuple to efficient numpy array
x_four_prime = Lambda @ x_four    # @ is matrix multiplication operator (equivalent to np.dot for matrices)

print(f"\nFour-vector in S:  {x_four}")
print(f"Four-vector in S': {x_four_prime}")
print(f"Match individual calculation? {np.allclose([c*t_p, x_p, y_p, z_p], x_four_prime)}")  # np.allclose() tests if all array elements are approximately equal
```

---

### 📊 Visualization: Lorentz Transformation Effects

```python
# Visualize how Lorentz transformations affect spacetime coordinates

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Different boost velocities
velocities = [0.3, 0.6, 0.9]
colors_v = [COLORS['blue'], COLORS['orange'], COLORS['green']]

# Create grid of events in frame S
t_grid = np.linspace(0, 5, 20)  # np.linspace() creates evenly spaced array between start and end
x_grid = np.linspace(0, 5, 20)  # np.linspace() creates evenly spaced array between start and end

# Plot 1: Time transformation
ax = axes[0, 0]
for v, color in zip(velocities, colors_v):
    gamma = lorentz_factor(v)
    beta = v

    t_prime_vals = []
    for t in t_grid:
        # For x = 0
        t_p, _, _, _ = lorentz_transform(t, 0, 0, 0, v)
        t_prime_vals.append(t_p)

    ax.plot(t_grid, t_prime_vals, color=color, linewidth=2, label=f'v = {v}c')

ax.plot(t_grid, t_grid, 'k--', linewidth=1, label='t\' = t (v=0)')
ax.set_xlabel('Time in S: t')
ax.set_ylabel('Time in S\': t\'')
ax.set_title('Time Transformation (x=0)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Space transformation
ax = axes[0, 1]
for v, color in zip(velocities, colors_v):
    gamma = lorentz_factor(v)

    x_prime_vals = []
    for x in x_grid:
        # For t = 0
        _, x_p, _, _ = lorentz_transform(0, x, 0, 0, v)
        x_prime_vals.append(x_p)

    ax.plot(x_grid, x_prime_vals, color=color, linewidth=2, label=f'v = {v}c')

ax.plot(x_grid, x_grid, 'k--', linewidth=1, label='x\' = x (v=0)')
ax.set_xlabel('Position in S: x')
ax.set_ylabel('Position in S\': x\'')
ax.set_title('Space Transformation (t=0)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Lorentz factor vs velocity
ax = axes[1, 0]
v_vals = np.linspace(0, 0.99, 100)  # np.linspace() creates evenly spaced array between start and end
gamma_vals = [lorentz_factor(v) for v in v_vals]

ax.plot(v_vals, gamma_vals, color=COLORS['purple'], linewidth=3)
ax.set_xlabel('Velocity v (units of c)')
ax.set_ylabel('Lorentz factor γ')
ax.set_title('Lorentz Factor: γ = 1/√(1-v²/c²)')
ax.grid(True, alpha=0.3)
ax.set_ylim([1, 10])

# Annotate important points
ax.plot([0.6], [lorentz_factor(0.6)], 'o', color=COLORS['red'], markersize=10)
ax.text(0.6, lorentz_factor(0.6) + 0.5, f'v=0.6c: γ={lorentz_factor(0.6):.2f}', ha='center')

# Plot 4: Simultaneity (relativity of)
ax = axes[1, 1]

# Events simultaneous in S (t = 2)
t_simul = 2.0
x_simul = np.linspace(0, 5, 50)  # np.linspace() creates evenly spaced array between start and end

for v, color in zip(velocities, colors_v):
    t_prime_simul = []
    for x in x_simul:
        t_p, _, _, _ = lorentz_transform(t_simul, x, 0, 0, v)
        t_prime_simul.append(t_p)

    ax.plot(x_simul, t_prime_simul, color=color, linewidth=2, label=f'v = {v}c')

ax.axhline(y=t_simul, color='k', linestyle='--', linewidth=1, label=f'Simultaneous in S (t={t_simul})')
ax.set_xlabel('Position x')
ax.set_ylabel('Time in S\': t\'')
ax.set_title('Relativity of Simultaneity\n(Events simultaneous in S are NOT in S\')')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Key observations:")
print("1. Time dilation: γ > 1, so Δt' > Δt for moving clocks")
print("2. Length contraction: Moving lengths appear contracted by factor γ")
print("3. As v → c, γ → ∞ (infinite time dilation)")
print("4. Simultaneity is relative: events simultaneous in S are not in S'")
```

---

## 4. Time Dilation and Length Contraction

### 📖 Concept

**Time Dilation:** Moving clocks run slow.

If a clock is at rest in frame S' and measures proper time Δτ, an observer in S measures:
```
Δt = γΔτ
```

**Proper time** = time measured in the rest frame of the clock.

**Example:** Muon decay
- Muons created in upper atmosphere (15 km up)
- Lifetime at rest: τ₀ = 2.2 μs
- At v = 0.98c: should only travel d = vτ₀ ≈ 0.65 km
- But they reach Earth's surface! Why?
- Time dilation: γ ≈ 5, so lifetime in Earth frame = 5 × 2.2 μs = 11 μs
- Distance traveled: d = v(γτ₀) ≈ 3.2 km ✓

**Length Contraction:** Moving objects are shortened along direction of motion.

If a rod has proper length L₀ (length in its rest frame), an observer seeing it move at velocity v measures:
```
L = L₀/γ
```

**Proper length** = length measured in the rest frame of the object.

---

### 💻 Code Example: Time Dilation

```python
def time_dilation(proper_time, v, c=1.0):
    """
    Compute dilated time.

    Parameters:
    proper_time: Time in rest frame (Δτ)
    v: Velocity of moving frame
    c: Speed of light

    Returns:
    Dilated time in stationary frame (Δt)
    """
    gamma = lorentz_factor(v, c)
    return gamma * proper_time

# Example: Muon decay
print("TIME DILATION: Muon Decay")
print("=" * 60)

# Muon parameters
tau_0 = 2.2e-6  # seconds (proper lifetime)
v_muon = 0.98  # fraction of c
c_ms = 3e8  # m/s

gamma_muon = lorentz_factor(v_muon)
tau_lab = time_dilation(tau_0, v_muon)

# Distances
d_no_dilation = v_muon * c_ms * tau_0
d_with_dilation = v_muon * c_ms * tau_lab

print(f"Muon proper lifetime: τ₀ = {tau_0*1e6:.2f} μs")
print(f"Muon velocity: v = {v_muon}c")
print(f"Lorentz factor: γ = {gamma_muon:.3f}")
print(f"\nLifetime in lab frame: τ_lab = γτ₀ = {tau_lab*1e6:.2f} μs")
print(f"\nDistance without time dilation: {d_no_dilation:.1f} m")
print(f"Distance with time dilation: {d_with_dilation:.1f} m")
print(f"Atmosphere height: ~15,000 m")
print(f"\nConclusion: Time dilation explains why muons reach Earth!")

# GPS satellite example
print("\n" + "=" * 60)
print("GPS SATELLITES: Time Dilation Effects")
print("=" * 60)

# GPS orbit: ~20,000 km altitude, v ≈ 3.87 km/s
v_gps = 3870  # m/s
c_gps = 3e8  # m/s
beta_gps = v_gps / c_gps

gamma_gps = lorentz_factor(v_gps, c_gps)

# Time dilation per day
seconds_per_day = 86400
tau_satellite = seconds_per_day
t_earth = gamma_gps * tau_satellite

time_diff = (t_earth - tau_satellite) * 1e6  # microseconds

print(f"GPS satellite velocity: v = {v_gps} m/s = {beta_gps:.6f}c")
print(f"Lorentz factor: γ = {gamma_gps:.12f}")
print(f"\nTime difference per day (SR only): {time_diff:.3f} μs")
print(f"Effect on position: ~{time_diff * 1e-6 * c_gps / 1000:.1f} km per day!")
print(f"\nNote: GR gravitational time dilation is larger (~45 μs/day)")
print(f"GPS must correct for BOTH SR and GR effects!")
```

---

### 📊 Visualization: Twin Paradox

```python
# Visualize the twin paradox on a spacetime diagram

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Parameters
v = 0.6  # velocity of traveling twin
gamma = lorentz_factor(v)
t_total = 10  # total time for stay-at-home twin

# Stay-at-home twin (Earth)
t_earth = np.linspace(0, t_total, 100)  # np.linspace() creates evenly spaced array between start and end
x_earth = np.zeros_like(t_earth)

# Traveling twin
t_out = t_total / 2  # outbound time
t_back = t_total / 2  # return time

# Outbound leg
t_outbound = np.linspace(0, t_out, 50)  # np.linspace() creates evenly spaced array between start and end
x_outbound = v * t_outbound

# Return leg
t_return = np.linspace(t_out, t_total, 50)  # np.linspace() creates evenly spaced array between start and end
x_return = x_outbound[-1] - v * (t_return - t_out)

# Plot worldlines
ax1.plot(x_earth, t_earth, color=COLORS['blue'], linewidth=3, label='Stay-at-home twin')
ax1.plot(x_outbound, t_outbound, color=COLORS['red'], linewidth=3, label='Traveling twin (out)')
ax1.plot(x_return, t_return, color=COLORS['orange'], linewidth=3, label='Traveling twin (back)')

# Light cone
x_light = np.linspace(0, t_total, 100)  # np.linspace() creates evenly spaced array between start and end
ax1.plot(x_light, x_light, 'k--', alpha=0.3, linewidth=1, label='Light ray')
ax1.plot(-x_light, x_light, 'k--', alpha=0.3, linewidth=1)

ax1.set_xlabel('Position x')
ax1.set_ylabel('Time t')
ax1.set_title('Twin Paradox: Spacetime Diagram')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Calculate ages
tau_earth = t_total
tau_traveler = t_out / gamma + t_back / gamma  # proper time

print(f"\nTWIN PARADOX")
print("=" * 60)
print(f"Stay-at-home twin's age: {tau_earth:.3f} years")
print(f"Traveling twin's age: {tau_traveler:.3f} years")
print(f"Age difference: {tau_earth - tau_traveler:.3f} years")
print(f"\nThe traveling twin is younger!")

# Right plot: Aging as function of velocity
velocities = np.linspace(0, 0.99, 100)  # np.linspace() creates evenly spaced array between start and end
age_ratios = [1/lorentz_factor(v) for v in velocities]

ax2.plot(velocities, age_ratios, color=COLORS['green'], linewidth=3)
ax2.fill_between(velocities, 0, age_ratios, alpha=0.3, color=COLORS['green'])
ax2.set_xlabel('Velocity v (units of c)')
ax2.set_ylabel('Traveler age / Earth age')
ax2.set_title('Aging Ratio vs Velocity\n(for same trip)')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# Annotate
ax2.plot([0.6], [1/gamma], 'o', color=COLORS['red'], markersize=10)
ax2.text(0.6, 1/gamma - 0.1, f'v=0.6c\nratio={1/gamma:.2f}', ha='center')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🎯 Practice Question #2

**Q:** A spacecraft travels to Alpha Centauri (4.3 light-years away) at v = 0.9c. How much time passes on Earth? How much time for the astronauts?

<details>
<summary>💡 Hint 1</summary>

Earth time: Δt = distance / velocity. Astronaut time: Δτ = Δt / γ.
</details>

<details>
<summary>💡 Hint 2</summary>

Compute γ = 1/√(1-0.9²) first.
</details>

<details>
<summary>✅ Answer</summary>

**Earth time:** Δt = 4.3 / 0.9 = **4.78 years**

**Astronaut time:**
- γ = 1/√(1-0.81) = 1/√0.19 ≈ 2.29
- Δτ = Δt/γ = 4.78/2.29 ≈ **2.09 years**

The astronauts age only ~2 years while Earth ages ~5 years!

```python
v = 0.9
d = 4.3  # light-years
t_earth = d / v
gamma = lorentz_factor(v)
t_ship = t_earth / gamma
print(f"Earth: {t_earth:.2f} years, Ship: {t_ship:.2f} years")
```
</details>

---

## 5. Spacetime Diagrams and Causality

### 📖 Concept

**Spacetime diagram** (Minkowski diagram): Plot with time on vertical axis, space on horizontal axis.

**Worldline:** Path of an object through spacetime
- Vertical line: object at rest
- Diagonal line at 45°: light ray (c = 1)
- Slope > 1: massive particle (v < c)

**Causality:** The light cone structure determines what can influence what.

**Future light cone:** Events that can be influenced by event O
**Past light cone:** Events that can influence event O
**Elsewhere:** Events causally disconnected from O

**No faster-than-light signaling:** Prevents causality violations (no time travel paradoxes)

---

### 💻 Code Example: Spacetime Diagrams

```python
def plot_spacetime_diagram(events, worldlines=None, title="Spacetime Diagram"):
    """
    Plot a spacetime diagram with events and worldlines.

    Parameters:
    events: List of (t, x, label) tuples
    worldlines: List of (t_array, x_array, label) tuples
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw light cone from origin
    t_cone = np.linspace(-5, 5, 100)  # np.linspace() creates evenly spaced array between start and end
    ax.plot(t_cone, t_cone, 'k--', alpha=0.3, linewidth=1, label='Light rays')
    ax.plot(-t_cone, t_cone, 'k--', alpha=0.3, linewidth=1)

    # Shade regions
    ax.fill_between(t_cone[t_cone >= 0], t_cone[t_cone >= 0], 5,
                    alpha=0.1, color=COLORS['orange'], label='Future')
    ax.fill_between(t_cone[t_cone >= 0], -t_cone[t_cone >= 0], -5,
                    alpha=0.1, color=COLORS['orange'])

    ax.fill_between(t_cone[t_cone <= 0], t_cone[t_cone <= 0], -5,
                    alpha=0.1, color=COLORS['blue'], label='Past')
    ax.fill_between(t_cone[t_cone <= 0], -t_cone[t_cone <= 0], 5,
                    alpha=0.1, color=COLORS['blue'])

    # Plot events
    if events:
        for t, x, label in events:
            ax.plot([x], [t], 'o', markersize=10, color=COLORS['red'])
            ax.text(x + 0.2, t + 0.2, label, fontsize=11)

    # Plot worldlines
    if worldlines:
        colors_wl = [COLORS['green'], COLORS['purple'], COLORS['cyan']]
        for i, (t_arr, x_arr, label) in enumerate(worldlines):
            ax.plot(x_arr, t_arr, linewidth=2, color=colors_wl[i % len(colors_wl)],
                   label=label)

    ax.set_xlabel('Space x')
    ax.set_ylabel('Time t')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

    return fig, ax

# Example: Various scenarios
events = [
    (0, 0, 'O (origin)'),
    (2, 1, 'A (timelike)'),
    (1, 2, 'B (spacelike)'),
    (2, 2, 'C (lightlike)'),
]

# Worldlines
t_stationary = np.linspace(-3, 3, 100)  # np.linspace() creates evenly spaced array between start and end
x_stationary = np.zeros_like(t_stationary)

t_moving = np.linspace(-3, 3, 100)  # np.linspace() creates evenly spaced array between start and end
x_moving = 0.5 * t_moving

worldlines = [
    (t_stationary, x_stationary, 'Stationary observer'),
    (t_moving, x_moving, 'Moving observer (v=0.5c)'),
]

fig, ax = plot_spacetime_diagram(events, worldlines,
                                 "Causality and Light Cones")
plt.show()  # plt.show() displays the figure window

print("Causality Analysis:")
print("=" * 60)
print("Event A (2, 1): Inside future light cone → Can be reached from O")
print("Event B (1, 2): Outside light cone → Cannot be reached from O")
print("Event C (2, 2): On light cone → Reached by light signal from O")
```

---

### 📊 Visualization: Causality Violation (why FTL is impossible)

```python
# Show why faster-than-light travel leads to causality violations

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Simple FTL scenario
t_vals = np.linspace(0, 5, 100)  # np.linspace() creates evenly spaced array between start and end

# Alice's worldline (stationary)
ax1.plot([0]*len(t_vals), t_vals, color=COLORS['blue'], linewidth=2, label='Alice (rest)')

# Bob's worldline (moving at 0.6c)
x_bob = 0.6 * t_vals
ax1.plot(x_bob, t_vals, color=COLORS['orange'], linewidth=2, label='Bob (v=0.6c)')

# FTL signal from Alice to Bob
ax1.arrow(0, 2, 2, 0.5, head_width=0.2, head_length=0.2,
         fc=COLORS['red'], ec=COLORS['red'], linewidth=2)
ax1.text(1, 2.5, 'FTL signal', fontsize=11, color=COLORS['red'])

# Light cone
ax1.plot(t_vals, t_vals, 'k--', alpha=0.5, label='Light cone')
ax1.plot(-t_vals, t_vals, 'k--', alpha=0.5)

ax1.set_xlabel('Space x')
ax1.set_ylabel('Time t')
ax1.set_title('FTL Signal (Frame S)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-1, 5])
ax1.set_ylim([0, 5])
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Right: Same scenario in Bob's frame (shows time travel!)
# Transform to Bob's frame
v_bob = 0.6
gamma_bob = lorentz_factor(v_bob)

# Transform Alice's position
t_alice_prime = gamma_bob * t_vals
x_alice_prime = -gamma_bob * v_bob * t_vals

ax2.plot(x_alice_prime, t_alice_prime, color=COLORS['blue'],
        linewidth=2, label='Alice (in Bob frame)')

# Bob is stationary in his own frame
ax2.plot([0]*len(t_vals), t_vals, color=COLORS['orange'],
        linewidth=2, label='Bob (rest in own frame)')

# Transform FTL signal
# Original: t=2, x=0 to t=2.5, x=2
t1, x1 = 2.0, 0.0
t2, x2 = 2.5, 2.0

t1_p, x1_p, _, _ = lorentz_transform(t1, x1, 0, 0, v_bob)
t2_p, x2_p, _, _ = lorentz_transform(t2, x2, 0, 0, v_bob)

ax2.arrow(x1_p, t1_p, x2_p - x1_p, t2_p - t1_p,
         head_width=0.2, head_length=0.2,
         fc=COLORS['red'], ec=COLORS['red'], linewidth=2)

if t2_p < t1_p:
    ax2.text((x1_p + x2_p)/2, (t1_p + t2_p)/2 - 0.5,
            'BACKWARDS IN TIME!', fontsize=11, color=COLORS['red'],
            weight='bold', ha='center')

# Light cone in Bob's frame
ax2.plot(t_vals, t_vals, 'k--', alpha=0.5, label='Light cone')
ax2.plot(-t_vals, t_vals, 'k--', alpha=0.5)

ax2.set_xlabel('Space x\' (Bob frame)')
ax2.set_ylabel('Time t\' (Bob frame)')
ax2.set_title('FTL Signal (Frame S\' - Bob\'s rest frame)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-5, 5])
ax2.set_ylim([0, 5])
ax2.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("CAUSALITY VIOLATION:")
print("=" * 60)
print(f"In Alice's frame: signal sent at t={t1:.1f}, received at t={t2:.1f}")
print(f"In Bob's frame: signal sent at t'={t1_p:.2f}, received at t'={t2_p:.2f}")
if t2_p < t1_p:
    print("\n*** RECEIVED BEFORE IT WAS SENT! ***")
    print("This violates causality and creates logical paradoxes.")
    print("Conclusion: FTL travel is impossible.")
```

---

## 6. Four-Vectors and Tensors

### 📖 Concept

**Four-vector:** A quantity with 4 components that transforms like (ct, x, y, z) under Lorentz transformations.

**Contravariant four-vector:** x^μ = (x⁰, x¹, x², x³)
**Covariant four-vector:** x_μ = (x₀, x₁, x₂, x₃) = (-x⁰, x¹, x², x³)

**Index notation:**
- Greek indices (μ, ν, ...) run from 0 to 3 (spacetime)
- Latin indices (i, j, ...) run from 1 to 3 (space only)
- Raised index = contravariant
- Lowered index = covariant

**Minkowski metric:**
```
η_μν = diag(-1, 1, 1, 1)
```

Raises/lowers indices:
```
x_μ = η_μν x^ν
x^μ = η^μν x_ν
```

**Examples of four-vectors:**
- **Position:** x^μ = (ct, x, y, z)
- **Velocity:** u^μ = dx^μ/dτ = γ(c, v_x, v_y, v_z)
- **Momentum:** p^μ = mu^μ = (E/c, p_x, p_y, p_z)
- **Wavevector:** k^μ = (ω/c, k_x, k_y, k_z)

**Lorentz scalar** (invariant): p_μ p^μ = p₀² - p₁² - p₂² - p₃²

---

### 💻 Code Example: Four-Vectors

```python
# Minkowski metric
eta = np.array([  # np.array() converts Python list/tuple to efficient numpy array
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  1,  0],
    [ 0,  0,  0,  1]
])

def lower_index(x_contravariant, metric=eta):
    """Lower index: x_μ = η_μν x^ν"""
    return metric @ x_contravariant    # @ is matrix multiplication operator (equivalent to np.dot for matrices)

def raise_index(x_covariant, metric=eta):
    """Raise index: x^μ = η^μν x_ν"""
    metric_inv = np.linalg.inv(metric)  # np.linalg.inv() computes matrix inverse
    return metric_inv @ x_covariant    # @ is matrix multiplication operator (equivalent to np.dot for matrices)

def inner_product(x, y):
    """Compute Lorentz-invariant inner product: x·y = x_μ y^μ"""
    x_lower = lower_index(x)
    return np.dot(x_lower, y)  # np.dot() computes dot product of two arrays

# Example: Four-position
print("FOUR-VECTORS IN SPECIAL RELATIVITY")
print("=" * 60)

t, x, y, z = 2.0, 1.0, 0.5, 0.3
c = 1.0

x_four = np.array([c*t, x, y, z])  # np.array() converts Python list/tuple to efficient numpy array
print(f"Four-position (contravariant): x^μ = {x_four}")

x_four_covariant = lower_index(x_four)
print(f"Four-position (covariant): x_μ = {x_four_covariant}")

# Inner product with itself
x_dot_x = inner_product(x_four, x_four)
print(f"\nInner product: x·x = x_μ x^μ = {x_dot_x:.4f}")
print(f"This is the interval: Δs² = {x_dot_x:.4f}")

# Four-velocity
print("\n" + "=" * 60)
print("FOUR-VELOCITY")
print("=" * 60)

v_3d = np.array([0.6, 0.3, 0.0])  # 3-velocity
v_mag = np.linalg.norm(v_3d)  # np.linalg.norm() computes vector magnitude (Euclidean norm)
gamma = lorentz_factor(v_mag)

u_four = gamma * np.array([c, v_3d[0], v_3d[1], v_3d[2]])  # np.array() converts Python list/tuple to efficient numpy array
print(f"3-velocity: v = {v_3d}")
print(f"Speed: |v| = {v_mag:.3f}c")
print(f"Lorentz factor: γ = {gamma:.4f}")
print(f"\nFour-velocity: u^μ = γ(c, v) = {u_four}")

# Verify u·u = -c²
u_dot_u = inner_product(u_four, u_four)
print(f"\nInner product: u·u = {u_dot_u:.4f}")
print(f"Expected: -c² = {-c**2:.4f}")
print(f"Verification: u·u = -c² for all velocities! ✓")

# Four-momentum
print("\n" + "=" * 60)
print("FOUR-MOMENTUM")
print("=" * 60)

m = 1.0  # rest mass
p_four = m * u_four

E = p_four[0]  # Energy/c
p_3d = p_four[1:4]  # 3-momentum

print(f"Mass: m = {m}")
print(f"Four-momentum: p^μ = mu^μ = {p_four}")
print(f"Energy: E = p⁰c = {E*c:.4f}")
print(f"3-momentum: p = {p_3d}")

# Verify E² - p²c² = m²c⁴
p_dot_p = inner_product(p_four, p_four)
print(f"\nInvariant mass: p·p = -(E/c)² + p² = {p_dot_p:.4f}")
print(f"Expected: -m²c² = {-m**2 * c**2:.4f}")
print(f"This gives E² = (pc)² + (mc²)²")
```

---

### 📊 Visualization: Four-Vector Transformations

```python
# Show how four-vectors transform under boosts

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Four-velocity components vs 3-velocity
v_3d_vals = np.linspace(0, 0.99, 100)  # np.linspace() creates evenly spaced array between start and end
gamma_vals = [lorentz_factor(v) for v in v_3d_vals]

u0_vals = [gamma for gamma in gamma_vals]  # u⁰ = γc (c=1)
u1_vals = [gamma * v for gamma, v in zip(gamma_vals, v_3d_vals)]  # u¹ = γv

ax = axes[0, 0]
ax.plot(v_3d_vals, u0_vals, color=COLORS['blue'], linewidth=2, label='u⁰ = γc')
ax.plot(v_3d_vals, u1_vals, color=COLORS['orange'], linewidth=2, label='u¹ = γv')
ax.set_xlabel('3-velocity v (units of c)')
ax.set_ylabel('Four-velocity component')
ax.set_title('Four-Velocity Components')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 10])

# Four-velocity magnitude (always -c²)
u_dot_u_vals = [-u0**2 + u1**2 for u0, u1 in zip(u0_vals, u1_vals)]

ax = axes[0, 1]
ax.plot(v_3d_vals, u_dot_u_vals, color=COLORS['green'], linewidth=3)
ax.axhline(y=-1, color=COLORS['red'], linestyle='--', linewidth=2, label='u·u = -c² (invariant)')
ax.set_xlabel('3-velocity v (units of c)')
ax.set_ylabel('u·u')
ax.set_title('Four-Velocity Invariant: u·u = -c²')
ax.legend()
ax.grid(True, alpha=0.3)

# Energy-momentum relation
masses = [0.5, 1.0, 2.0]
colors_m = [COLORS['blue'], COLORS['orange'], COLORS['green']]

ax = axes[1, 0]
for m, color in zip(masses, colors_m):
    E_vals = []
    for v in v_3d_vals:
        gamma = lorentz_factor(v)
        E = gamma * m  # E = γmc² (c=1)
        E_vals.append(E)

    ax.plot(v_3d_vals, E_vals, color=color, linewidth=2, label=f'm = {m}')

ax.set_xlabel('Velocity v (units of c)')
ax.set_ylabel('Energy E (units of mc²)')
ax.set_title('Relativistic Energy: E = γmc²')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 10])

# Momentum-energy relation
ax = axes[1, 1]
p_vals = np.linspace(0, 5, 100)  # np.linspace() creates evenly spaced array between start and end

for m, color in zip(masses, colors_m):
    E_from_p = np.sqrt(p_vals**2 + m**2)  # E² = p² + m² (c=1)
    ax.plot(p_vals, E_from_p, color=color, linewidth=2, label=f'm = {m}')

# Massless particles (photons)
E_photon = p_vals
ax.plot(p_vals, E_photon, color=COLORS['red'], linewidth=2,
       linestyle='--', label='m = 0 (photon)')

ax.set_xlabel('Momentum p (units of mc)')
ax.set_ylabel('Energy E (units of mc²)')
ax.set_title('Energy-Momentum Relation: E² = (pc)² + (mc²)²')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Key properties of four-vectors:")
print("1. Transform as x'^μ = Λ^μ_ν x^ν under Lorentz boosts")
print("2. Inner products are Lorentz invariant: x·y = x_μ y^μ")
print("3. Four-velocity always has u·u = -c²")
print("4. Four-momentum always has p·p = -m²c²")
```

---

## 7. Relativistic Energy and Momentum

### 📖 Concept

**Relativistic energy:**
```
E = γmc²
```

**Relativistic momentum:**
```
p = γmv
```

**Energy-momentum relation:**
```
E² = (pc)² + (mc²)²
```

**For photons** (m = 0):
```
E = pc
```

**Key results:**

**Rest energy:**
```
E₀ = mc²
```

**Kinetic energy:**
```
K = E - mc² = (γ - 1)mc²
```

**Low-velocity limit** (v << c):
```
γ ≈ 1 + v²/(2c²)
K ≈ (1/2)mv²  [Newtonian!]
```

**Why this matters for GR:**
- Energy-momentum tensor T^μν is source of curvature
- Mass and energy both curve spacetime
- E = mc² verified in particle physics daily

---

### 💻 Code Example: Relativistic Energy and Momentum

```python
def relativistic_energy(m, v, c=1.0):
    """E = γmc²"""
    gamma = lorentz_factor(v, c)
    return gamma * m * c**2

def relativistic_momentum(m, v, c=1.0):
    """p = γmv"""
    gamma = lorentz_factor(v, c)
    return gamma * m * v

def kinetic_energy(m, v, c=1.0):
    """K = (γ-1)mc²"""
    gamma = lorentz_factor(v, c)
    return (gamma - 1) * m * c**2

# Example: Electron acceleration
print("RELATIVISTIC DYNAMICS: Electron Acceleration")
print("=" * 60)

m_electron = 9.109e-31  # kg
c_ms = 3e8  # m/s

velocities = [0.1, 0.5, 0.9, 0.99, 0.999]  # fractions of c

print(f"{'v/c':<8} {'γ':<10} {'E/mc²':<10} {'p/(mc)':<10} {'K/mc²':<10}")
print("-" * 60)

for v_frac in velocities:
    v = v_frac * c_ms
    gamma = lorentz_factor(v, c_ms)

    E = relativistic_energy(m_electron, v, c_ms)
    p = relativistic_momentum(m_electron, v, c_ms)
    K = kinetic_energy(m_electron, v, c_ms)

    E_norm = E / (m_electron * c_ms**2)
    p_norm = p / (m_electron * c_ms)
    K_norm = K / (m_electron * c_ms**2)

    print(f"{v_frac:<8.3f} {gamma:<10.4f} {E_norm:<10.4f} {p_norm:<10.4f} {K_norm:<10.4f}")

# Particle collision example
print("\n" + "=" * 60)
print("PARTICLE COLLISION: Conservation Laws")
print("=" * 60)

# Two particles colliding head-on
m1, m2 = 1.0, 1.0  # Equal masses
v1, v2 = 0.6, -0.6  # Opposite velocities (c=1)

# Before collision
gamma1 = lorentz_factor(v1)
gamma2 = lorentz_factor(abs(v2))

E1 = gamma1 * m1
E2 = gamma2 * m2
p1 = gamma1 * m1 * v1
p2 = gamma2 * m2 * v2

E_total = E1 + E2
p_total = p1 + p2

print("Before collision:")
print(f"Particle 1: E = {E1:.4f}, p = {p1:.4f}")
print(f"Particle 2: E = {E2:.4f}, p = {p2:.4f}")
print(f"Total: E_tot = {E_total:.4f}, p_tot = {p_total:.4f}")

# After collision: particles stick together
# Conservation: E_tot = γ_final M c², p_tot = γ_final M v_final
# For head-on collision with equal masses and velocities, p_tot = 0

M_final = E_total  # Since p_tot ≈ 0, γ ≈ 1, so M ≈ E_tot
v_final = 0  # Center of mass at rest

print(f"\nAfter collision:")
print(f"Combined particle: M = {M_final:.4f} (rest), v = {v_final}")
print(f"Energy conserved: {E_total:.4f} = {M_final:.4f} ✓")
print(f"Momentum conserved: {p_total:.4f} ≈ 0 ✓")
print(f"\nNote: Kinetic energy converted to rest mass!")
```

---

### 📊 Visualization: E = mc²

```python
# Visualize energy-momentum relationships

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Kinetic energy vs velocity
v_vals = np.linspace(0, 0.99, 200)  # np.linspace() creates evenly spaced array between start and end
m = 1.0
c = 1.0

K_relativistic = [(lorentz_factor(v) - 1) * m * c**2 for v in v_vals]
K_newtonian = [0.5 * m * v**2 for v in v_vals]

ax = axes[0, 0]
ax.plot(v_vals, K_relativistic, color=COLORS['blue'], linewidth=2,
       label='Relativistic: K = (γ-1)mc²')
ax.plot(v_vals, K_newtonian, color=COLORS['red'], linewidth=2,
       linestyle='--', label='Newtonian: K = ½mv²')
ax.set_xlabel('Velocity v (units of c)')
ax.set_ylabel('Kinetic Energy K (units of mc²)')
ax.set_title('Kinetic Energy: Relativistic vs Newtonian')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 10])

# Energy components
E_total = [lorentz_factor(v) * m * c**2 for v in v_vals]
E_rest = [m * c**2] * len(v_vals)

ax = axes[0, 1]
ax.fill_between(v_vals, 0, E_rest, alpha=0.5, color=COLORS['gray'],
               label='Rest energy mc²')
ax.fill_between(v_vals, E_rest, E_total, alpha=0.5, color=COLORS['orange'],
               label='Kinetic energy (γ-1)mc²')
ax.plot(v_vals, E_total, color=COLORS['blue'], linewidth=2,
       label='Total energy γmc²')
ax.set_xlabel('Velocity v (units of c)')
ax.set_ylabel('Energy (units of mc²)')
ax.set_title('Energy Composition: E = mc² + K')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 10])

# Energy-momentum diagram
p_vals = np.linspace(0, 5, 100)  # np.linspace() creates evenly spaced array between start and end

masses = [0, 0.5, 1.0, 2.0]
colors_mass = [COLORS['red'], COLORS['blue'], COLORS['orange'], COLORS['green']]

ax = axes[1, 0]
for mass, color in zip(masses, colors_mass):
    if mass == 0:
        E_vals = p_vals  # E = pc for photons
        label = 'm = 0 (photon)'
    else:
        E_vals = np.sqrt(p_vals**2 + mass**2)  # np.sqrt() computes square root
        label = f'm = {mass}'

    ax.plot(p_vals, E_vals, color=color, linewidth=2, label=label)

ax.set_xlabel('Momentum p (units of mc)')
ax.set_ylabel('Energy E (units of mc²)')
ax.set_title('Energy-Momentum: E² = (pc)² + (mc²)²')
ax.legend()
ax.grid(True, alpha=0.3)

# Force vs velocity (harder to accelerate at high v)
F_constant = 1.0  # Constant force
a_newtonian = F_constant / m  # a = F/m

a_relativistic = [F_constant / (lorentz_factor(v)**3 * m) for v in v_vals]

ax = axes[1, 1]
ax.plot(v_vals, [a_newtonian]*len(v_vals), color=COLORS['red'],
       linewidth=2, linestyle='--', label='Newtonian: a = F/m')
ax.plot(v_vals, a_relativistic, color=COLORS['blue'], linewidth=2,
       label='Relativistic: a = F/(γ³m)')
ax.set_xlabel('Velocity v (units of c)')
ax.set_ylabel('Acceleration a (for fixed force)')
ax.set_title('Why Nothing Reaches c:\nAcceleration → 0 as v → c')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 2])

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Key insights:")
print("1. Kinetic energy grows without bound as v → c")
print("2. Total energy E = mc² + K = γmc²")
print("3. For photons (m=0): E = pc (light has momentum!)")
print("4. Infinite energy needed to accelerate mass to c")
```

---

## 8. Practice Questions

### Basics

**Q1:** What are Einstein's two postulates of special relativity?

<details>
<summary>✅ Answer</summary>

1. **Principle of Relativity:** Laws of physics are the same in all inertial frames
2. **Constancy of Light Speed:** Speed of light c is the same in all inertial frames

These two simple assumptions lead to all of SR!
</details>

---

**Q2:** Calculate the Lorentz factor γ for v = 0.8c.

<details>
<summary>💡 Hint</summary>

γ = 1/√(1 - v²/c²)
</details>

<details>
<summary>✅ Answer</summary>

**γ = 1/√(1 - 0.64) = 1/√0.36 = 1/0.6 = 5/3 ≈ 1.667**

```python
v = 0.8
gamma = 1 / np.sqrt(1 - v**2)  # np.sqrt() computes square root
print(f"γ = {gamma:.3f}")
```
</details>

---

### Spacetime Interval

**Q3:** Two events have Δt = 5 s and Δx = 3 light-seconds. What is Δs²? Is it timelike, lightlike, or spacelike?

<details>
<summary>✅ Answer</summary>

**Δs² = -(5)² + (3)² = -25 + 9 = -16** (light-seconds)²

Since Δs² < 0, it's **timelike**.

Proper time: τ = √(-Δs²) = 4 seconds
</details>

---

### Time Dilation & Length Contraction

**Q4:** A muon has lifetime τ₀ = 2.2 μs in its rest frame. At v = 0.95c, what lifetime do we measure in the lab?

<details>
<summary>💡 Hint</summary>

Δt = γΔτ where γ = 1/√(1 - 0.95²)
</details>

<details>
<summary>✅ Answer</summary>

**τ_lab = γτ₀**

γ = 1/√(1 - 0.95²) = 1/√0.0975 ≈ 3.20

τ_lab = 3.20 × 2.2 μs ≈ **7.04 μs**

```python
tau_0 = 2.2e-6
v = 0.95
gamma = lorentz_factor(v)
tau_lab = gamma * tau_0
print(f"Lab lifetime: {tau_lab*1e6:.2f} μs")
```
</details>

---

**Q5:** A meter stick moves past you at 0.6c. How long does it appear?

<details>
<summary>💡 Hint</summary>

L = L₀/γ (length contraction)
</details>

<details>
<summary>✅ Answer</summary>

**L = L₀/γ**

γ = 1/√(1 - 0.36) = 1.25

L = 1 m / 1.25 = **0.8 m**

The meter stick appears 80 cm long!
</details>

---

### Four-Vectors

**Q6:** What is the inner product of the four-velocity with itself?

<details>
<summary>✅ Answer</summary>

**u·u = u_μ u^μ = -c²**

This is true for ALL velocities! It's a Lorentz invariant.

Proof:
```
u^μ = γ(c, v)
u·u = -γ²c² + γ²v² = γ²(v² - c²) = c²(v² - c²)/(c² - v²) = -c²
```
</details>

---

**Q7:** A photon has energy E. What is its momentum?

<details>
<summary>💡 Hint</summary>

For photons, m = 0, so E² = (pc)².
</details>

<details>
<summary>✅ Answer</summary>

**p = E/c**

From E² = (pc)² + (mc²)² with m = 0:
E = pc, so p = E/c

Example: Red photon (λ = 700 nm):
- E = hc/λ ≈ 2.8 × 10⁻¹⁹ J
- p = E/c ≈ 9.4 × 10⁻²⁸ kg·m/s

Light has momentum! (Used in solar sails)
</details>

---

### Energy-Momentum

**Q8:** A particle has rest mass m = 1 GeV/c² and kinetic energy K = 3 GeV. What is its total energy and momentum?

<details>
<summary>💡 Hint</summary>

E = mc² + K, then use E² = (pc)² + (mc²)²
</details>

<details>
<summary>✅ Answer</summary>

**E = mc² + K = 1 + 3 = 4 GeV**

**p = √(E² - (mc²)²)/c = √(16 - 1)/c = √15/c ≈ 3.87 GeV/c**

```python
m_c2 = 1  # GeV
K = 3     # GeV
E = m_c2 + K
p_c = np.sqrt(E**2 - m_c2**2)  # np.sqrt() computes square root
print(f"E = {E} GeV, pc = {p_c:.2f} GeV")
```
</details>

---

## Summary: The Bridge to General Relativity

Special Relativity is the foundation for General Relativity:

| SR Concept | GR Generalization |
|------------|-------------------|
| Minkowski metric η_μν | General metric g_μν(x) |
| Lorentz transformations | General coordinate transformations |
| Flat spacetime | Curved spacetime |
| Inertial frames | Locally inertial frames |
| Global coordinates | Local coordinates |
| ∂_μ derivatives | ∇_μ covariant derivatives |
| No gravity | Gravity = spacetime curvature |

**Next:** Learn how to describe curved spacetime and Einstein's field equations!

---

## Next Steps

✅ Master Lorentz transformations and spacetime intervals
✅ Practice computing time dilation and length contraction
✅ Understand four-vector formalism
✅ Work with relativistic energy and momentum
✅ **Ready for Lesson 10: Foundations of General Relativity** (Equivalence principle, curved spacetime)

**Additional Resources:**
- Einstein's original 1905 paper "On the Electrodynamics of Moving Bodies"
- Taylor & Wheeler: "Spacetime Physics" (Excellent intro)
- Susskind's "Special Relativity and Classical Field Theory" (Videos)
- MinutePhysics: "Special Relativity" series (YouTube)

---

**Ready to continue?** → [Lesson 10: Foundations of General Relativity](../10_gr_foundations/LESSON.md)
