# Lesson 11: Solutions to Einstein's Equations

**Topics:** Schwarzschild Solution, Black Holes, Kerr Solution, FLRW Cosmology, Gravitational Waves, de Sitter Space

**Prerequisites:** Lesson 10 (GR Foundations), Lessons 6-7 (Tensors and Riemannian Geometry)

**Time:** ~6-7 hours

---

## Table of Contents

1. [The Schwarzschild Solution](#1-the-schwarzschild-solution)
2. [Black Holes and Event Horizons](#2-black-holes-and-event-horizons)
3. [The Kerr Solution: Rotating Black Holes](#3-the-kerr-solution-rotating-black-holes)
4. [FLRW Cosmology: The Expanding Universe](#4-flrw-cosmology-the-expanding-universe)
5. [Gravitational Wave Solutions](#5-gravitational-wave-solutions)
6. [de Sitter and Anti-de Sitter Spaces](#6-de-sitter-and-anti-de-sitter-spaces)
7. [Other Important Solutions](#7-other-important-solutions)
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

**⚠️ Metric Signature Convention:** All solutions in this lesson use the **(-,+,+,+) signature** for the spacetime metric. This is consistent with Lessons 7, 9, and 10, and matches most modern GR textbooks.

---

## 1. The Schwarzschild Solution

### 📖 Concept

The **Schwarzschild solution** (1916) is the first and most important exact solution to Einstein's field equations. It describes the spacetime geometry around a **spherically symmetric, non-rotating mass** (like a star or non-rotating black hole).

**The Schwarzschild Metric:**

In Schwarzschild coordinates (t, r, θ, φ):

```
ds² = -(1 - r_s/r)c²dt² + (1 - r_s/r)^(-1) dr² + r²(dθ² + sin²θ dφ²)
```

Where the **Schwarzschild radius** is:

```
r_s = 2GM/c²
```

**Key Properties:**
- **Asymptotically flat:** As r → ∞, metric → Minkowski (flat spacetime)
- **Static:** Independent of time coordinate t
- **Spherically symmetric:** Same in all angular directions
- **Vacuum solution:** T_μν = 0 everywhere except at r = 0
- **Unique:** Birkhoff's theorem says this is the only spherically symmetric vacuum solution

**Physical Interpretation:**
- For r >> r_s: Weak field limit, recovers Newtonian gravity
- For r ~ r_s: Strong curvature effects
- At r = r_s: Event horizon (for black holes)
- At r = 0: Singularity (curvature → ∞)

---

### 💻 Code Example: Schwarzschild Metric Components

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

def schwarzschild_metric(r, M, c=1, G=1):
    """
    Compute Schwarzschild metric components.

    Parameters:
    r: radial coordinate
    M: mass
    c: speed of light (default 1 for geometric units)
    G: gravitational constant (default 1 for geometric units)

    Returns:
    g_tt, g_rr, g_theta_theta, g_phi_phi
    """
    r_s = 2 * G * M / c**2  # Schwarzschild radius

    if r <= r_s:
        # Inside horizon - coordinates become singular
        return None, None, None, None

    g_tt = -(1 - r_s/r)
    g_rr = 1 / (1 - r_s/r)
    g_theta_theta = r**2
    g_phi_phi = r**2  # Will be multiplied by sin²θ

    return g_tt, g_rr, g_theta_theta, g_phi_phi

# Example: Solar mass black hole (in geometric units: G=c=1)
M_sun = 1.0  # Mass in solar masses
r_s_sun = 2 * M_sun  # Schwarzschild radius

print(f"Schwarzschild Solution for M = {M_sun} solar mass")
print(f"Schwarzschild radius: r_s = {r_s_sun:.2f} km (geometric units)")
print(f"(In SI: r_s ≈ 3 km for solar mass)\n")

# Compute metric at various radii
radii = [10*r_s_sun, 5*r_s_sun, 2*r_s_sun, 1.5*r_s_sun]

for r in radii:
    g_tt, g_rr, g_theta_theta, _ = schwarzschild_metric(r, M_sun)
    if g_tt is not None:
        print(f"At r = {r/r_s_sun:.1f} r_s:")
        print(f"  g_tt = {g_tt:.4f}  (time dilation factor)")
        print(f"  g_rr = {g_rr:.4f}  (radial distance stretching)")
        print()
```

**Expected output:**
```
Schwarzschild Solution for M = 1.0 solar mass
Schwarzschild radius: r_s = 2.00 km (geometric units)
(In SI: r_s ≈ 3 km for solar mass)

At r = 10.0 r_s:
  g_tt = -0.9000  (time dilation factor)
  g_rr = 1.1111  (radial distance stretching)

At r = 5.0 r_s:
  g_tt = -0.8000  (time dilation factor)
  g_rr = 1.2500  (radial distance stretching)

At r = 2.0 r_s:
  g_tt = -0.5000  (time dilation factor)
  g_rr = 2.0000  (radial distance stretching)

At r = 1.5 r_s:
  g_tt = -0.3333  (time dilation factor)
  g_rr = 3.0000  (radial distance stretching)
```

---

### 📊 Visualization: Schwarzschild Metric Components

```python
# Plot metric components vs radius

r_vals = np.linspace(1.01 * r_s_sun, 10 * r_s_sun, 200)  # np.linspace() creates evenly spaced array between start and end
g_tt_vals = []
g_rr_vals = []

for r in r_vals:
    g_tt, g_rr, _, _ = schwarzschild_metric(r, M_sun)
    if g_tt is not None:
        g_tt_vals.append(g_tt)
        g_rr_vals.append(g_rr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: g_tt (time component)
ax = axes[0]
ax.plot(r_vals/r_s_sun, g_tt_vals, color=COLORS['blue'], linewidth=2)
ax.axhline(y=-1, color=COLORS['gray'], linestyle='--', label='Flat space')
ax.axvline(x=1, color=COLORS['red'], linestyle='--', alpha=0.7, label='Event horizon (r=r_s)')
ax.set_xlabel('r / r_s', fontsize=12)
ax.set_ylabel('g_tt', fontsize=12)
ax.set_title('Time Metric Component', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(1, 10)

# Right: g_rr (radial component)
ax = axes[1]
ax.plot(r_vals/r_s_sun, g_rr_vals, color=COLORS['orange'], linewidth=2)
ax.axhline(y=1, color=COLORS['gray'], linestyle='--', label='Flat space')
ax.axvline(x=1, color=COLORS['red'], linestyle='--', alpha=0.7, label='Event horizon')
ax.set_xlabel('r / r_s', fontsize=12)
ax.set_ylabel('g_rr', fontsize=12)
ax.set_title('Radial Metric Component', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(1, 10)
ax.set_ylim(0, 10)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Notice:")
print("- As r → r_s, g_tt → 0 (time 'stops' at horizon)")
print("- As r → r_s, g_rr → ∞ (infinite radial stretching)")
print("- As r → ∞, both → flat space values (-1 and 1)")
```

---

### 🔬 Explore: Proper Time and Coordinate Time

```python
# Time dilation for a stationary observer at radius r

def time_dilation_factor(r, M):
    """
    Returns dτ/dt for stationary observer at radius r.
    This is √(-g_tt).
    """
    r_s = 2 * M
    if r <= r_s:
        return 0
    return np.sqrt(1 - r_s/r)  # np.sqrt() computes square root

# Two observers: one at r1, one at r2
r1 = 2 * r_s_sun  # Close to horizon
r2 = 10 * r_s_sun  # Far from horizon

factor1 = time_dilation_factor(r1, M_sun)
factor2 = time_dilation_factor(r2, M_sun)

# If observer at infinity measures Δt = 1 year
dt_infinity = 1.0  # year

# How much proper time elapses for each observer?
dtau1 = factor1 * dt_infinity
dtau2 = factor2 * dt_infinity

print(f"Time Dilation Near Black Hole")
print(f"=" * 50)
print(f"\nObserver at r = {r1/r_s_sun:.1f} r_s:")
print(f"  Time dilation factor: {factor1:.4f}")
print(f"  Proper time when Δt = 1 yr: Δτ = {dtau1:.4f} yr")
print(f"\nObserver at r = {r2/r_s_sun:.1f} r_s:")
print(f"  Time dilation factor: {factor2:.4f}")
print(f"  Proper time when Δt = 1 yr: Δτ = {dtau2:.4f} yr")
print(f"\nObserver closer to black hole ages slower!")
```

---

### 🎯 Practice Question #1

**Q:** What is the Schwarzschild radius for Earth (M_Earth ≈ 6 × 10^24 kg)?

<details>
<summary>💡 Hint</summary>

Use r_s = 2GM/c² with G = 6.67 × 10^(-11) m³/(kg·s²) and c = 3 × 10^8 m/s.
</details>

<details>
<summary>✅ Answer</summary>

r_s = 2GM/c² = 2 × (6.67 × 10^(-11)) × (6 × 10^24) / (3 × 10^8)²

r_s ≈ **8.9 mm** (about 9 millimeters!)

If you could compress Earth to a sphere smaller than 9 mm in radius, it would become a black hole. Earth's actual radius is ~6371 km, so we're very far from that!

```python
G = 6.67e-11  # m^3/(kg·s^2)
c = 3e8       # m/s
M_earth = 6e24  # kg

r_s = 2 * G * M_earth / c**2
print(f"Earth's Schwarzschild radius: {r_s*1000:.1f} mm")
```
</details>

---

## 2. Black Holes and Event Horizons

### 📖 Concept

When **all** of a mass M is compressed within its Schwarzschild radius r_s, you get a **black hole**.

**Event Horizon (r = r_s):**
- **One-way boundary:** Things can fall in, but nothing can escape (not even light)
- **Not a physical surface:** Just a mathematical boundary in spacetime
- **Coordinate singularity:** Schwarzschild coordinates break down here, but other coordinate systems (Kruskal-Szekeres, Eddington-Finkelstein) work fine

**Inside the Event Horizon (r < r_s):**
- The r and t coordinates switch roles: r becomes timelike, t becomes spacelike
- Falling to r = 0 is as inevitable as moving forward in time
- The singularity at r = 0 is in your future, not at a place

**Singularity (r = 0):**
- Curvature becomes infinite
- Classical GR breaks down
- Need quantum gravity (not yet fully understood!)

**Types of Horizons:**
1. **Event horizon:** r = r_s (schwarzschild)
2. **Cauchy horizon:** In rotating/charged black holes (r < r_s)
3. **Cosmological horizon:** In expanding universes

---

### 💻 Code Example: Black Hole Properties

```python
def black_hole_properties(M, units='solar'):
    """
    Calculate key properties of a Schwarzschild black hole.

    Parameters:
    M: mass (in solar masses if units='solar', kg if units='SI')
    units: 'solar' or 'SI'
    """
    if units == 'solar':
        M_kg = M * 1.989e30  # Convert to kg
    else:
        M_kg = M

    G = 6.67430e-11  # m^3/(kg·s^2)
    c = 299792458    # m/s

    # Schwarzschild radius
    r_s = 2 * G * M_kg / c**2  # meters

    # Event horizon area
    A = 4 * np.pi * r_s**2  # m^2

    # Surface gravity (at horizon)
    kappa = c**4 / (4 * G * M_kg)  # m/s^2

    # Hawking temperature
    hbar = 1.054571817e-34  # J·s
    k_B = 1.380649e-23      # J/K
    T_H = hbar * c**3 / (8 * np.pi * k_B * G * M_kg)  # Kelvin

    # Evaporation time (Hawking radiation)
    t_evap = 2.1e67 * (M_kg / 1.989e30)**3  # seconds (approximate)

    return {
        'r_s_km': r_s / 1000,
        'r_s_m': r_s,
        'area_m2': A,
        'surface_gravity_ms2': kappa,
        'hawking_temp_K': T_H,
        'evap_time_years': t_evap / (365.25 * 86400)
    }

# Examples: Different mass black holes
masses = [1, 10, 1e6, 1e9]  # Solar masses
names = ['Stellar', '10 M_sun', 'Supermassive (Sgr A*)', 'Quasar']

print("Black Hole Properties")
print("=" * 80)

for M, name in zip(masses, names):
    props = black_hole_properties(M, units='solar')
    print(f"\n{name} Black Hole (M = {M:.0e} M_sun):")
    print(f"  Schwarzschild radius: {props['r_s_km']:.2e} km")
    print(f"  Event horizon area: {props['area_m2']:.2e} m²")
    print(f"  Surface gravity: {props['surface_gravity_ms2']:.2e} m/s²")
    print(f"  Hawking temperature: {props['hawking_temp_K']:.2e} K")
    print(f"  Evaporation time: {props['evap_time_years']:.2e} years")
```

---

### 📊 Visualization: Schwarzschild Black Hole

```python
# Embedding diagram (spatial slice at t=const)

# For Schwarzschild, spatial metric at t=const, θ=π/2:
# dl² = dr²/(1-r_s/r) + r² dφ²

# Embedding in 3D: rotate around z-axis
theta_embed = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
r_embed = np.linspace(1.01*r_s_sun, 5*r_s_sun, 50)  # np.linspace() creates evenly spaced array between start and end

R, Theta = np.meshgrid(r_embed, theta_embed)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# Compute embedding surface z(r)
# For Schwarzschild: z = ±∫ √(r_s/r - r_s²/r²) dr
def z_embedding(r, r_s):
    """Z coordinate for Schwarzschild embedding diagram."""
    # Approximate embedding function
    if r <= r_s:
        return 0
    return -2 * np.sqrt(r_s * (r - r_s))  # np.sqrt() computes square root

Z = np.array([[z_embedding(r, r_s_sun) for r in r_embed] for _ in theta_embed])  # np.array() converts Python list/tuple to efficient numpy array

# Convert to Cartesian coordinates
X = R * np.cos(Theta)  # np.cos() computes cosine (element-wise for arrays)
Y = R * np.sin(Theta)  # np.sin() computes sine (element-wise for arrays)

fig = plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Left: 3D embedding
ax1 = fig.add_subplot(121, projection='3d')  projection='3d'  # Create 3D axes
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)  # .plot_surface() draws 3D surface plot
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.set_zlabel('z', fontsize=10)
ax1.set_title('Embedding Diagram\n(Spatial Curvature)', fontsize=12, fontweight='bold')
ax1.view_init(elev=25, azim=45)

# Right: Cross-section
ax2 = fig.add_subplot(122)
r_cross = r_embed
z_cross = [z_embedding(r, r_s_sun) for r in r_cross]

ax2.plot(r_cross/r_s_sun, z_cross/r_s_sun, color=COLORS['blue'], linewidth=2)
ax2.plot(-r_cross/r_s_sun, z_cross/r_s_sun, color=COLORS['blue'], linewidth=2)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# Event horizon
circle = plt.Circle((0, 0), 1, color=COLORS['red'], alpha=0.3, label='Event horizon')
ax2.add_patch(circle)

ax2.set_xlabel('r / r_s', fontsize=12)
ax2.set_ylabel('z / r_s', fontsize=12)
ax2.set_title('Cross-Section\n("Throat" at Event Horizon)', fontsize=12, fontweight='bold')
ax2.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(-5, 5)
ax2.set_ylim(-6, 1)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("The 'funnel' shape shows how space curves near a black hole.")
print("The throat narrows to the event horizon at r = r_s.")
```

---

### 🔬 Explore: Photon Sphere

The **photon sphere** at r = 3M (or 1.5 r_s) is where photons can orbit the black hole!

```python
# Orbital velocities at different radii

def orbital_velocity(r, M):
    """
    Circular orbital velocity at radius r around mass M.
    In Schwarzschild spacetime.
    """
    r_s = 2 * M
    if r <= r_s:
        return None
    # v/c for circular orbit
    v_over_c = np.sqrt(M / r)  # np.sqrt() computes square root
    return v_over_c

# Special radii
r_photon = 3 * M_sun  # Photon sphere
r_isco = 6 * M_sun    # Innermost stable circular orbit

r_values = np.linspace(1.5*r_s_sun, 20*r_s_sun, 200)  # np.linspace() creates evenly spaced array between start and end
v_values = [orbital_velocity(r, M_sun) for r in r_values]

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(r_values/r_s_sun, v_values, color=COLORS['blue'], linewidth=2)  # plt.plot() draws line plot
plt.axvline(x=r_photon/r_s_sun, color=COLORS['orange'], linestyle='--',  # plt.axvline() draws vertical line across plot
            linewidth=2, label=f'Photon sphere (r=1.5r_s)')
plt.axvline(x=r_isco/r_s_sun, color=COLORS['green'], linestyle='--',  # plt.axvline() draws vertical line across plot
            linewidth=2, label=f'ISCO (r=3r_s)')
plt.axhline(y=1, color=COLORS['red'], linestyle=':', alpha=0.7, label='Speed of light')  # plt.axhline() draws horizontal line across plot
plt.xlabel('r / r_s', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Orbital velocity v/c', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Circular Orbital Velocity Around Black Hole', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.xlim(1.5, 20)  # plt.xlim() sets x-axis limits
plt.ylim(0, 1.1)  # plt.ylim() sets y-axis limits
plt.show()  # plt.show() displays the figure window

v_photon = orbital_velocity(r_photon, M_sun)
v_isco = orbital_velocity(r_isco, M_sun)

print(f"Photon sphere (r = 1.5 r_s):")
print(f"  Orbital velocity: v = {v_photon:.3f} c (light speed!)")
print(f"\nISCO (r = 3 r_s):")
print(f"  Orbital velocity: v = {v_isco:.3f} c")
print(f"\nOrbits below ISCO are unstable - they spiral in!")
```

---

### 📊 Visualization: Penrose Diagram

**Penrose diagrams** (also called conformal diagrams) show the **causal structure** of spacetime by compressing infinity to finite boundaries. They make it easy to visualize light cones, horizons, and singularities.

```python
# Create Penrose diagram for Schwarzschild black hole

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # plt.subplots() creates figure with multiple subplots

# Left panel: Standard spacetime diagram
ax1.set_xlim(-3, 6)
ax1.set_ylim(-3, 6)
ax1.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Event horizon (45° line for ingoing light)
r_horizon_t = np.linspace(-3, 6, 100)  # np.linspace() creates evenly spaced array between start and end
r_horizon_x = np.linspace(-3, 6, 100) - 3  # np.linspace() creates evenly spaced array between start and end
ax1.fill_betweenx(r_horizon_t, -3, r_horizon_x, alpha=0.15, color=COLORS['red'],
                  label='Inside horizon (r < r_s)')

# Singularity (r = 0)
ax1.plot([2, 2], [2, 6], color=COLORS['red'], linewidth=3, label='Singularity (r=0)')
ax1.fill_between([2, 6], 2, 6, alpha=0.3, color=COLORS['red'])

# Event horizon line
ax1.plot([-3, 2], [-3, 2], color=COLORS['orange'], linewidth=2.5,
         linestyle='--', label='Event horizon (r=r_s)')

# Sample worldlines
# Infalling observer
t_infall = np.linspace(-2, 2, 50)  # np.linspace() creates evenly spaced array between start and end
r_infall = -2 + 0.8*t_infall + 0.15*t_infall**2
ax1.plot(r_infall, t_infall, color=COLORS['blue'], linewidth=2, label='Infalling observer')
ax1.arrow(r_infall[-10], t_infall[-10], r_infall[-5]-r_infall[-10], t_infall[-5]-t_infall[-10],
         head_width=0.2, head_length=0.15, fc=COLORS['blue'], ec=COLORS['blue'])

# Light ray (45° slope)
ax1.plot([-2, 3], [-2, 3], color=COLORS['yellow'], linewidth=1.5, linestyle=':', label='Outgoing light')
ax1.plot([1, -1.5], [1, 3.5], color=COLORS['cyan'], linewidth=1.5, linestyle=':', label='Ingoing light')

ax1.set_xlabel('r (space)', fontsize=12)
ax1.set_ylabel('t (time)', fontsize=12)
ax1.set_title('Standard Schwarzschild Coordinates\n(Horizon at 45° line)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Right panel: Penrose diagram
ax2.set_xlim(-0.2, np.pi + 0.2)  # np.pi for π constant
ax2.set_ylim(-0.2, np.pi + 0.2)  # np.pi for π constant
ax2.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes

# Draw Penrose diamond shape
# Region I (exterior): r > r_s
U_ext = np.linspace(0, np.pi/2, 100)  # np.linspace() creates evenly spaced array between start and end
V_ext = np.linspace(0, np.pi/2, 100)  # np.linspace() creates evenly spaced array between start and end

# Boundaries
# r = infinity (past and future null infinity)
ax2.plot([0, np.pi/2], [np.pi/2, 0], 'k-', linewidth=2, label='Past null infinity (I⁻)')  # np.pi for π constant
ax2.plot([np.pi/2, np.pi], [np.pi, np.pi/2], 'k-', linewidth=2, label='Future null infinity (I⁺)')  # np.pi for π constant

# Event horizon (r = r_s)
ax2.plot([0, np.pi/2], [0, np.pi/2], color=COLORS['orange'], linewidth=3,  # np.pi for π constant
         linestyle='--', label='Event horizon')

# Singularity (r = 0)
ax2.plot([np.pi/2, np.pi], [np.pi/2, np.pi/2], color=COLORS['red'], linewidth=4,  # np.pi for π constant
         label='Singularity (r=0)')

# Spatial infinity
ax2.plot([np.pi/2, np.pi/2], [0, np.pi], 'k-', linewidth=2, label='Spatial infinity (i⁰)')  # np.pi for π constant

# Shade regions
# Region I: outside horizon
U_I, V_I = np.meshgrid(np.linspace(0, np.pi/2, 50), np.linspace(0, np.pi/2, 50))  # np.meshgrid() creates coordinate matrices from coordinate vectors
mask_I = (U_I + V_I) <= np.pi/2  # np.pi for π constant
ax2.contourf(U_I, V_I, mask_I.astype(float), levels=[0.5, 1.5], colors=[COLORS['blue']], alpha=0.1)

# Region II: inside horizon
U_II, V_II = np.meshgrid(np.linspace(0, np.pi/2, 50), np.linspace(np.pi/2, np.pi, 50))  # np.meshgrid() creates coordinate matrices from coordinate vectors
ax2.fill([0, np.pi/2, np.pi/2, 0], [np.pi/2, np.pi/2, np.pi, np.pi/2],  # np.pi for π constant
         color=COLORS['red'], alpha=0.15, label='Black hole interior')

# Sample worldlines (light cones at 45°)
# Ingoing observer
u_vals = np.linspace(0.3, np.pi/2-0.1, 30)  # np.linspace() creates evenly spaced array between start and end
v_vals = u_vals + 0.5
ax2.plot(u_vals, v_vals, color=COLORS['blue'], linewidth=2)
ax2.arrow(u_vals[-5], v_vals[-5], u_vals[-1]-u_vals[-5], v_vals[-1]-v_vals[-5],
         head_width=0.08, head_length=0.06, fc=COLORS['blue'], ec=COLORS['blue'])

# Light rays (45° lines)
ax2.plot([0.2, 0.7], [0.2, 0.7], color=COLORS['yellow'], linewidth=1.5, linestyle=':')
ax2.plot([0.5, 1.0], [0.3, 0.8], color=COLORS['cyan'], linewidth=1.5, linestyle=':')

ax2.set_xlabel('U (null coordinate)', fontsize=12)
ax2.set_ylabel('V (null coordinate)', fontsize=12)
ax2.set_title('Penrose Diagram for Schwarzschild\n(Causal Structure)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)

# Add labels for regions
ax2.text(0.6, 0.4, 'Region I\n(Exterior)', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.text(0.6, 2.0, 'Region II\n(Interior)', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("\nKey features of Penrose diagrams:")
print("1. Light rays always travel at 45° angles")
print("2. Entire infinity compressed to finite boundaries")
print("3. Event horizon: boundary where escape becomes impossible")
print("4. Singularity: spacelike surface where curvature diverges")
print("5. Observers cross horizon in finite proper time (but infinite coordinate time)")
```

**What you should see:**
- **Left:** Standard spacetime diagram showing event horizon and infalling paths
- **Right:** Penrose diagram compressing all of spacetime to a finite diamond, making causal structure clear

The Penrose diagram makes it obvious that:
- Nothing can escape from inside the horizon (all future-directed paths hit the singularity)
- The singularity is in the *future* of infalling observers, not at a location
- Light cones tilt inward as you approach the horizon

---

### 🎯 Practice Question #2

**Q:** An astronaut hovers at r = 2r_s above a black hole for 1 hour (by their clock). How much time passes for a distant observer?

<details>
<summary>💡 Hint 1</summary>

Use the time dilation formula: dτ/dt = √(1 - r_s/r)
</details>

<details>
<summary>💡 Hint 2</summary>

Solve for dt given dτ = 1 hour.
</details>

<details>
<summary>✅ Answer</summary>

Time dilation factor: √(1 - r_s/(2r_s)) = √(1 - 1/2) = √(1/2) = 1/√2

So: dτ/dt = 1/√2, which means dt/dτ = √2

If dτ = 1 hour, then: dt = √2 hours ≈ **1.41 hours**

The distant observer measures more time passing (astronaut's clock runs slow).

```python
r = 2 * r_s_sun
factor = time_dilation_factor(r, M_sun)
dtau = 1.0  # hour (astronaut's proper time)
dt = dtau / factor

print(f"Astronaut's proper time: {dtau:.2f} hours")
print(f"Distant observer's time: {dt:.2f} hours")
```
</details>

---

## 3. The Kerr Solution: Rotating Black Holes

### 📖 Concept

The **Kerr solution** (1963) describes a **rotating black hole**. It's the most general stationary black hole solution with mass M and angular momentum J.

**Kerr Metric** (in Boyer-Lindquist coordinates):

```
ds² = -(1 - 2Mr/Σ)dt² - (4Mar sin²θ/Σ)dt dφ
      + (Σ/Δ)dr² + Σ dθ²
      + ((r²+a²)² - a²Δsin²θ)/Σ sin²θ dφ²
```

Where:
- **a = J/M**: Angular momentum per unit mass (spin parameter)
- **Σ = r² + a²cos²θ**
- **Δ = r² - 2Mr + a²**
- For **a = 0**: Reduces to Schwarzschild solution

**Key Features:**
1. **Two horizons:**
   - Outer horizon: r_+ = M + √(M² - a²)
   - Inner horizon: r_- = M - √(M² - a²)
   - Maximum spin: a ≤ M (extremal Kerr when a = M)

2. **Ergosphere:** Region where spacetime is dragged around (frame-dragging)
   - Boundary: r_ergo = M + √(M² - a²cos²θ)
   - Between ergosphere and outer horizon: *must* co-rotate with black hole

3. **Ring singularity:** At r = 0, θ = π/2 (not a point!)

---

### 💻 Code Example: Kerr Black Hole Horizons

```python
def kerr_horizons(M, a):
    """
    Calculate horizons for Kerr black hole.

    Parameters:
    M: mass
    a: angular momentum parameter (a = J/M, must satisfy |a| ≤ M)

    Returns:
    r_plus: outer horizon
    r_minus: inner horizon
    r_ergo_eq: ergosphere radius at equator
    """
    if abs(a) > M:
        print("Warning: a > M gives naked singularity (unphysical!)")
        return None, None, None

    # Event horizons
    r_plus = M + np.sqrt(M**2 - a**2)  # np.sqrt() computes square root
    r_minus = M - np.sqrt(M**2 - a**2)  # np.sqrt() computes square root

    # Ergosphere at equator (θ = π/2)
    r_ergo_eq = M + np.sqrt(M**2 - a**2 * 0)  # cos²(π/2) = 0
    r_ergo_eq = 2 * M  # Simplifies to this

    return r_plus, r_minus, r_ergo_eq

# Compare different spins
M = 1.0
spins = [0, 0.5, 0.9, 0.99, 1.0]  # a/M ratios

print("Kerr Black Hole Horizons (M = 1)")
print("=" * 60)

for spin in spins:
    a = spin * M
    r_plus, r_minus, r_ergo = kerr_horizons(M, a)

    if r_plus is not None:
        print(f"\na/M = {spin:.2f}:")
        print(f"  Outer horizon: r_+ = {r_plus:.4f} M")
        print(f"  Inner horizon: r_- = {r_minus:.4f} M")
        print(f"  Ergosphere (equator): r_ergo = {r_ergo:.4f} M")
        print(f"  Horizon area: A = {4 * np.pi * (r_plus**2 + a**2):.4f} (units of M²)")

# Note: For Schwarzschild (a=0), r_+ = 2M
```

---

### 📊 Visualization: Kerr Black Hole Structure

```python
# Visualize horizons and ergosphere for different spins

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Horizon radii vs spin
spins = np.linspace(0, 1, 100)  # np.linspace() creates evenly spaced array between start and end
r_plus_vals = []
r_minus_vals = []

for s in spins:
    a = s * M
    r_plus, r_minus, _ = kerr_horizons(M, a)
    r_plus_vals.append(r_plus)
    r_minus_vals.append(r_minus)

ax = axes[0]
ax.plot(spins, r_plus_vals, color=COLORS['blue'], linewidth=2, label='Outer horizon r_+')
ax.plot(spins, r_minus_vals, color=COLORS['orange'], linewidth=2, label='Inner horizon r_-')
ax.axhline(y=2*M, color=COLORS['gray'], linestyle='--', alpha=0.7, label='Schwarzschild r_s')
ax.fill_between(spins, r_minus_vals, r_plus_vals, alpha=0.2, color=COLORS['red'])
ax.set_xlabel('Spin parameter a/M', fontsize=12)
ax.set_ylabel('Horizon radius / M', fontsize=12)
ax.set_title('Kerr Horizons vs Spin', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 2.5)

# Right: Cross-section showing ergosphere
ax = axes[1]

# For a specific spin (a = 0.9 M)
a_demo = 0.9 * M
r_plus_demo, r_minus_demo, _ = kerr_horizons(M, a_demo)

theta = np.linspace(0, np.pi, 100)  # np.linspace() creates evenly spaced array between start and end

# Ergosphere radius as function of θ
r_ergo_theta = M + np.sqrt(M**2 - a_demo**2 * np.cos(theta)**2)  # np.cos() computes cosine (element-wise for arrays)

# Convert to Cartesian for plotting
x_outer = r_plus_demo * np.sin(theta)  # np.sin() computes sine (element-wise for arrays)
z_outer = r_plus_demo * np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)

x_inner = r_minus_demo * np.sin(theta)  # np.sin() computes sine (element-wise for arrays)
z_inner = r_minus_demo * np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)

x_ergo = r_ergo_theta * np.sin(theta)  # np.sin() computes sine (element-wise for arrays)
z_ergo = r_ergo_theta * np.cos(theta)  # np.cos() computes cosine (element-wise for arrays)

ax.plot(x_outer, z_outer, color=COLORS['blue'], linewidth=2, label='Outer horizon')
ax.plot(-x_outer, z_outer, color=COLORS['blue'], linewidth=2)
ax.plot(x_inner, z_inner, color=COLORS['orange'], linewidth=2, label='Inner horizon')
ax.plot(-x_inner, z_inner, color=COLORS['orange'], linewidth=2)
ax.plot(x_ergo, z_ergo, color=COLORS['red'], linewidth=2, linestyle='--', label='Ergosphere')
ax.plot(-x_ergo, z_ergo, color=COLORS['red'], linewidth=2, linestyle='--')

# Fill ergosphere region
ax.fill_between(x_ergo, z_ergo, -z_ergo, alpha=0.1, color=COLORS['red'])
ax.fill_between(-x_ergo, z_ergo, -z_ergo, alpha=0.1, color=COLORS['red'])

ax.set_xlabel('x / M', fontsize=12)
ax.set_ylabel('z / M (rotation axis)', fontsize=12)
ax.set_title(f'Kerr Black Hole Structure (a={a_demo:.1f}M)', fontsize=13, fontweight='bold')
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("The ergosphere (red) is where frame-dragging is extreme.")
print("Inside ergosphere but outside horizon: can still escape, but must co-rotate!")
```

---

### 🔬 Explore: Frame Dragging (Lense-Thirring Effect)

```python
# Angular velocity of frame dragging

def frame_dragging_omega(r, theta, M, a):
    """
    Angular velocity of local inertial frames around Kerr black hole.
    ω = dφ/dt for zero angular momentum observers (ZAMOs).
    """
    Sigma = r**2 + a**2 * np.cos(theta)**2  # np.cos() computes cosine (element-wise for arrays)
    Delta = r**2 - 2*M*r + a**2

    # Frame-dragging angular velocity
    omega = (2 * M * a * r) / (Sigma * ((r**2 + a**2)**2 - a**2 * Delta * np.sin(theta)**2))  # np.sin() computes sine (element-wise for arrays)

    return omega

# Example: At equator of rapidly spinning black hole
a_fast = 0.9 * M
theta_eq = np.pi / 2  # Equator

r_vals = np.linspace(1.5, 10, 100)  # np.linspace() creates evenly spaced array between start and end
omega_vals = [frame_dragging_omega(r, theta_eq, M, a_fast) for r in r_vals]

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(r_vals, omega_vals, color=COLORS['purple'], linewidth=2)  # plt.plot() draws line plot
plt.xlabel('Radius r / M', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Frame-dragging angular velocity ω', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Frame Dragging Around Kerr Black Hole (a=0.9M)', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.show()  # plt.show() displays the figure window

r_test = 3.0
omega_test = frame_dragging_omega(r_test, theta_eq, M, a_fast)
print(f"At r = {r_test} M:")
print(f"  Frame-dragging ω = {omega_test:.4f} / M")
print(f"  Local frames are dragged around the black hole!")
```

---

## 4. FLRW Cosmology: The Expanding Universe

### 📖 Concept

The **Friedmann-Lemaître-Robertson-Walker (FLRW) metric** describes a homogeneous and isotropic universe (same everywhere, same in all directions). This is our best model for the **large-scale structure of the cosmos**.

**FLRW Metric:**

```
ds² = -dt² + a(t)² [dr²/(1-kr²) + r²(dθ² + sin²θ dφ²)]
```

Where:
- **a(t)**: Scale factor (how the universe expands)
- **k**: Spatial curvature
  - k = +1: Closed (positive curvature, sphere)
  - k = 0: Flat (Euclidean)
  - k = -1: Open (negative curvature, hyperbolic)

**Friedmann Equations** (from Einstein equations):

```
H² = (ȧ/a)² = (8πG/3)ρ - k/a² + Λ/3

ä/a = -(4πG/3)(ρ + 3p) + Λ/3
```

Where:
- **H = ȧ/a**: Hubble parameter
- **ρ**: Energy density
- **p**: Pressure
- **Λ**: Cosmological constant

**Current Universe (ΛCDM model):**
- Flat: k = 0
- Dark energy: Λ ≈ 70%
- Dark matter: ≈ 25%
- Ordinary matter: ≈ 5%

---

### 💻 Code Example: Friedmann Equations

```python
# Solve Friedmann equation numerically

from scipy.integrate import odeint  # ODE solver for initial value problems

def friedmann_ode(y, t, Omega_m, Omega_Lambda, Omega_k):
    """
    Friedmann equation as ODE system.
    y = [a, ȧ]
    Returns [ȧ, ä]
    """
    a, a_dot = y

    if a <= 0:
        return [0, 0]

    H0 = 1.0  # Hubble constant (normalized)

    # Current Hubble parameter
    H_squared = H0**2 * (Omega_m / a**3 + Omega_k / a**2 + Omega_Lambda)

    # Acceleration
    a_ddot = -0.5 * H0**2 * (Omega_m / a**2 - 2 * Omega_Lambda) * a

    return [a_dot, a_ddot]

# Initial conditions
a0 = 1.0  # Scale factor today (normalized)
H0 = 1.0  # Hubble constant today (normalized)

# Different cosmologies
cosmologies = [
    ("Matter-dominated", 1.0, 0.0, 0.0),
    ("ΛCDM (current)", 0.3, 0.7, 0.0),
    ("Empty (coasting)", 0.0, 0.0, 0.0),
]

time = np.linspace(0, 2, 500)  # Normalized time

plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Left: Scale factor evolution
plt.subplot(1, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)

for name, Om, OL, Ok in cosmologies:
    # Solve ODE
    y0 = [a0, H0 * a0]  # Initial: a=1, ȧ=H0
    solution = odeint(friedmann_ode, y0, time, args=(Om, OL, Ok))  # odeint() solves ODE system with given initial conditions
    a_vals = solution[:, 0]

    plt.plot(time, a_vals, linewidth=2, label=name)  # plt.plot() draws line plot

plt.xlabel('Time (normalized)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Scale factor a(t)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Universe Expansion (Different Models)', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.axhline(y=1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)  # plt.axhline() draws horizontal line across plot

# Right: Hubble parameter
plt.subplot(1, 2, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)

for name, Om, OL, Ok in cosmologies:
    y0 = [a0, H0 * a0]
    solution = odeint(friedmann_ode, y0, time, args=(Om, OL, Ok))  # odeint() solves ODE system with given initial conditions
    a_vals = solution[:, 0]
    a_dot_vals = solution[:, 1]

    # H = ȧ/a
    H_vals = a_dot_vals / a_vals

    plt.plot(time, H_vals, linewidth=2, label=name)  # plt.plot() draws line plot

plt.xlabel('Time (normalized)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Hubble parameter H(t)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Expansion Rate Over Time', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.axhline(y=1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)  # plt.axhline() draws horizontal line across plot

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Notice:")
print("- Matter-dominated: decelerating expansion")
print("- ΛCDM: accelerating expansion (due to dark energy)")
print("- Empty: constant expansion (Milne universe)")
```

---

### 📊 Visualization: Cosmic History

```python
# Plot the history of our universe (ΛCDM model)

# Time from Big Bang to now: ~13.8 billion years
# Scale factor: a(now) = 1

t_now = 13.8  # Billion years
t_history = np.array([  # np.array() converts Python list/tuple to efficient numpy array
    0.00038,  # Recombination (CMB)
    0.5,      # First stars
    1.0,      # First galaxies
    9.0,      # Solar system forms
    13.8      # Today
])

a_history = np.array([  # np.array() converts Python list/tuple to efficient numpy array
    1/1100,  # z ~ 1100 at recombination
    1/20,    # z ~ 20
    1/10,    # z ~ 10
    1/1.5,   # z ~ 0.5
    1.0      # z = 0 today
])

events = [
    'CMB (Recombination)',
    'First Stars',
    'First Galaxies',
    'Solar System',
    'Today'
]

fig, ax = plt.subplots(figsize=(14, 6))

# Plot timeline
ax.plot(t_history, a_history, 'o-', color=COLORS['blue'], linewidth=3, markersize=10)

# Add event labels
for t, a, event in zip(t_history, a_history, events):
    ax.annotate(event, xy=(t, a), xytext=(t, a + 0.15),
                ha='center', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_xlabel('Time since Big Bang (billion years)', fontsize=13)
ax.set_ylabel('Scale factor a(t)', fontsize=13)
ax.set_title('Cosmic History: Scale Factor Evolution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 15)
ax.set_ylim(0, 1.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Cosmic Timeline:")
print(f"  Age of universe: {t_now} billion years")
print(f"  CMB emitted at z ~ 1100 (a = 1/1100)")
print(f"  Universe has expanded by factor of 1100 since then!")
```

---

### 🎯 Practice Question #3

**Q:** If a galaxy is at redshift z = 2, what was the scale factor a when the light we see now was emitted?

<details>
<summary>💡 Hint</summary>

Redshift relates to scale factor by: 1 + z = a_now / a_then = 1/a (if a_now = 1)
</details>

<details>
<summary>✅ Answer</summary>

1 + z = 1/a

So: a = 1/(1+z) = 1/(1+2) = **1/3**

The universe was 1/3 its current size when that light was emitted.

```python
z = 2
a = 1 / (1 + z)
print(f"Redshift z = {z}")
print(f"Scale factor a = {a:.3f}")
print(f"Universe was {1/a:.1f}× smaller")
```
</details>

---

## 5. Gravitational Wave Solutions

### 📖 Concept

Gravitational waves are **ripples in spacetime** - solutions to the linearized Einstein equations. They were predicted by Einstein (1916) and detected by LIGO (2015)!

**Linearized Metric:**

```
g_μν = η_μν + h_μν    where |h_μν| << 1
```

**Wave Equation** (in transverse-traceless gauge):

```
□ h_μν = 0    where □ = -∂²/∂t² + ∇²
```

**Plane Wave Solution:**

```
h_μν = A_μν exp(i(k·x - ωt))
```

**Polarizations:**
- **Plus (+):** Stretches/squeezes along x and y axes
- **Cross (×):** Stretches/squeezes along diagonals

**Properties:**
- Travel at speed of light: v = c
- Transverse: perpendicular to propagation
- Two independent polarizations
- Quadrupole radiation (not dipole like EM waves)

---

### 💻 Code Example: GW from Binary System

```python
# Simulate gravitational wave from inspiraling binary

def gw_from_binary(t, M1, M2, r, f):
    """
    Gravitational wave strain from binary system.

    Parameters:
    t: time array
    M1, M2: masses of binary components
    r: distance to observer
    f: orbital frequency

    Returns:
    h_plus, h_cross: strains in two polarizations
    """
    # Total mass and reduced mass
    M = M1 + M2
    mu = M1 * M2 / M

    # Chirp mass
    M_chirp = (M1 * M2)**(3/5) / (M1 + M2)**(1/5)

    # Gravitational wave frequency (twice orbital frequency)
    f_gw = 2 * f

    # Amplitude (rough estimate in geometric units)
    h0 = (4 / r) * (np.pi * M_chirp * f_gw)**(2/3)

    # Plus polarization (observed face-on)
    h_plus = h0 * np.cos(2 * np.pi * f_gw * t)  # np.cos() computes cosine (element-wise for arrays)

    # Cross polarization
    h_cross = h0 * np.sin(2 * np.pi * f_gw * t)  # np.sin() computes sine (element-wise for arrays)

    return h_plus, h_cross

# Example: Binary neutron star merger
M1 = 1.4  # Solar masses
M2 = 1.4
distance = 1e6  # Mpc (very rough units)
f_orbit = 100  # Hz (late inspiral)

t_signal = np.linspace(0, 0.1, 1000)  # np.linspace() creates evenly spaced array between start and end
h_plus, h_cross = gw_from_binary(t_signal, M1, M2, distance, f_orbit)

plt.figure(figsize=(12, 5))  # plt.figure() creates a new figure for plotting

# Left: Plus polarization
plt.subplot(1, 2, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(t_signal * 1000, h_plus, color=COLORS['blue'], linewidth=1.5)  # plt.plot() draws line plot
plt.xlabel('Time (ms)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Strain h_+', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Gravitational Wave: + Polarization', fontsize=12, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

# Right: Cross polarization
plt.subplot(1, 2, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(t_signal * 1000, h_cross, color=COLORS['orange'], linewidth=1.5)  # plt.plot() draws line plot
plt.xlabel('Time (ms)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Strain h_×', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Gravitational Wave: × Polarization', fontsize=12, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print(f"Binary system: {M1} M_sun + {M2} M_sun")
print(f"Orbital frequency: {f_orbit} Hz")
print(f"GW frequency: {2*f_orbit} Hz (twice orbital frequency)")
print(f"Approximate strain amplitude: {np.max(np.abs(h_plus)):.2e}")
```

---

### 🔬 Explore: LIGO Detection

```python
# Recreate a simplified LIGO-like detection

# GW150914: First detection (Sept 14, 2015)
# Two black holes: ~36 M_sun and ~29 M_sun
# Distance: ~410 Mpc

t_chirp = np.linspace(0, 0.2, 2000)  # np.linspace() creates evenly spaced array between start and end

# Chirp signal: frequency increases as binary inspirals
def chirp_frequency(t, f0, tau):
    """Frequency sweep of inspiral."""
    return f0 / (1 - t/tau)**(3/8)

f0 = 35  # Hz (initial)
tau = 0.18  # Chirp time

# Instantaneous frequency and phase
freq = chirp_frequency(t_chirp, f0, tau)
phase = np.cumsum(freq) * (t_chirp[1] - t_chirp[0]) * 2 * np.pi

# Amplitude grows as frequency increases (closer orbit = stronger radiation)
amplitude = (freq / f0)**(2/3)

# Strain signal
h_signal = amplitude * np.sin(phase)  # np.sin() computes sine (element-wise for arrays)

# Add noise (white noise, simplified)
noise = 0.5 * np.random.randn(len(t_chirp))
h_observed = h_signal + noise

plt.figure(figsize=(14, 10))  # plt.figure() creates a new figure for plotting

# Top: Clean signal
plt.subplot(3, 1, 1)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(t_chirp * 1000, h_signal, color=COLORS['blue'], linewidth=1)  # plt.plot() draws line plot
plt.ylabel('Strain h (clean)', fontsize=11)  # plt.ylabel() sets y-axis label
plt.title('GW150914-like Chirp Signal', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

# Middle: Noisy observation
plt.subplot(3, 1, 2)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(t_chirp * 1000, h_observed, color=COLORS['gray'], linewidth=0.5, alpha=0.7)  # plt.plot() draws line plot
plt.plot(t_chirp * 1000, h_signal, color=COLORS['red'], linewidth=1.5, label='True signal')  # plt.plot() draws line plot
plt.ylabel('Strain h (observed)', fontsize=11)  # plt.ylabel() sets y-axis label
plt.title('With Detector Noise', fontsize=12, fontweight='bold')  # plt.title() sets plot title
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

# Bottom: Frequency vs time
plt.subplot(3, 1, 3)  # plt.subplot() creates subplot in grid layout (rows, cols, position)
plt.plot(t_chirp * 1000, freq, color=COLORS['green'], linewidth=2)  # plt.plot() draws line plot
plt.xlabel('Time (ms)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Frequency (Hz)', fontsize=11)  # plt.ylabel() sets y-axis label
plt.title('Chirp Frequency (increases as binary inspirals)', fontsize=12, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("The 'chirp' signature:")
print("- Frequency increases as orbit shrinks")
print("- Amplitude grows as binary gets closer")
print("- Ends at merger (ringdown not shown)")
print("\nThis is how LIGO identifies binary mergers!")
```

---

## 6. de Sitter and Anti-de Sitter Spaces

### 📖 Concept

**de Sitter (dS) space:** Maximally symmetric solution with **positive cosmological constant** Λ > 0.

Metric (static coordinates):

```
ds² = -(1 - Λr²/3)dt² + (1 - Λr²/3)^(-1) dr² + r²dΩ²
```

- Exponentially expanding universe
- Cosmological horizon at r = √(3/Λ)
- No matter/energy (pure vacuum with Λ)

**Anti-de Sitter (AdS) space:** Maximally symmetric solution with **negative cosmological constant** Λ < 0.

- Constant negative curvature
- Important in AdS/CFT correspondence (string theory)
- Bounded spatial slices

---

### 💻 Code Example: de Sitter Expansion

```python
# de Sitter universe evolution

def de_sitter_scale_factor(t, Lambda):
    """
    Scale factor for de Sitter universe.
    a(t) = exp(√(Λ/3) t)
    """
    H = np.sqrt(Lambda / 3)  # np.sqrt() computes square root
    return np.exp(H * t)  # np.exp() computes exponential e^x

# Example
Lambda = 1.0  # Cosmological constant
t_vals = np.linspace(0, 5, 100)  # np.linspace() creates evenly spaced array between start and end

a_vals = [de_sitter_scale_factor(t, Lambda) for t in t_vals]

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(t_vals, a_vals, color=COLORS['blue'], linewidth=2)  # plt.plot() draws line plot
plt.xlabel('Time t', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Scale factor a(t)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('de Sitter Universe: Exponential Expansion', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.yscale('log')
plt.show()  # plt.show() displays the figure window

# Growth rate
H_dS = np.sqrt(Lambda / 3)  # np.sqrt() computes square root
print(f"de Sitter Hubble constant: H = √(Λ/3) = {H_dS:.4f}")
print(f"Scale factor doubles every: {np.log(2)/H_dS:.2f} time units")
print("\nPure exponential expansion - models dark energy dominated universe!")
```

---

## 7. Other Important Solutions

### 📖 Concept

**Reissner-Nordström:** Charged, non-rotating black hole
- Mass M, charge Q
- Two horizons (like Kerr)
- Extremal case: M = Q (naked singularity if Q > M)

**Kerr-Newman:** Rotating AND charged
- Most general stationary black hole
- Parameters: M, J, Q

**Schwarzschild-de Sitter:** Black hole in expanding universe
- Schwarzschild + cosmological constant
- Two horizons: black hole and cosmological

**BTZ Black Hole:** 3D black hole in AdS space
- Important for holography and quantum gravity

**pp-waves:** Plane-fronted gravitational waves
- Exact solutions (not linearized)
- Multiple physical applications

---

### 💻 Code Example: Solution Summary

```python
# Summary of major GR solutions

solutions_summary = {
    'Schwarzschild': {
        'Year': 1916,
        'Properties': 'Spherical, static, vacuum',
        'Parameters': 'M (mass)',
        'Applications': 'Non-rotating stars, stellar black holes'
    },
    'Kerr': {
        'Year': 1963,
        'Properties': 'Axisymmetric, stationary, vacuum',
        'Parameters': 'M (mass), a (spin)',
        'Applications': 'Rotating black holes, astrophysical BHs'
    },
    'FLRW': {
        'Year': '1920s',
        'Properties': 'Homogeneous, isotropic, expanding',
        'Parameters': 'a(t) (scale factor), k (curvature)',
        'Applications': 'Cosmology, large-scale universe'
    },
    'de Sitter': {
        'Year': 1917,
        'Properties': 'Maximal symmetry, Λ > 0',
        'Parameters': 'Λ (cosmo constant)',
        'Applications': 'Inflating universe, dark energy'
    },
    'Reissner-Nordström': {
        'Year': 1916-18,
        'Properties': 'Spherical, static, charged',
        'Parameters': 'M (mass), Q (charge)',
        'Applications': 'Charged black holes (theoretical)'
    },
    'Gravitational Waves': {
        'Year': 1916,
        'Properties': 'Weak field, wavelike',
        'Parameters': 'h_μν (strain), f (frequency)',
        'Applications': 'LIGO detections, binary mergers'
    }
}

print("=" * 80)
print("MAJOR SOLUTIONS TO EINSTEIN'S EQUATIONS")
print("=" * 80)

for name, info in solutions_summary.items():
    print(f"\n{name} ({info['Year']})")
    print(f"  Properties: {info['Properties']}")
    print(f"  Parameters: {info['Parameters']}")
    print(f"  Applications: {info['Applications']}")

print("\n" + "=" * 80)
print("All of these emerge from the same fundamental equation:")
print("  G_μν = (8πG/c⁴) T_μν")
print("=" * 80)
```

---

## Practice Questions

Test your understanding of GR solutions:

### Schwarzschild Solution

1. Calculate the Schwarzschild radius for a black hole with mass M = 10 M_sun.

2. At what radius is the gravitational time dilation factor exactly 1/2?

3. What is special about the photon sphere at r = 1.5 r_s?

### Black Holes

4. What is the ISCO (innermost stable circular orbit) for a Schwarzschild black hole?

5. Can an observer inside the event horizon ever escape? Why or why not?

6. What is the difference between a coordinate singularity and a physical singularity?

### Kerr Black Holes

7. What is the maximum possible spin parameter a for a Kerr black hole?

8. Describe the ergosphere. Where is it located?

9. How many horizons does a Kerr black hole have (for a < M)?

### Cosmology

10. If the scale factor doubles, what happens to the wavelength of light?

11. In a flat (k=0) matter-dominated universe, how does a(t) scale with time?

12. What is the physical meaning of the Hubble parameter H = ȧ/a?

### Gravitational Waves

13. How many independent polarizations do gravitational waves have in GR?

14. Why does gravitational radiation require quadrupole (not dipole) moments?

15. If a GW has strain h = 10^(-21) and a detector arm is L = 4 km, what is ΔL?

---

### 📝 Check Your Answers

Run the quiz script:
```bash
cd /Users/clarkcarter/Claude/personal/gr/lessons/11_gr_solutions
python quiz.py
```

---

## Next Steps

Congratulations! You've explored the major solutions to Einstein's equations:

- Black holes (Schwarzschild and Kerr)
- Cosmological models (FLRW)
- Gravitational waves
- Exotic spacetimes (de Sitter, AdS)

**Final lesson:**
- Lesson 12: GR Phenomena - See these solutions in action! Gravitational lensing, GPS corrections, black hole thermodynamics, and observational cosmology.

**Additional Resources:**
- Misner, Thorne, Wheeler: "Gravitation" (Chapter 33: Black Holes)
- Carroll: "Spacetime and Geometry" (Chapters 6-8)
- Hartle: "Gravity: An Introduction to Einstein's General Relativity"
- Wald: "General Relativity" (advanced)

---

**Ready for the finale?** → [Lesson 12: GR Phenomena](../12_gr_phenomena/LESSON.md)
