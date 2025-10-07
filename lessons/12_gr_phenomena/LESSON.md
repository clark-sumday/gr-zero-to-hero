# Lesson 12: General Relativity Phenomena

**Topics:** Gravitational Lensing, Frame Dragging, GPS Corrections, Gravitational Redshift, Black Hole Thermodynamics, Cosmological Observations

**Prerequisites:** Lessons 10-11 (GR Foundations and Solutions)

**Time:** ~6-7 hours

---

## Table of Contents

1. [Gravitational Lensing](#1-gravitational-lensing)
2. [Gravitational Redshift](#2-gravitational-redshift)
3. [Frame Dragging and the Lense-Thirring Effect](#3-frame-dragging-and-the-lense-thirring-effect)
4. [GPS and Gravitational Time Dilation](#4-gps-and-gravitational-time-dilation)
5. [Black Hole Thermodynamics](#5-black-hole-thermodynamics)
6. [Cosmological Observations](#6-cosmological-observations)
7. [Future Tests and Frontiers](#7-future-tests-and-frontiers)
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

**This is the final lesson!** We'll see General Relativity in action through real observations and applications.

**⚠️ Metric Signature Convention:** All calculations in this lesson use the **(-,+,+,+) signature** for the spacetime metric, consistent with Lessons 7, 9, 10, and 11.

---

## 1. Gravitational Lensing

### 📖 Concept

**Gravitational lensing** occurs when mass curves spacetime, bending the paths of light rays passing nearby. This was the first major experimental confirmation of General Relativity (1919 solar eclipse).

**Types of Lensing:**

1. **Strong Lensing:** Multiple images, arcs, Einstein rings
   - Massive galaxy clusters
   - Individual galaxies
   - Observable distortion

2. **Weak Lensing:** Statistical distortion of background galaxies
   - Used to map dark matter
   - Requires statistical analysis of many galaxies

3. **Microlensing:** Temporary brightening of stars
   - Compact objects (planets, brown dwarfs, black holes)
   - No image distortion visible (too small)

**Deflection Angle** (for point mass M):

```
α = 4GM/(c²b)
```

where b is the **impact parameter** (closest approach distance).

**Einstein Radius** (when source, lens, and observer are aligned):

```
θ_E = √(4GM/c² × D_LS/(D_L × D_S))
```

where D_L, D_S, D_LS are angular diameter distances.

**Physical Insight:** Light follows geodesics in curved spacetime. Near a mass, spacetime curves, so "straight" paths (geodesics) appear bent from our perspective.

---

### 💻 Code Example: Light Deflection

```python
import numpy as np  # NumPy for numerical arrays and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

def deflection_angle(M, b, G=1, c=1):
    """
    Calculate light deflection angle for point mass.

    Parameters:
    M: mass of lens
    b: impact parameter (closest approach)
    G, c: constants (default geometric units)

    Returns:
    α: deflection angle (radians)
    """
    return 4 * G * M / (c**2 * b)

# Example: Light grazing the Sun
M_sun = 1.989e30  # kg
R_sun = 6.96e8    # meters
G = 6.674e-11     # m^3/(kg·s^2)
c = 2.998e8       # m/s

# Impact parameter = solar radius (light just grazing surface)
b = R_sun
alpha_sun = deflection_angle(M_sun, b, G, c)

# Convert to arcseconds
alpha_arcsec = alpha_sun * (180/np.pi) * 3600

print("Gravitational Light Deflection by the Sun")
print("=" * 60)
print(f"Impact parameter b = {b:.2e} m (solar radius)")
print(f"Deflection angle α = {alpha_sun:.6e} radians")
print(f"                   = {alpha_arcsec:.2f} arcseconds")
print(f"\nMeasured during 1919 solar eclipse: ~1.75 arcsec ✓")
print("This confirmed Einstein's prediction!")

# Plot deflection vs impact parameter
b_values = np.linspace(R_sun, 10*R_sun, 100)  # np.linspace() creates evenly spaced array between start and end
alpha_values = [deflection_angle(M_sun, b, G, c) * (180/np.pi) * 3600
                for b in b_values]

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(b_values/R_sun, alpha_values, color=COLORS['blue'], linewidth=2)  # plt.plot() draws line plot
plt.axhline(y=alpha_arcsec, color=COLORS['red'], linestyle='--',  # plt.axhline() draws horizontal line across plot
            label=f'At surface: {alpha_arcsec:.2f}"')
plt.xlabel('Impact parameter (solar radii)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Deflection angle (arcseconds)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Light Deflection by the Sun', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels
plt.show()  # plt.show() displays the figure window
```

**Expected output:**
```
Gravitational Light Deflection by the Sun
============================================================
Impact parameter b = 6.96e+08 m (solar radius)
Deflection angle α = 8.487e-06 radians
                   = 1.75 arcseconds

Measured during 1919 solar eclipse: ~1.75 arcsec ✓
This confirmed Einstein's prediction!
```

---

### 📊 Visualization: Einstein Ring

```python
# Simulate Einstein ring formation

def einstein_ring(M_lens, D_L, D_S, D_LS, G=6.674e-11, c=2.998e8):
    """
    Calculate Einstein ring radius.

    Parameters:
    M_lens: lens mass (kg)
    D_L: distance to lens (m)
    D_S: distance to source (m)
    D_LS: distance from lens to source (m)

    Returns:
    theta_E: Einstein radius (radians)
    """
    # Einstein radius formula
    theta_E_squared = (4 * G * M_lens / c**2) * (D_LS / (D_L * D_S))
    return np.sqrt(theta_E_squared)  # np.sqrt() computes square root

# Example: Galaxy cluster lens
M_cluster = 1e14 * M_sun  # Massive cluster
D_L = 1e9 * 9.461e15      # 1 Gpc ~ 3 billion light-years
D_S = 2e9 * 9.461e15      # 2 Gpc ~ 6 billion light-years
D_LS = D_S - D_L

theta_E = einstein_ring(M_cluster, D_L, D_S, D_LS, G, c)
theta_E_arcsec = theta_E * (180/np.pi) * 3600

print(f"Einstein Ring for Galaxy Cluster")
print(f"  Cluster mass: {M_cluster/M_sun:.1e} M_sun")
print(f"  Einstein radius: {theta_E_arcsec:.2f} arcseconds")

# Visualize Einstein ring
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Configuration: Observer - Lens - Source aligned
angles = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end

# Left: Perfect alignment → Einstein ring
ax = axes[0]
ring_x = theta_E_arcsec * np.cos(angles)  # np.cos() computes cosine (element-wise for arrays)
ring_y = theta_E_arcsec * np.sin(angles)  # np.sin() computes sine (element-wise for arrays)
ax.plot(ring_x, ring_y, color=COLORS['blue'], linewidth=3, label='Einstein ring')
ax.plot(0, 0, 'o', color=COLORS['red'], markersize=15, label='Lens (galaxy)')
ax.plot(0, 0, '*', color=COLORS['yellow'], markersize=20, label='Source (behind lens)')
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.set_xlabel('θ_x (arcsec)', fontsize=11)
ax.set_ylabel('θ_y (arcsec)', fontsize=11)
ax.set_title('Perfect Alignment\n(Einstein Ring)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Middle: Slight misalignment → arcs
ax = axes[1]
offset_x = 0.3 * theta_E_arcsec
offset_y = 0.2 * theta_E_arcsec

# Two images (approximate positions for strong lensing)
image1_x = -theta_E_arcsec + offset_x
image1_y = offset_y
image2_x = theta_E_arcsec + offset_x
image2_y = offset_y

ax.plot(0, 0, 'o', color=COLORS['red'], markersize=15, label='Lens')
ax.plot(offset_x, offset_y, '*', color=COLORS['yellow'], markersize=15, label='Source (offset)')
ax.plot([image1_x], [image1_y], 's', color=COLORS['blue'], markersize=10, label='Image 1')
ax.plot([image2_x], [image2_y], 's', color=COLORS['cyan'], markersize=10, label='Image 2')

# Draw arcs
arc1_angles = np.linspace(np.pi*0.7, np.pi*1.3, 30)  # np.linspace() creates evenly spaced array between start and end
arc1_x = theta_E_arcsec * np.cos(arc1_angles) + offset_x  # np.cos() computes cosine (element-wise for arrays)
arc1_y = theta_E_arcsec * np.sin(arc1_angles) + offset_y  # np.sin() computes sine (element-wise for arrays)
ax.plot(arc1_x, arc1_y, color=COLORS['blue'], linewidth=3, alpha=0.6)

arc2_angles = np.linspace(-np.pi*0.3, np.pi*0.3, 30)  # np.linspace() creates evenly spaced array between start and end
arc2_x = theta_E_arcsec * np.cos(arc2_angles) + offset_x  # np.cos() computes cosine (element-wise for arrays)
arc2_y = theta_E_arcsec * np.sin(arc2_angles) + offset_y  # np.sin() computes sine (element-wise for arrays)
ax.plot(arc2_x, arc2_y, color=COLORS['cyan'], linewidth=3, alpha=0.6)

ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.set_xlabel('θ_x (arcsec)', fontsize=11)
ax.set_ylabel('θ_y (arcsec)', fontsize=11)
ax.set_title('Slight Misalignment\n(Giant Arcs)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Right: Large offset → weak lensing
ax = axes[2]
n_galaxies = 50
np.random.seed(42)

# Background galaxies (intrinsically circular)
for i in range(n_galaxies):
    pos_x = np.random.uniform(-15, 15)
    pos_y = np.random.uniform(-15, 15)

    # Distance from lens
    r = np.sqrt(pos_x**2 + pos_y**2)  # np.sqrt() computes square root

    # Shear (weak lensing effect) - galaxies get tangentially stretched
    shear = 0.3 * (theta_E_arcsec / max(r, 1))**2

    # Ellipse parameters
    a = 0.5 * (1 + shear)  # Semi-major axis
    b = 0.5 * (1 - shear)  # Semi-minor axis
    angle = np.arctan2(pos_y, pos_x)  # Tangential alignment

    # Draw ellipse
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((pos_x, pos_y), 2*a, 2*b, angle=np.degrees(angle),
                     facecolor='none', edgecolor=COLORS['blue'], linewidth=0.8, alpha=0.6)
    ax.add_patch(ellipse)

ax.plot(0, 0, 'o', color=COLORS['red'], markersize=15, label='Lens')
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.set_xlabel('θ_x (arcsec)', fontsize=11)
ax.set_ylabel('θ_y (arcsec)', fontsize=11)
ax.set_title('Weak Lensing\n(Tangential Distortion)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("\nThree regimes of gravitational lensing:")
print("  1. Perfect alignment → Full Einstein ring")
print("  2. Slight offset → Giant arcs and multiple images")
print("  3. Large offset → Weak shear distortions")
```

---

### 🔬 Explore: Microlensing Light Curves

```python
# Simulate microlensing event (e.g., planet detection)

def microlensing_magnification(t, t0, tE, u0):
    """
    Magnification for microlensing event.

    Parameters:
    t: time
    t0: time of closest approach
    tE: Einstein crossing time
    u0: impact parameter (in units of Einstein radius)

    Returns:
    A: magnification factor
    """
    # Normalized separation
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)  # np.sqrt() computes square root

    # Magnification formula
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))  # np.sqrt() computes square root

    return A

# Simulate event
t_event = np.linspace(-100, 100, 500)  # Days
t0 = 0      # Peak at day 0
tE = 20     # Einstein crossing time (days)
u0 = 0.1    # Close approach

magnification = [microlensing_magnification(t, t0, tE, u0) for t in t_event]

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(t_event, magnification, color=COLORS['purple'], linewidth=2)  # plt.plot() draws line plot
plt.axhline(y=1, color=COLORS['gray'], linestyle='--', label='No lensing')  # plt.axhline() draws horizontal line across plot
plt.xlabel('Time (days)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Magnification A', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Microlensing Event Light Curve', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels
plt.show()  # plt.show() displays the figure window

peak_mag = max(magnification)
print(f"Microlensing Event:")
print(f"  Einstein crossing time: {tE} days")
print(f"  Impact parameter: u₀ = {u0} θ_E")
print(f"  Peak magnification: A = {peak_mag:.2f}×")
print(f"\nThis is how we detect exoplanets via gravitational lensing!")
```

---

### 🎯 Practice Question #1

**Q:** If light is deflected by 1.75 arcseconds when grazing the Sun, what would the deflection be at twice the solar radius (b = 2R_sun)?

<details>
<summary>💡 Hint</summary>

The deflection angle α is inversely proportional to the impact parameter b.
</details>

<details>
<summary>✅ Answer</summary>

Since α ∝ 1/b, doubling the impact parameter halves the deflection:

α(2R_sun) = α(R_sun) / 2 = 1.75" / 2 = **0.875 arcseconds**

```python
alpha_1 = 1.75  # arcsec at R_sun
b_1 = 1.0       # R_sun
b_2 = 2.0       # 2 R_sun
alpha_2 = alpha_1 * (b_1 / b_2)
print(f"Deflection at 2 R_sun: {alpha_2:.3f} arcseconds")
```
</details>

---

## 2. Gravitational Redshift

### 📖 Concept

**Gravitational redshift** occurs because time runs slower in stronger gravitational fields. Light climbing out of a gravitational well loses energy, increasing its wavelength.

**Redshift Formula:**

For a photon emitted at radius r_emit and observed at r_obs:

```
z = λ_obs/λ_emit - 1 = √(g_00(r_obs)) / √(g_00(r_emit)) - 1
```

For Schwarzschild metric:

```
z = √(1 - r_s/r_obs) / √(1 - r_s/r_emit) - 1
```

**Special Cases:**

1. **Weak field** (r >> r_s):
   ```
   z ≈ (GM/c²)(1/r_emit - 1/r_obs)
   ```

2. **From surface to infinity:**
   ```
   z = 1/√(1 - r_s/R) - 1
   ```

3. **At event horizon** (r_emit → r_s):
   ```
   z → ∞  (infinite redshift!)
   ```

**Applications:**
- White dwarfs and neutron stars (measurable redshift)
- GPS satellites (must correct for Earth's gravitational field)
- Cosmological redshift (different mechanism - expansion)

---

### 💻 Code Example: Gravitational Redshift

```python
def gravitational_redshift(r_emit, r_obs, M, G=1, c=1):
    """
    Calculate gravitational redshift.

    Parameters:
    r_emit: emission radius
    r_obs: observation radius
    M: mass of gravitating body

    Returns:
    z: redshift factor
    """
    r_s = 2 * G * M / c**2

    if r_emit <= r_s or r_obs <= r_s:
        return np.inf  # Inside horizon

    # Metric components
    g00_obs = -(1 - r_s / r_obs)
    g00_emit = -(1 - r_s / r_emit)

    # Redshift
    z = np.sqrt(-g00_obs) / np.sqrt(-g00_emit) - 1  # np.sqrt() computes square root

    return z

# Example 1: White dwarf
M_wd = 0.6 * M_sun
R_wd = 5e6  # meters (Earth-sized)

z_wd = gravitational_redshift(R_wd, 1e15, M_wd, G, c)  # To infinity

print("Gravitational Redshift Examples")
print("=" * 60)
print(f"\n1. White Dwarf:")
print(f"   Mass: {M_wd/M_sun:.1f} M_sun")
print(f"   Radius: {R_wd/1e6:.0f} km")
print(f"   Redshift: z = {z_wd:.2e}")
print(f"   Wavelength shift: Δλ/λ = {z_wd*100:.4f}%")

# Example 2: Neutron star
M_ns = 1.4 * M_sun
R_ns = 1e4  # meters (10 km)

z_ns = gravitational_redshift(R_ns, 1e15, M_ns, G, c)

print(f"\n2. Neutron Star:")
print(f"   Mass: {M_ns/M_sun:.1f} M_sun")
print(f"   Radius: {R_ns/1e3:.0f} km")
print(f"   Redshift: z = {z_ns:.4f}")
print(f"   Wavelength shift: Δλ/λ = {z_ns*100:.2f}%")

# Example 3: Near black hole event horizon
M_bh = 10 * M_sun
r_s_bh = 2 * G * M_bh / c**2

# Emit just outside horizon
r_emit_vals = np.linspace(1.01*r_s_bh, 10*r_s_bh, 100)  # np.linspace() creates evenly spaced array between start and end
z_vals = [gravitational_redshift(r, 1e15, M_bh, G, c) for r in r_emit_vals]

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(r_emit_vals/r_s_bh, z_vals, color=COLORS['red'], linewidth=2)  # plt.plot() draws line plot
plt.axvline(x=1, color=COLORS['gray'], linestyle='--', label='Event horizon')  # plt.axvline() draws vertical line across plot
plt.xlabel('Emission radius / r_s', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Redshift z', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Gravitational Redshift Near Black Hole', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.yscale('log')
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.legend()  # plt.legend() displays legend with labels
plt.ylim(0.01, 100)  # plt.ylim() sets y-axis limits
plt.show()  # plt.show() displays the figure window

print(f"\n3. Black Hole (M = {M_bh/M_sun:.0f} M_sun):")
print(f"   At r = 1.5 r_s: z = {gravitational_redshift(1.5*r_s_bh, 1e15, M_bh, G, c):.2f}")
print(f"   At r = 2 r_s:   z = {gravitational_redshift(2*r_s_bh, 1e15, M_bh, G, c):.2f}")
print(f"   As r → r_s:     z → ∞")
```

---

### 📊 Visualization: Pound-Rebka Experiment

```python
# The Pound-Rebka experiment (1959) - first lab measurement of gravitational redshift

h_tower = 22.5  # meters (Harvard tower)
g = 9.8         # m/s²
c = 2.998e8     # m/s

# Predicted redshift (weak field)
z_pred = g * h_tower / c**2

# Frequency shift
f0 = 1.4e19  # Hz (gamma ray frequency, roughly)
delta_f = z_pred * f0

print("Pound-Rebka Experiment (1959)")
print("=" * 60)
print(f"Tower height: {h_tower} m")
print(f"Predicted redshift: z = gh/c² = {z_pred:.2e}")
print(f"Fractional frequency shift: Δf/f = {z_pred:.2e}")
print(f"\nFor f₀ = {f0:.2e} Hz:")
print(f"  Frequency shift: Δf = {delta_f:.2e} Hz")
print(f"\nMeasurement agreed with GR to within 1% ✓")
print("First terrestrial confirmation of gravitational redshift!")

# Visualize potential and redshift
heights = np.linspace(0, 100, 100)  # np.linspace() creates evenly spaced array between start and end
z_heights = g * heights / c**2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Gravitational potential
ax = axes[0]
potential = -g * heights
ax.plot(heights, potential, color=COLORS['blue'], linewidth=2)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=h_tower, color=COLORS['red'], linestyle='--', alpha=0.7,
           label=f'Pound-Rebka height ({h_tower} m)')
ax.set_xlabel('Height h (m)', fontsize=12)
ax.set_ylabel('Potential Φ (m²/s²)', fontsize=12)
ax.set_title('Gravitational Potential', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Right: Redshift
ax = axes[1]
ax.plot(heights, z_heights, color=COLORS['orange'], linewidth=2)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=h_tower, color=COLORS['red'], linestyle='--', alpha=0.7,
           label=f'h = {h_tower} m')
ax.plot([h_tower], [z_pred], 'o', color=COLORS['red'], markersize=10)
ax.set_xlabel('Height h (m)', fontsize=12)
ax.set_ylabel('Redshift z = gh/c²', fontsize=12)
ax.set_title('Gravitational Redshift vs Height', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window
```

---

### 🎯 Practice Question #2

**Q:** A photon is emitted from the surface of a neutron star (M = 1.4 M_sun, R = 10 km) and observed far away. If the emitted wavelength is 500 nm, what wavelength is observed?

<details>
<summary>💡 Hint 1</summary>

Use the gravitational redshift formula to find z.
</details>

<details>
<summary>💡 Hint 2</summary>

λ_obs = λ_emit × (1 + z)
</details>

<details>
<summary>✅ Answer</summary>

First calculate the Schwarzschild radius and redshift:

```python
M_ns = 1.4 * M_sun
R_ns = 10e3  # meters
r_s = 2 * G * M_ns / c**2  # ~4.1 km

z = 1/np.sqrt(1 - r_s/R_ns) - 1  # np.sqrt() computes square root
```

For these values: r_s ≈ 4.1 km, so r_s/R ≈ 0.41

z ≈ 0.30 (30% redshift!)

λ_obs = λ_emit × (1 + z) = 500 nm × 1.30 = **650 nm**

The photon shifts from green (500 nm) to red (650 nm)!

```python
lambda_emit = 500  # nm
lambda_obs = lambda_emit * (1 + z)
print(f"Emitted: {lambda_emit} nm (green)")
print(f"Observed: {lambda_obs:.0f} nm (red)")
print(f"Redshift z = {z:.2f}")
```
</details>

---

## 3. Frame Dragging and the Lense-Thirring Effect

### 📖 Concept

**Frame dragging** (Lense-Thirring effect) is the phenomenon where rotating masses "drag" spacetime around with them. This is a unique prediction of General Relativity with no Newtonian analogue.

**Physical Picture:**
Imagine a massive sphere rotating in honey. The honey near the surface gets dragged along. Similarly, a rotating mass drags the spacetime around it.

**Lense-Thirring Precession:**

For a gyroscope orbiting a rotating mass at radius r:

```
Ω_LT = (2GJ/c²r³) × [3(n̂·Ĵ)n̂ - Ĵ]
```

where:
- J is the angular momentum of the central body
- n̂ is the unit vector pointing to the orbit location
- Ĵ is the unit vector along J

**Simplified magnitude** (for equatorial orbit): |Ω_LT| ≈ GJ/(c²r³)

**Key Effects:**
1. **Gyroscope precession:** Gravity Probe B measured this around Earth
2. **Orbital plane precession:** Satellite orbits precess
3. **Ergosphere:** Region where frame-dragging is extreme (Kerr black holes)

**Gravity Probe B Results (2011):**
- Measured frame-dragging around Earth
- Agreement with GR: within 19% (limited by technical issues)
- Confirmed effect: ~0.039 arcsec/year

---

### 💻 Code Example: Frame Dragging Around Earth

```python
def lense_thirring_rate(r, J, M, G=6.674e-11, c=2.998e8):
    """
    Calculate Lense-Thirring precession rate.

    Parameters:
    r: orbital radius
    J: angular momentum of central body
    M: mass of central body

    Returns:
    Omega_LT: precession rate (rad/s)
    """
    # Lense-Thirring angular velocity
    Omega_LT = G * J / (c**2 * r**3)

    return Omega_LT

# Earth parameters
M_earth = 5.972e24  # kg
R_earth = 6.371e6   # m
I_earth = 0.33 * M_earth * R_earth**2  # Moment of inertia (rough)
omega_earth = 2 * np.pi / 86400  # rad/s (rotation rate)
J_earth = I_earth * omega_earth  # Angular momentum

# Gravity Probe B orbit
r_gpb = R_earth + 642e3  # 642 km altitude

Omega_LT = lense_thirring_rate(r_gpb, J_earth, M_earth, G, c)

# Convert to arcseconds per year
arcsec_per_year = Omega_LT * (180/np.pi) * 3600 * 365.25 * 86400

print("Frame Dragging Around Earth")
print("=" * 60)
print(f"Earth angular momentum: J = {J_earth:.2e} kg·m²/s")
print(f"Gravity Probe B orbit: r = {r_gpb/1e6:.0f} km")
print(f"\nLense-Thirring precession rate:")
print(f"  Ω_LT = {Omega_LT:.2e} rad/s")
print(f"       = {arcsec_per_year:.4f} arcsec/year")
print(f"\nGravity Probe B measured: ~0.039 arcsec/year ✓")
print("Tiny effect, but measurable with precise gyroscopes!")

# Compare with geodetic precession (de Sitter effect)
# Geodetic precession is larger: ~6.6 arcsec/year for GPB
Omega_geodetic = (3 * G * M_earth) / (2 * c**2 * r_gpb)
geodetic_arcsec = Omega_geodetic * (180/np.pi) * 3600 * 365.25 * 86400

print(f"\nFor comparison:")
print(f"  Geodetic precession: {geodetic_arcsec:.2f} arcsec/year")
print(f"  Frame dragging: {arcsec_per_year:.4f} arcsec/year")
print(f"  Ratio: {geodetic_arcsec/arcsec_per_year:.0f}:1")
```

---

### 📊 Visualization: Frame Dragging Field

```python
# Visualize frame-dragging "velocity" field around rotating Earth

from matplotlib.patches import Circle

fig, ax = plt.subplots(figsize=(10, 10))

# Grid of points
x = np.linspace(-3, 3, 20)  # np.linspace() creates evenly spaced array between start and end
y = np.linspace(-3, 3, 20)  # np.linspace() creates evenly spaced array between start and end
X, Y = np.meshgrid(x, y)  # np.meshgrid() creates coordinate matrices from coordinate vectors

# Distance from center (in Earth radii)
R_grid = np.sqrt(X**2 + Y**2)  # np.sqrt() computes square root

# Avoid singularity at center
R_grid[R_grid < 0.1] = 0.1

# Frame-dragging angular velocity (∝ 1/r³)
omega_fd = 1.0 / R_grid**3  # Normalized units

# Velocity field (tangential)
# v_φ = ω r in cylindrical coords
V_x = -omega_fd * Y
V_y = omega_fd * X

# Normalize for visualization
V_norm = np.sqrt(V_x**2 + V_y**2)  # np.sqrt() computes square root
V_x_norm = V_x / (V_norm + 0.01)
V_y_norm = V_y / (V_norm + 0.01)

# Plot vector field
ax.quiver(X, Y, V_x_norm, V_y_norm, V_norm,
         cmap='viridis', scale=25, width=0.003, alpha=0.7)

# Earth
earth = Circle((0, 0), 1, color=COLORS['blue'], alpha=0.7, label='Earth')
ax.add_patch(earth)

# Rotation arrow
ax.annotate('', xy=(0, 0.7), xytext=(0, 1.2),
           arrowprops=dict(arrowstyle='->', lw=3, color='white'))
ax.text(0.3, 1.0, 'Rotation', fontsize=12, color='white', fontweight='bold')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')  # .set_aspect() sets aspect ratio of axes
ax.set_xlabel('x (Earth radii)', fontsize=12)
ax.set_ylabel('y (Earth radii)', fontsize=12)
ax.set_title('Frame Dragging: Spacetime "Twisted" by Rotation', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.colorbar(ax.collections[0], ax=ax, label='Frame-drag strength (normalized)')  # plt.colorbar() adds color scale bar to plot
plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Spacetime is 'twisted' around rotating masses.")
print("Effect is strongest near the surface, falls off as 1/r³.")
```

---

### 🔬 Explore: Kerr Black Hole Ergosphere

```python
# In the ergosphere of a Kerr black hole, frame-dragging is so strong
# that nothing can remain stationary - everything must co-rotate!

def ergosphere_radius(theta, M, a):
    """
    Radius of ergosphere for Kerr black hole.

    Parameters:
    theta: polar angle (0 = north pole, π/2 = equator)
    M: mass (in geometric units where G=c=1)
    a: spin parameter (in geometric units, dimensionless, a ≤ M)

    Returns:
    r_ergo: ergosphere radius (in geometric units)

    Note: In SI units, replace M → GM/c², a → Ja/(Mc)
    """
    return M + np.sqrt(M**2 - a**2 * np.cos(theta)**2)  # np.cos() computes cosine (element-wise for arrays)

M = 1.0
a = 0.9 * M  # Fast spin

theta_vals = np.linspace(0, np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
r_ergo = [ergosphere_radius(th, M, a) for th in theta_vals]

# Event horizon (constant radius for Kerr)
r_plus = M + np.sqrt(M**2 - a**2)  # np.sqrt() computes square root

# Convert to Cartesian for visualization
x_ergo = [r * np.sin(th) for r, th in zip(r_ergo, theta_vals)]  # np.sin() computes sine (element-wise for arrays)
z_ergo = [r * np.cos(th) for r, th in zip(r_ergo, theta_vals)]  # np.cos() computes cosine (element-wise for arrays)

plt.figure(figsize=(10, 8))  # plt.figure() creates a new figure for plotting
plt.plot(x_ergo, z_ergo, color=COLORS['red'], linewidth=3, label='Ergosphere')  # plt.plot() draws line plot
plt.plot([-x for x in x_ergo], z_ergo, color=COLORS['red'], linewidth=3)  # plt.plot() draws line plot

# Event horizon (circle)
horizon_theta = np.linspace(0, 2*np.pi, 100)  # np.linspace() creates evenly spaced array between start and end
x_horizon = r_plus * np.cos(horizon_theta)  # np.cos() computes cosine (element-wise for arrays)
y_horizon = r_plus * np.sin(horizon_theta)  # np.sin() computes sine (element-wise for arrays)
plt.plot(x_horizon, y_horizon, color=COLORS['blue'], linewidth=3, label='Event horizon')  # plt.plot() draws line plot

# Fill ergosphere region
plt.fill_between(x_ergo, z_ergo, -np.array(z_ergo), alpha=0.2, color=COLORS['red'])  # np.array() converts Python list/tuple to efficient numpy array
plt.fill_between([-x for x in x_ergo], z_ergo, -np.array(z_ergo), alpha=0.2, color=COLORS['red'])  # np.array() converts Python list/tuple to efficient numpy array

plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.axvline(x=0, color='k', linewidth=0.5)  # plt.axvline() draws vertical line across plot
plt.xlabel('x / M', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('z / M (rotation axis)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title(f'Kerr Black Hole Ergosphere (a = {a:.1f} M)', fontsize=14, fontweight='bold')  # plt.title() sets plot title
plt.legend(fontsize=11)  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.axis('equal')
plt.xlim(-2.5, 2.5)  # plt.xlim() sets x-axis limits
plt.ylim(-2.5, 2.5)  # plt.ylim() sets y-axis limits
plt.show()  # plt.show() displays the figure window

print(f"Ergosphere (oblate):")
print(f"  At equator (θ=π/2): r = {ergosphere_radius(np.pi/2, M, a):.3f} M")
print(f"  At pole (θ=0):      r = {ergosphere_radius(0, M, a):.3f} M")
print(f"Event horizon: r_+ = {r_plus:.3f} M")
print(f"\nInside ergosphere: MUST co-rotate with black hole!")
print("This is extreme frame dragging!")
```

---

## 4. GPS and Gravitational Time Dilation

### 📖 Concept

The **Global Positioning System (GPS)** must account for *both* Special and General Relativity effects to achieve meter-level accuracy!

**Two competing effects:**

1. **Special Relativity (kinematic time dilation):**
   Satellites move at v ≈ 3.9 km/s, so their clocks run *slower*
   ```
   Δt_SR = -v²/(2c²) × t     [first-order approximation for v << c]
   ```
   Effect: -7.2 μs/day (clock runs slow)

2. **General Relativity (gravitational time dilation):**
   Satellites are higher in Earth's gravitational potential, so clocks run *faster*
   ```
   Δt_GR = (GM/c²)(1/R_earth - 1/r_sat) × t     [weak-field approximation]
   ```
   Effect: +45.9 μs/day (clock runs fast)

**Net effect:** +38.7 μs/day (GR wins!)

Without corrections, GPS errors would accumulate at ~10 km/day!

---

### 💻 Code Example: GPS Relativistic Corrections

```python
def gps_relativistic_effects(r_sat, v_sat, t_days=1):
    """
    Calculate SR and GR time dilation for GPS satellite.

    Parameters:
    r_sat: satellite orbital radius (m)
    v_sat: satellite orbital velocity (m/s)
    t_days: time period (days)

    Returns:
    dt_SR, dt_GR, dt_total: time corrections (microseconds)
    """
    # Constants
    c = 2.998e8           # m/s
    G = 6.674e-11         # m^3/(kg·s^2)
    M_earth = 5.972e24    # kg
    R_earth = 6.371e6     # m

    # Convert time to seconds
    t_sec = t_days * 86400

    # Special Relativity: Δt = -v²/(2c²) × t
    dt_SR = -(v_sat**2 / (2 * c**2)) * t_sec

    # General Relativity: Δt = (GM/c²)(1/R - 1/r) × t
    dt_GR = (G * M_earth / c**2) * (1/R_earth - 1/r_sat) * t_sec

    # Total
    dt_total = dt_SR + dt_GR

    # Convert to microseconds
    return dt_SR * 1e6, dt_GR * 1e6, dt_total * 1e6

# GPS satellite parameters
h_gps = 20200e3  # Altitude: 20,200 km
r_gps = R_earth + h_gps
v_gps = np.sqrt(G * M_earth / r_gps)  # Orbital velocity

dt_SR, dt_GR, dt_total = gps_relativistic_effects(r_gps, v_gps, t_days=1)

print("GPS Satellite Relativistic Corrections")
print("=" * 60)
print(f"Orbital altitude: {h_gps/1e3:.0f} km")
print(f"Orbital radius: {r_gps/1e6:.2f} × 10⁶ m")
print(f"Orbital velocity: {v_gps/1e3:.2f} km/s")
print(f"\nTime dilation effects (per day):")
print(f"  Special Relativity: {dt_SR:.1f} μs (clock runs SLOW)")
print(f"  General Relativity: {dt_GR:.1f} μs (clock runs FAST)")
print(f"  Net effect:         {dt_total:.1f} μs (GR dominates!)")
print(f"\nWithout correction:")
print(f"  Position error: ~{abs(dt_total * 1e-6 * c / 1e3):.0f} km/day")
print(f"\nGPS proves General Relativity every day! ✓")

# Visualize corrections over time
days = np.linspace(0, 10, 100)  # np.linspace() creates evenly spaced array between start and end
corrections = [gps_relativistic_effects(r_gps, v_gps, t)[2] for t in days]

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(days, corrections, color=COLORS['purple'], linewidth=2)  # plt.plot() draws line plot
plt.xlabel('Time (days)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Accumulated time error (μs)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('GPS Clock Error Without Relativistic Corrections', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.axhline(y=0, color='k', linewidth=0.5)  # plt.axhline() draws horizontal line across plot
plt.show()  # plt.show() displays the figure window

# Position error
position_errors = [abs(c * 1e-6) * c / 1e3 for c in corrections]  # km

plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(days, position_errors, color=COLORS['red'], linewidth=2)  # plt.plot() draws line plot
plt.xlabel('Time (days)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Position error (km)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('GPS Position Error Without Relativistic Corrections', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.show()  # plt.show() displays the figure window
```

---

### 📊 Visualization: SR vs GR Contributions

```python
# Compare SR and GR effects at different altitudes

altitudes = np.linspace(0, 30000e3, 100)  # 0 to 30,000 km
radii = R_earth + altitudes

dt_SR_vals = []
dt_GR_vals = []
dt_total_vals = []

for r in radii:
    v = np.sqrt(G * M_earth / r)  # Circular orbit velocity
    dt_SR, dt_GR, dt_tot = gps_relativistic_effects(r, v, t_days=1)
    dt_SR_vals.append(dt_SR)
    dt_GR_vals.append(dt_GR)
    dt_total_vals.append(dt_tot)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Individual contributions
ax = axes[0]
ax.plot(altitudes/1e3, dt_SR_vals, color=COLORS['blue'], linewidth=2, label='SR (kinematic)')
ax.plot(altitudes/1e3, dt_GR_vals, color=COLORS['orange'], linewidth=2, label='GR (gravitational)')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=h_gps/1e3, color=COLORS['red'], linestyle='--', alpha=0.7, label=f'GPS altitude')
ax.set_xlabel('Altitude (km)', fontsize=12)
ax.set_ylabel('Time correction (μs/day)', fontsize=12)
ax.set_title('SR vs GR Contributions', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Net effect
ax = axes[1]
ax.plot(altitudes/1e3, dt_total_vals, color=COLORS['purple'], linewidth=2)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=h_gps/1e3, color=COLORS['red'], linestyle='--', alpha=0.7,
          label=f'GPS: {dt_total:.1f} μs/day')
ax.set_xlabel('Altitude (km)', fontsize=12)
ax.set_ylabel('Net time correction (μs/day)', fontsize=12)
ax.set_title('Total Relativistic Effect', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Notice:")
print("- SR effect (blue): increases with altitude (higher velocity)")
print("- GR effect (orange): increases with altitude (weaker gravity)")
print("- GR dominates at GPS altitude!")
```

---

### 🎯 Practice Question #3

**Q:** If GPS satellites did not correct for relativistic effects, how much position error would accumulate in one hour?

<details>
<summary>💡 Hint</summary>

Use the time error of 38.7 μs/day and scale it to 1 hour. Then multiply by c to get distance.
</details>

<details>
<summary>✅ Answer</summary>

Time error per hour: (38.7 μs/day) × (1 hour / 24 hours) = 1.61 μs/hour

Position error: Δx = c × Δt = 3×10⁸ m/s × 1.61×10⁻⁶ s ≈ **483 meters**

After just one hour, GPS would be off by nearly half a kilometer!

```python
dt_per_day = 38.7e-6  # seconds
dt_per_hour = dt_per_day / 24
position_error = c * dt_per_hour
print(f"Time error per hour: {dt_per_hour*1e6:.2f} μs")
print(f"Position error: {position_error:.0f} meters")
```
</details>

---

## 5. Black Hole Thermodynamics

### 📖 Concept

Black holes obey laws analogous to thermodynamics! This deep connection hints at quantum gravity.

**The Four Laws:**

**0th Law:** Surface gravity κ is constant over the event horizon
- Like temperature being uniform in thermal equilibrium

**1st Law (Energy conservation):**
```
dM = (κ/8πG) dA + Ω_H dJ + Φ_H dQ
```
- dM: mass change
- dA: horizon area change
- dJ: angular momentum change
- dQ: charge change
- Analogous to dE = T dS + work terms

**2nd Law (Area theorem):**
```
dA ≥ 0
```
- Horizon area never decreases (classically)
- Analogous to entropy increasing

**3rd Law:** Cannot reach κ = 0 (extremal black hole) in finite steps
- Like absolute zero in thermodynamics

**Bekenstein-Hawking Entropy:**
```
S_BH = (k_B c³ A) / (4 ℏ G) = (k_B A) / (4 L_P²)
```

where L_P = √(ℏG/c³) is the Planck length.

**Hawking Temperature:**
```
T_H = (ℏ c³ κ) / (2π k_B G M)
```

For Schwarzschild: T_H ∝ 1/M (larger black holes are colder!)

---

### 💻 Code Example: Black Hole Thermodynamics

```python
def black_hole_thermodynamics(M, units='SI'):
    """
    Calculate thermodynamic properties of Schwarzschild black hole.

    Parameters:
    M: mass (kg in SI units, solar masses if units='solar')
    units: 'SI' or 'solar'

    Returns:
    dict with r_s, A, S, T, t_evap
    """
    if units == 'solar':
        M_kg = M * M_sun
    else:
        M_kg = M

    # Constants
    G = 6.674e-11       # m^3/(kg·s^2)
    c = 2.998e8         # m/s
    hbar = 1.055e-34    # J·s
    k_B = 1.381e-23     # J/K

    # Schwarzschild radius
    r_s = 2 * G * M_kg / c**2

    # Event horizon area
    A = 4 * np.pi * r_s**2

    # Bekenstein-Hawking entropy
    S = (k_B * c**3 * A) / (4 * hbar * G)

    # Hawking temperature
    T_H = (hbar * c**3) / (8 * np.pi * G * M_kg * k_B)

    # Evaporation time (approximate)
    t_evap = (5120 * np.pi * G**2 * M_kg**3) / (hbar * c**4)

    return {
        'mass_kg': M_kg,
        'r_s_m': r_s,
        'area_m2': A,
        'entropy_JK': S,
        'entropy_bits': S / (k_B * np.log(2)),
        'temperature_K': T_H,
        'evap_time_s': t_evap,
        'evap_time_years': t_evap / (365.25 * 86400)
    }

# Examples
masses = [1, 10, 1e6, 1e9]  # Solar masses
names = ['Stellar', '10 M_sun', 'Supermassive (Sgr A*)', 'Quasar']

print("Black Hole Thermodynamics")
print("=" * 80)

for M, name in zip(masses, names):
    props = black_hole_thermodynamics(M, units='solar')

    print(f"\n{name} Black Hole (M = {M:.0e} M_sun):")
    print(f"  Schwarzschild radius: {props['r_s_m']/1e3:.2e} km")
    print(f"  Event horizon area: {props['area_m2']:.2e} m²")
    print(f"  Entropy: S = {props['entropy_JK']:.2e} J/K")
    print(f"           = {props['entropy_bits']:.2e} bits")
    print(f"  Hawking temperature: T = {props['temperature_K']:.2e} K")
    print(f"  Evaporation time: {props['evap_time_years']:.2e} years")

# Compare with Sun's entropy (rough estimate)
S_sun_rough = 1e58 * k_B  # Approximate
props_1solar = black_hole_thermodynamics(1, units='solar')
print(f"\n" + "=" * 80)
print(f"Comparison:")
print(f"  Sun's thermal entropy: ~{S_sun_rough:.2e} J/K")
print(f"  1 M_sun black hole entropy: {props_1solar['entropy_JK']:.2e} J/K")
print(f"  Ratio: {props_1solar['entropy_JK']/S_sun_rough:.2e}")
print(f"\nBlack holes are the most entropic objects in the universe!")
```

---

### 📊 Visualization: Hawking Temperature vs Mass

```python
# Plot temperature and evaporation time vs black hole mass

M_range = np.logspace(-10, 10, 200)  # Solar masses (huge range!)

temperatures = []
evap_times = []

for M in M_range:
    props = black_hole_thermodynamics(M, units='solar')
    temperatures.append(props['temperature_K'])
    evap_times.append(props['evap_time_years'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Temperature
ax = axes[0]
ax.loglog(M_range, temperatures, color=COLORS['red'], linewidth=2)
ax.axhline(y=2.7, color=COLORS['blue'], linestyle='--', alpha=0.7, label='CMB temp (2.7 K)')
ax.axvline(x=1, color=COLORS['gray'], linestyle='--', alpha=0.7, label='1 M_sun')
ax.set_xlabel('Black Hole Mass (M_sun)', fontsize=12)
ax.set_ylabel('Hawking Temperature (K)', fontsize=12)
ax.set_title('Temperature vs Mass (T ∝ 1/M)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Right: Evaporation time
ax = axes[1]
age_universe = 13.8e9  # years
ax.loglog(M_range, evap_times, color=COLORS['purple'], linewidth=2)
ax.axhline(y=age_universe, color=COLORS['orange'], linestyle='--', alpha=0.7,
          label='Age of universe')
ax.axvline(x=1, color=COLORS['gray'], linestyle='--', alpha=0.7, label='1 M_sun')
ax.set_xlabel('Black Hole Mass (M_sun)', fontsize=12)
ax.set_ylabel('Evaporation Time (years)', fontsize=12)
ax.set_title('Evaporation Time vs Mass (t ∝ M³)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

# Find mass where T = 2.7 K (CMB temperature)
# Black holes colder than this will grow (absorb more than they radiate)
M_critical = None
for M in M_range:
    props = black_hole_thermodynamics(M, units='solar')
    if props['temperature_K'] <= 2.7:
        M_critical = M
        break

print(f"Critical mass (T = CMB): M ≈ {M_critical:.2e} M_sun")
print(f"\nBlack holes larger than this are COLDER than the CMB.")
print(f"They grow by absorbing CMB radiation (net).")
print(f"\nStellar-mass black holes won't evaporate for ~10^67 years!")
```

---

### 🔬 Explore: Information Paradox

```python
print("Black Hole Information Paradox")
print("=" * 60)
print()
print("Setup:")
print("  1. Drop a book into a black hole")
print("  2. Black hole evaporates via Hawking radiation")
print("  3. Hawking radiation is thermal (random)")
print()
print("Question: Where did the information (text in the book) go?")
print()
print("Possible resolutions:")
print("  A) Information is destroyed → violates quantum mechanics")
print("  B) Information escapes in radiation → how?")
print("  C) Information remains in black hole remnant")
print("  D) Information is encoded on horizon (holographic principle)")
print()
print("Current thinking: Option D - Holographic principle")
print("  - Information is encoded on the 2D horizon surface")
print("  - Related to AdS/CFT correspondence (string theory)")
print("  - Entropy S ∝ Area (not Volume!) supports this")
print()
print("Still an active area of research!")
```

---

## 6. Cosmological Observations

### 📖 Concept

General Relativity predictions have been confirmed by cosmological observations.

**Key Observations:**

1. **Hubble Expansion:**
   - Galaxies recede with v = H₀ d
   - H₀ ≈ 70 km/s/Mpc (current measurement)
   - Predicted by FLRW cosmology

2. **Cosmic Microwave Background (CMB):**
   - Thermal radiation from early universe (T ≈ 2.7 K)
   - Released at recombination (z ≈ 1100)
   - Confirms hot Big Bang model

3. **Nucleosynthesis:**
   - Abundance of light elements (H, He, Li)
   - Matches predictions from early universe physics

4. **Large Scale Structure:**
   - Distribution of galaxies
   - Gravitational instability in expanding universe

5. **Accelerating Expansion:**
   - Type Ia supernovae observations (1998)
   - Evidence for dark energy (Λ > 0)
   - Nobel Prize 2011

---

### 💻 Code Example: Hubble Law

```python
# Hubble's law: v = H₀ × d

def hubble_velocity(distance, H0=70):
    """
    Calculate recession velocity from Hubble law.

    Parameters:
    distance: distance in Mpc (megaparsecs)
    H0: Hubble constant (km/s/Mpc)

    Returns:
    velocity in km/s
    """
    return H0 * distance

# Example galaxies
galaxies = [
    ("Virgo Cluster", 16.5),      # Mpc
    ("Coma Cluster", 100),
    ("Distant galaxy", 1000),
    ("Very distant", 3000)
]

print("Hubble's Law: v = H₀ × d")
print(f"Hubble constant: H₀ = 70 km/s/Mpc")
print("=" * 60)

distances = []
velocities = []

for name, d in galaxies:
    v = hubble_velocity(d)
    distances.append(d)
    velocities.append(v)

    # Redshift
    c = 3e5  # km/s
    z = v / c if v < c else 'non-linear'

    print(f"\n{name}:")
    print(f"  Distance: {d:.1f} Mpc")
    print(f"  Recession velocity: {v:.0f} km/s")
    if isinstance(z, float):
        print(f"  Redshift: z ≈ {z:.3f}")

# Plot Hubble diagram
plt.figure(figsize=(10, 6))  # plt.figure() creates a new figure for plotting
plt.plot(distances, velocities, 'o', color=COLORS['blue'], markersize=10)  # plt.plot() draws line plot

# Linear fit
d_fit = np.linspace(0, max(distances), 100)  # np.linspace() creates evenly spaced array between start and end
v_fit = hubble_velocity(d_fit)
plt.plot(d_fit, v_fit, '--', color=COLORS['red'], linewidth=2, label=f'v = {70} × d')  # plt.plot() draws line plot

# Speed of light
c_line = 3e5
plt.axhline(y=c_line, color=COLORS['gray'], linestyle=':', label='Speed of light')  # plt.axhline() draws horizontal line across plot

plt.xlabel('Distance (Mpc)', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Recession velocity (km/s)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title("Hubble's Law: The Expanding Universe", fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.legend()  # plt.legend() displays legend with labels
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.show()  # plt.show() displays the figure window

# Hubble time (rough age of universe)
H0_SI = 70 * 1e3 / (3.086e22)  # Convert to 1/s
t_hubble = 1 / H0_SI  # seconds
t_hubble_years = t_hubble / (365.25 * 86400)

print(f"\n" + "=" * 60)
print(f"Hubble time: t_H = 1/H₀ ≈ {t_hubble_years/1e9:.1f} billion years")
print(f"(Rough estimate of universe age)")
print(f"Actual age: ~13.8 billion years (ΛCDM model)")
```

---

### 📊 Visualization: CMB Power Spectrum

```python
# Simplified CMB angular power spectrum

# Realistic values (approximate)
ell = np.logspace(0.5, 3.5, 100)  # Multipole moment

# Simplified model (not accurate, just illustrative)
def cmb_power_spectrum(ell):
    """Simplified CMB temperature power spectrum."""
    # Peak around ell ~ 200 (roughly 1 degree)
    peak1 = 6000 * np.exp(-((ell - 220) / 80)**2)  # np.exp() computes exponential e^x
    peak2 = 3000 * np.exp(-((ell - 540) / 120)**2)  # np.exp() computes exponential e^x
    peak3 = 2000 * np.exp(-((ell - 800) / 150)**2)  # np.exp() computes exponential e^x

    # Damping at high ell
    damping = np.exp(-ell / 2000)  # np.exp() computes exponential e^x

    return (peak1 + peak2 + peak3) * damping

C_ell = cmb_power_spectrum(ell)

plt.figure(figsize=(12, 6))  # plt.figure() creates a new figure for plotting
plt.plot(ell, C_ell, color=COLORS['red'], linewidth=2)  # plt.plot() draws line plot
plt.xlabel('Multipole moment ℓ', fontsize=12)  # plt.xlabel() sets x-axis label
plt.ylabel('Power C_ℓ (μK²)', fontsize=12)  # plt.ylabel() sets y-axis label
plt.title('Cosmic Microwave Background Angular Power Spectrum (Simplified)', fontsize=13, fontweight='bold')  # plt.title() sets plot title
plt.xscale('log')
plt.grid(True, alpha=0.3)  # plt.grid() adds grid lines to plot
plt.annotate('1st acoustic peak\n(~1° scale)', xy=(220, 6000), xytext=(100, 7000),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=11)
plt.show()  # plt.show() displays the figure window

print("CMB Power Spectrum:")
print("  - Peaks correspond to acoustic oscillations in early universe")
print("  - Peak positions constrain cosmological parameters")
print("  - Measured by WMAP, Planck satellites")
print("\nResults:")
print("  - Flat universe (k = 0)")
print("  - Ω_matter ≈ 0.3, Ω_Λ ≈ 0.7")
print("  - Age: 13.8 billion years")
print("\nPrecision cosmology confirms GR + ΛCDM model!")
```

---

### 🎯 Practice Question #4

**Q:** A galaxy is at distance d = 100 Mpc. Using H₀ = 70 km/s/Mpc, what is its recession velocity? What is the approximate redshift?

<details>
<summary>💡 Hint</summary>

Use Hubble's law: v = H₀ × d. For small velocities, z ≈ v/c.
</details>

<details>
<summary>✅ Answer</summary>

v = H₀ × d = 70 km/s/Mpc × 100 Mpc = **7000 km/s**

Redshift: z ≈ v/c = 7000 / 300000 ≈ **0.023**

```python
H0 = 70  # km/s/Mpc
d = 100  # Mpc
v = H0 * d
c = 3e5  # km/s
z = v / c

print(f"Distance: {d} Mpc")
print(f"Velocity: {v} km/s")
print(f"Redshift: z ≈ {z:.3f}")
```
</details>

---

## 7. Future Tests and Frontiers

### 📖 Concept

General Relativity continues to be tested at ever-higher precision!

**Ongoing and Future Tests:**

1. **Gravitational Wave Astronomy:**
   - LIGO/Virgo: Binary mergers
   - LISA (space-based): Supermassive black hole mergers
   - Pulsar timing arrays: Nanohertz GWs

2. **Black Hole Imaging:**
   - Event Horizon Telescope (EHT): M87*, Sgr A*
   - Testing strong-field GR

3. **Pulsar Timing:**
   - Binary pulsars: Orbital decay from GW emission
   - Tests of equivalence principle
   - Potential gravitational wave background detection

4. **Cosmological Probes:**
   - Dark energy surveys (DES, Euclid, Rubin)
   - CMB polarization (searching for primordial GWs)
   - 21cm cosmology (early universe)

5. **Solar System Tests:**
   - Mercury perihelion precession
   - Light deflection during solar eclipses
   - Shapiro time delay
   - Frame dragging (LAGEOS, Gravity Probe B)

**Open Questions:**

- What is dark energy?
- What is dark matter? (Not a GR question, but related)
- Quantum gravity: How to unify GR and quantum mechanics?
- Information paradox resolution
- Cosmic censorship: Are naked singularities possible?
- Nature of singularities: What happens at r = 0?

---

### 💻 Code Example: Tests of GR Summary

```python
# Summary of key GR tests and their status

tests_summary = {
    'Light Deflection (1919)': {
        'Prediction': '1.75 arcsec (grazing Sun)',
        'Measurement': '1.75 ± 0.2 arcsec',
        'Status': 'CONFIRMED ✓',
        'Precision': '~10%'
    },
    'Mercury Perihelion (1915)': {
        'Prediction': '43 arcsec/century (anomalous)',
        'Measurement': '43.1 ± 0.5 arcsec/century',
        'Status': 'CONFIRMED ✓',
        'Precision': '~1%'
    },
    'Gravitational Redshift (1959)': {
        'Prediction': 'Δf/f = gh/c²',
        'Measurement': 'Pound-Rebka: within 1%',
        'Status': 'CONFIRMED ✓',
        'Precision': '~1% (1959), ~0.01% (modern)'
    },
    'Time Dilation (GPS)': {
        'Prediction': '38 μs/day (satellite clocks)',
        'Measurement': 'Daily corrections applied',
        'Status': 'CONFIRMED ✓',
        'Precision': '~1 ns (operational)'
    },
    'Frame Dragging (2011)': {
        'Prediction': '0.039 arcsec/year (Earth)',
        'Measurement': 'Gravity Probe B: within 19%',
        'Status': 'CONFIRMED ✓',
        'Precision': '~20%'
    },
    'Binary Pulsar (1975-)': {
        'Prediction': 'Orbital decay from GW emission',
        'Measurement': 'Matches to 0.1% over decades',
        'Status': 'CONFIRMED ✓',
        'Precision': '~0.1%'
    },
    'Gravitational Waves (2015)': {
        'Prediction': 'GW from binary black holes',
        'Measurement': 'LIGO detections (100+ events)',
        'Status': 'CONFIRMED ✓',
        'Precision': 'Shape matches within ~1%'
    },
    'Black Hole Shadow (2019)': {
        'Prediction': 'Shadow ~2.6 r_s',
        'Measurement': 'M87*: 42 ± 3 μas',
        'Status': 'CONFIRMED ✓',
        'Precision': '~7%'
    },
    'Cosmological Expansion': {
        'Prediction': 'FLRW metric, accelerating expansion',
        'Measurement': 'SNe Ia, CMB, BAO',
        'Status': 'CONFIRMED ✓',
        'Precision': '<1% on H₀ (controversial)'
    }
}

print("=" * 80)
print("TESTS OF GENERAL RELATIVITY")
print("=" * 80)

for test_name, info in tests_summary.items():
    print(f"\n{test_name}:")
    print(f"  Prediction: {info['Prediction']}")
    print(f"  Measurement: {info['Measurement']}")
    print(f"  Status: {info['Status']}")
    print(f"  Precision: {info['Precision']}")

print("\n" + "=" * 80)
print("General Relativity has passed every test for over 100 years!")
print("From millimeter scales (lab tests) to billion light-year scales (cosmology)")
print("=" * 80)
```

---

### 📊 Visualization: GR Tests Timeline

```python
# Timeline of major GR confirmations

events = [
    (1915, 'Einstein publishes GR', 'theory'),
    (1919, 'Solar eclipse light deflection', 'confirm'),
    (1959, 'Pound-Rebka experiment', 'confirm'),
    (1975, 'Binary pulsar discovered', 'confirm'),
    (1998, 'Accelerating expansion (SNe Ia)', 'discover'),
    (2011, 'Gravity Probe B results', 'confirm'),
    (2015, 'First GW detection (LIGO)', 'confirm'),
    (2019, 'First black hole image (M87*)', 'confirm'),
    (2022, 'Sgr A* image (our galaxy)', 'confirm')
]

years = [e[0] for e in events]
names = [e[1] for e in events]
types = [e[2] for e in events]

fig, ax = plt.subplots(figsize=(14, 8))

# Plot timeline
for i, (year, name, typ) in enumerate(events):
    color = COLORS['blue'] if typ == 'confirm' else COLORS['orange']
    marker = 'o' if typ == 'confirm' else '*'
    size = 12 if typ == 'confirm' else 20

    ax.plot(year, 0, marker, color=color, markersize=size)

    # Alternate text above/below
    y_text = 0.3 if i % 2 == 0 else -0.3
    va = 'bottom' if i % 2 == 0 else 'top'

    ax.annotate(f"{year}\n{name}",
               xy=(year, 0), xytext=(year, y_text),
               ha='center', va=va, fontsize=10,
               arrowprops=dict(arrowstyle='->', lw=1, color='black'))

ax.axhline(y=0, color='black', linewidth=2)
ax.set_xlim(1910, 2025)
ax.set_ylim(-0.8, 0.8)
ax.set_xlabel('Year', fontsize=13)
ax.set_title('Timeline of General Relativity Confirmations', fontsize=14, fontweight='bold')
ax.set_yticks([])
ax.grid(True, axis='x', alpha=0.3)

# Legend
ax.plot([], [], 'o', color=COLORS['blue'], markersize=10, label='Confirmation')
ax.plot([], [], '*', color=COLORS['orange'], markersize=15, label='Discovery')
ax.legend(loc='upper left', fontsize=11)

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot spacing
plt.show()  # plt.show() displays the figure window

print("Over a century of testing, GR continues to succeed!")
```

---

## Practice Questions

Final comprehensive questions:

### Gravitational Lensing

1. Calculate the deflection angle for light passing the Sun at 2 solar radii.

2. What is an Einstein ring and when does it form?

3. How can weak gravitational lensing be used to map dark matter?

### Gravitational Redshift and Time Dilation

4. Calculate the gravitational redshift for light emitted from a white dwarf (M = 0.6 M_sun, R = 5000 km).

5. Why do GPS satellites need relativistic corrections? Which effect dominates?

6. At what altitude would SR and GR time dilation effects cancel (for Earth)?

### Frame Dragging

7. What is the physical interpretation of frame dragging?

8. Where is frame dragging strongest around a Kerr black hole?

9. How did Gravity Probe B measure frame dragging?

### Black Hole Thermodynamics

10. Calculate the Hawking temperature of a solar-mass black hole.

11. Why do larger black holes have lower temperatures?

12. What is the black hole information paradox?

### Cosmological Observations

13. A galaxy has redshift z = 0.1. What is its approximate recession velocity?

14. What are the three main pieces of evidence for the Big Bang?

15. What observations led to the discovery of dark energy?

---

### 📝 Check Your Answers

Run the quiz script:
```bash
cd /Users/clarkcarter/Claude/personal/gr/lessons/12_gr_phenomena
python quiz.py
```

---

## Conclusion: You've Completed the GR Journey!

**Congratulations!** You've completed the full General Relativity tutorial, from linear algebra to black hole thermodynamics. You now understand:

**Mathematics:**
- Linear algebra, calculus, differential equations
- Differential geometry, manifolds, tensors
- Riemannian geometry and curvature

**Physics:**
- Special and General Relativity
- Einstein's field equations
- Solutions: Schwarzschild, Kerr, FLRW
- Gravitational waves

**Phenomena:**
- Gravitational lensing
- Time dilation and redshift
- Frame dragging
- Black hole thermodynamics
- Cosmological observations

**Applications:**
- GPS corrections
- LIGO detections
- Black hole imaging
- Cosmology

---

## What's Next?

**If you want to go deeper:**

1. **Advanced GR Textbooks:**
   - Misner, Thorne, Wheeler: "Gravitation"
   - Wald: "General Relativity"
   - Carroll: "Spacetime and Geometry"

2. **Specialized Topics:**
   - Numerical Relativity (black hole simulations)
   - Cosmological Perturbation Theory
   - Quantum Field Theory in Curved Spacetime
   - String Theory and Quantum Gravity

3. **Research Areas:**
   - Gravitational wave astronomy
   - Black hole physics
   - Cosmology and dark energy
   - Tests of GR in strong fields

4. **Computational Tools:**
   - EinsteinPy (Python GR library)
   - SageMath (differential geometry)
   - Numerical relativity codes

**Stay curious and keep exploring!**

**Additional Resources:**
- arXiv.org: gr-qc (General Relativity and Quantum Cosmology)
- Living Reviews in Relativity
- Stack Exchange (Physics)
- YouTube: Leonard Susskind's lectures

---

**Thank you for learning General Relativity with us!**

You've mastered one of the most beautiful and profound theories in physics. Einstein would be proud.

*"The most incomprehensible thing about the universe is that it is comprehensible."* — Albert Einstein

---

**End of Lesson 12 • End of GR Tutorial**
