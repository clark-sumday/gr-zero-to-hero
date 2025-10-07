#!/usr/bin/env python3
"""
Example: Kerr Solution - Rotating Black Holes
Demonstrates frame dragging, ergosphere, and spin effects
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("KERR SOLUTION: ROTATING BLACK HOLES")
print("=" * 60)
print("\nKerr metric (1963): Describes rotating black holes")
print("Characterized by mass M and spin parameter a = J/M")
print("where J is angular momentum")

# Geometric units: G = c = 1
M = 1.0  # Mass

# Different spin parameters
a_values = [0, 0.5, 0.9, 0.998]  # 0 = Schwarzschild, 1 = extremal
spin_labels = ['No spin\n(Schwarzschild)', 'Slow spin\n(a=0.5M)',
               'Fast spin\n(a=0.9M)', 'Near-extremal\n(a≈M)']

print(f"\nBlack hole mass: M = {M}")
print(f"\nSpin parameter a = J/M (where J is angular momentum)")
print(f"  • a = 0: Non-rotating (Schwarzschild)")
print(f"  • 0 < a < M: Rotating (Kerr)")
print(f"  • a = M: Extremal (maximum spin)")

# Key radii for different spins
def kerr_horizons(M, a):
    """Compute Kerr black hole horizons"""
    # Outer (event) horizon
    r_plus = M + np.sqrt(M**2 - a**2)
    # Inner (Cauchy) horizon
    r_minus = M - np.sqrt(M**2 - a**2)
    return r_plus, r_minus

def kerr_ergosphere(M, a, theta):
    """Compute ergosphere boundary at angle theta"""
    return M + np.sqrt(M**2 - a**2 * np.cos(theta)**2)

def kerr_isco(M, a):
    """Compute ISCO radius (prograde orbit at equator)"""
    Z1 = 1 + (1 - a**2/M**2)**(1/3) * ((1 + a/M)**(1/3) + (1 - a/M)**(1/3))
    Z2 = np.sqrt(3*a**2/M**2 + Z1**2)
    r_isco = M * (3 + Z2 - np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))
    return r_isco

print("\n" + "=" * 60)
print("CRITICAL RADII vs SPIN")
print("=" * 60)

for a, label in zip(a_values, spin_labels):
    if a < M:
        r_plus, r_minus = kerr_horizons(M, a)
        r_ergo_eq = kerr_ergosphere(M, a, np.pi/2)  # At equator
        r_isco_val = kerr_isco(M, a)
        print(f"\n{label.replace(chr(10), ' ')}:")
        print(f"  Event horizon: r₊ = {r_plus:.3f}M")
        print(f"  Inner horizon: r₋ = {r_minus:.3f}M")
        print(f"  Ergosphere (eq): r_ergo = {r_ergo_eq:.3f}M")
        print(f"  ISCO (prograde): r_isco = {r_isco_val:.3f}M")

# Visualization
fig = plt.figure(figsize=(16, 10))

# Create subplots for different spin values
for idx, (a, label) in enumerate(zip(a_values, spin_labels)):
    ax = plt.subplot(2, 4, idx + 1)

    if a < M:
        # Horizons
        r_plus, r_minus = kerr_horizons(M, a)

        # Angular coordinate
        theta = np.linspace(0, np.pi, 100)

        # Event horizon (sphere)
        x_plus = r_plus * np.sin(theta)
        y_plus = r_plus * np.cos(theta)

        # Ergosphere boundary (oblate)
        r_ergo = np.array([kerr_ergosphere(M, a, th) for th in theta])
        x_ergo = r_ergo * np.sin(theta)
        y_ergo = r_ergo * np.cos(theta)

        # Plot
        ax.fill(x_plus, y_plus, color=COLORS['black'], alpha=0.9)
        ax.fill_between(x_ergo, y_ergo, x_plus, where=(x_ergo >= x_plus),
                        color=COLORS['orange'], alpha=0.3, label='Ergosphere')
        ax.fill_between(-x_ergo, y_ergo, -x_plus, where=(x_ergo >= x_plus),
                        color=COLORS['orange'], alpha=0.3)

        ax.plot(x_plus, y_plus, color=COLORS['red'], linewidth=2, label='Event horizon')
        ax.plot(-x_plus, y_plus, color=COLORS['red'], linewidth=2)
        ax.plot(x_ergo, y_ergo, color=COLORS['orange'], linewidth=2,
                linestyle='--', label='Ergosphere')
        ax.plot(-x_ergo, y_ergo, color=COLORS['orange'], linewidth=2, linestyle='--')

        # ISCO (prograde, at equator)
        r_isco_val = kerr_isco(M, a)
        circle_isco = plt.Circle((0, 0), r_isco_val, fill=False,
                                 color=COLORS['blue'], linewidth=1.5,
                                 linestyle=':', label='ISCO')
        ax.add_patch(circle_isco)

        # Rotation arrow
        ax.annotate('', xy=(0.3*M, 0.3*M), xytext=(-0.3*M, 0.3*M),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['green']))
        ax.text(0, 0.5*M, '↻', fontsize=16, ha='center', color=COLORS['green'])

    ax.set_xlim(-4*M, 4*M)
    ax.set_ylim(-4*M, 4*M)
    ax.set_aspect('equal')
    ax.set_xlabel('Equatorial plane', fontsize=9)
    ax.set_ylabel('Polar axis', fontsize=9)
    ax.set_title(f'{label}\na = {a}M', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

# Plot 5-8: Properties vs spin
ax5 = plt.subplot(2, 4, 5)
a_range = np.linspace(0, 0.998, 100)
r_plus_range = M + np.sqrt(M**2 - a_range**2)
r_minus_range = M - np.sqrt(M**2 - a_range**2)

ax5.plot(a_range/M, r_plus_range/M, color=COLORS['red'], linewidth=2,
         label='Event horizon r₊')
ax5.plot(a_range/M, r_minus_range/M, color=COLORS['purple'], linewidth=2,
         label='Inner horizon r₋')
ax5.axhline(y=2, color=COLORS['gray'], linestyle=':', alpha=0.5,
            label='r = 2M (Schwarzschild)')
ax5.set_xlabel('Spin a/M', fontsize=11)
ax5.set_ylabel('Horizon radius / M', fontsize=11)
ax5.set_title('Horizons vs Spin', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)
ax5.set_xlim(0, 1)

# ISCO vs spin
ax6 = plt.subplot(2, 4, 6)
r_isco_range = [kerr_isco(M, a) for a in a_range]
ax6.plot(a_range/M, np.array(r_isco_range)/M, color=COLORS['blue'], linewidth=2)
ax6.axhline(y=6, color=COLORS['gray'], linestyle=':', alpha=0.5,
            label='r = 6M (Schwarzschild)')
ax6.set_xlabel('Spin a/M', fontsize=11)
ax6.set_ylabel('ISCO radius / M', fontsize=11)
ax6.set_title('ISCO vs Spin (Prograde)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=9)
ax6.set_xlim(0, 1)
ax6.text(0.5, 3, 'Closer ISCO\n→ More energy\nextracted',
         fontsize=9, color=COLORS['blue'], ha='center')

# Frame dragging angular velocity
ax7 = plt.subplot(2, 4, 7)
r_range = np.linspace(1.5*M, 20*M, 100)
# Frame dragging: ω = 2Mar/(r³ + a²r + 2Ma²) at equator
omega_drag_slow = 2*M*0.5*M*r_range / (r_range**3 + (0.5*M)**2 * r_range + 2*M*(0.5*M)**2)
omega_drag_fast = 2*M*0.9*M*r_range / (r_range**3 + (0.9*M)**2 * r_range + 2*M*(0.9*M)**2)

ax7.plot(r_range/M, omega_drag_slow, color=COLORS['orange'], linewidth=2,
         label='a = 0.5M')
ax7.plot(r_range/M, omega_drag_fast, color=COLORS['red'], linewidth=2,
         label='a = 0.9M')
ax7.set_xlabel('Radius r/M', fontsize=11)
ax7.set_ylabel('Frame dragging ω', fontsize=11)
ax7.set_title('Frame Dragging Angular Velocity', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend(fontsize=9)
ax7.set_xlim(1.5, 20)

# Efficiency of energy extraction
ax8 = plt.subplot(2, 4, 8)
# Maximum efficiency from accretion disk: η = 1 - E_ISCO
# where E_ISCO is specific energy at ISCO
efficiency = []
for a in a_range:
    r_isco_val = kerr_isco(M, a)
    # Specific energy at ISCO (prograde, equatorial)
    # Approximate formula
    E_isco = np.sqrt(1 - 2*M/(3*r_isco_val))
    eta = 1 - E_isco
    efficiency.append(eta)

ax8.plot(a_range/M, np.array(efficiency)*100, color=COLORS['green'], linewidth=2)
ax8.axhline(y=5.7, color=COLORS['gray'], linestyle=':', alpha=0.5,
            label='η = 5.7% (Schwarzschild)')
ax8.set_xlabel('Spin a/M', fontsize=11)
ax8.set_ylabel('Max efficiency η (%)', fontsize=11)
ax8.set_title('Energy Extraction Efficiency', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=9)
ax8.set_xlim(0, 1)
ax8.text(0.5, 35, 'Faster spin\n→ More efficient!',
         fontsize=9, color=COLORS['green'], ha='center')

plt.tight_layout()
plt.suptitle('Kerr Black Holes: Rotation Effects',
             fontsize=14, fontweight='bold', y=0.995)
plt.show()

print("\n" + "=" * 60)
print("KEY FEATURES OF ROTATING BLACK HOLES")
print("=" * 60)
print("• TWO horizons: r₊ (event), r₋ (inner/Cauchy)")
print("  - Merge to r = M for extremal (a = M)")
print("\n• ERGOSPHERE: Region where nothing can stay still")
print("  - Between r₊ and r_ergo = M + √(M² - a²cos²θ)")
print("  - Frame dragging forces rotation")
print("  - Penrose process: Extract energy!")
print("\n• ISCO decreases with spin:")
print("  - Schwarzschild: r_isco = 6M")
print("  - Extremal: r_isco = M (prograde)")
print("  - Closer orbits → more energy extraction")
print("\n• FRAME DRAGGING: Spacetime rotates with black hole")
print("  - Stronger near horizon")
print("  - Measured by Gravity Probe B (Earth)")
print("\n• Energy extraction efficiency:")
print("  - Schwarzschild: η ≈ 5.7%")
print("  - Extremal Kerr: η ≈ 42% (prograde)")
print("=" * 60)
