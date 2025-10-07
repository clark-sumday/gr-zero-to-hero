#!/usr/bin/env python3
"""
Example: Schwarzschild Metric and Coordinate Systems
Visualizes metric components and compares different coordinate systems
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("SCHWARZSCHILD METRIC")
print("=" * 60)
print("\nThe first exact solution to Einstein's equations (1916)")
print("\nMetric (Schwarzschild coordinates):")
print("  ds² = -(1-r_s/r)c²dt² + (1-r_s/r)⁻¹dr² + r²(dθ² + sin²θ dφ²)")
print(f"\nwhere r_s = 2GM/c² is the Schwarzschild radius")

# Physical constants (SI units)
G = 6.674e-11  # m^3 kg^-1 s^-2
c = 2.998e8    # m/s

# Example: Solar mass black hole
M_sun = 1.989e30  # kg
r_s_sun = 2 * G * M_sun / c**2

print(f"\nFor a solar mass black hole:")
print(f"  M = {M_sun:.3e} kg")
print(f"  r_s = {r_s_sun:.2f} m = {r_s_sun/1000:.2f} km")

# Use geometric units (G=c=1) for simplicity
M = 1.0
r_s = 2 * M

print(f"\nGeometric units (G=c=1, M={M}):")
print(f"  r_s = {r_s}")

# Radial coordinate array
r = np.linspace(r_s * 1.01, 20*M, 1000)

# Metric components
g_tt = -(1 - r_s/r)                # Time-time component
g_rr = 1 / (1 - r_s/r)             # Radial-radial component
g_theta_theta = r**2               # Angular component
g_phi_phi = r**2 * np.sin(np.pi/2)**2  # Azimuthal (at equator)

# Proper distance from r to infinity
# Proper distance: dl = √(g_rr) dr
# For Schwarzschild: dl = dr/√(1-r_s/r)
def proper_distance_to_infinity(r_val):
    """Compute proper radial distance from r to infinity"""
    if r_val <= r_s:
        return np.inf
    # Analytical result: l = r + r_s ln((r-r_s)/r_s)
    return r_val - r_s + r_s * np.log((r_val - r_s) / r_s)

proper_dist = np.array([proper_distance_to_infinity(r_val) for r_val in r])

# Time dilation factor
time_dilation = np.sqrt(-g_tt)  # dt_proper / dt_coordinate

print("\n" + "=" * 60)
print("METRIC COMPONENTS AT KEY RADII")
print("=" * 60)

key_radii = [r_s * 1.01, 2*r_s, 3*r_s, 6*r_s, 10*r_s]
for r_val in key_radii:
    idx = np.argmin(np.abs(r - r_val))
    print(f"\nr = {r_val/r_s:.2f} r_s:")
    print(f"  g_tt = {g_tt[idx]:+.6f}")
    print(f"  g_rr = {g_rr[idx]:+.6f}")
    print(f"  Time dilation = {time_dilation[idx]:.6f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Metric components
ax1 = axes[0, 0]
ax1.plot(r/r_s, g_tt, color=COLORS['blue'], linewidth=2, label='g_tt')
ax1.plot(r/r_s, g_rr, color=COLORS['orange'], linewidth=2, label='g_rr')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axhline(y=-1, color=COLORS['gray'], linestyle=':', alpha=0.5)
ax1.axhline(y=1, color=COLORS['gray'], linestyle=':', alpha=0.5)
ax1.axvline(x=1, color=COLORS['red'], linestyle='--', linewidth=2,
            alpha=0.5, label='Horizon (r=r_s)')
ax1.set_xlabel('r / r_s', fontsize=11)
ax1.set_ylabel('Metric Component', fontsize=11)
ax1.set_title('Schwarzschild Metric Components', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xlim(1, 10)
ax1.set_ylim(-2, 10)

# Plot 2: Time dilation
ax2 = axes[0, 1]
ax2.plot(r/r_s, time_dilation, color=COLORS['green'], linewidth=2)
ax2.axvline(x=1, color=COLORS['red'], linestyle='--', linewidth=2,
            alpha=0.5, label='Horizon')
ax2.axhline(y=1, color=COLORS['gray'], linestyle=':', alpha=0.5,
            label='No dilation (r→∞)')
ax2.set_xlabel('r / r_s', fontsize=11)
ax2.set_ylabel('√(-g_tt) = dt_proper/dt_coord', fontsize=11)
ax2.set_title('Gravitational Time Dilation', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim(1, 10)
ax2.text(2, 0.5, 'Time slows\nnear horizon',
         fontsize=10, color=COLORS['green'], ha='center')

# Plot 3: Proper distance vs coordinate distance
ax3 = axes[1, 0]
ax3.plot(r/r_s, proper_dist/r_s, color=COLORS['purple'], linewidth=2,
         label='Proper distance to ∞')
ax3.plot(r/r_s, r/r_s, color=COLORS['gray'], linestyle='--', linewidth=1,
         label='Coordinate distance', alpha=0.7)
ax3.axvline(x=1, color=COLORS['red'], linestyle='--', linewidth=2,
            alpha=0.5)
ax3.set_xlabel('r / r_s', fontsize=11)
ax3.set_ylabel('Distance / r_s', fontsize=11)
ax3.set_title('Coordinate vs Proper Distance', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_xlim(1, 10)
ax3.text(2, 10, 'Proper distance > coordinate distance\n(space is stretched)',
         fontsize=9, color=COLORS['purple'], ha='left')

# Plot 4: Embedding diagram concept
ax4 = axes[1, 1]
# Create embedding diagram for Schwarzschild geometry
# At constant t, θ=π/2: ds² = (1-r_s/r)⁻¹ dr² + r² dφ²
theta_embed = np.linspace(0, 2*np.pi, 100)
r_embed_vals = [2.1*r_s, 3*r_s, 5*r_s, 8*r_s]

for r_emb in r_embed_vals:
    x_emb = r_emb * np.cos(theta_embed)
    y_emb = r_emb * np.sin(theta_embed)
    ax4.plot(x_emb/r_s, y_emb/r_s, linewidth=2, alpha=0.7)

# Event horizon
x_h = r_s * np.cos(theta_embed)
y_h = r_s * np.sin(theta_embed)
ax4.fill(x_h/r_s, y_h/r_s, color=COLORS['black'], alpha=0.8)
ax4.plot(x_h/r_s, y_h/r_s, color=COLORS['red'], linewidth=2,
         linestyle='--', label='Event horizon')

ax4.set_xlabel('x / r_s', fontsize=11)
ax4.set_ylabel('y / r_s', fontsize=11)
ax4.set_title('Schwarzschild Geometry (Equatorial Slice)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')
ax4.legend(fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("KEY PROPERTIES")
print("=" * 60)
print("• Static: Independent of time t")
print("• Spherically symmetric: Only depends on r")
print("• Asymptotically flat: g_μν → η_μν as r → ∞")
print("• Vacuum solution: R_μν = 0")
print("• Event horizon at r = r_s (coordinate singularity)")
print("• True singularity at r = 0 (curvature → ∞)")
print("• Unique by Birkhoff's theorem")
print("=" * 60)
