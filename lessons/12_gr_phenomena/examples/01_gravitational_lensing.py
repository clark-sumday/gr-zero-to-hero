#!/usr/bin/env python3
"""
Example: Gravitational Lensing
Demonstrates light bending and Einstein ring formation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("GRAVITATIONAL LENSING")
print("=" * 60)
print("\nLight bends in curved spacetime")
print("First confirmed in 1919 solar eclipse (Eddington)")

# Constants (SI units)
G = 6.674e-11  # m^3 kg^-1 s^-2
c = 2.998e8    # m/s
M_sun = 1.989e30  # kg

# Deflection angle formula: α = 4GM/(c²b)
# where b is impact parameter (closest approach)

def deflection_angle(M, b, G=6.674e-11, c=2.998e8):
    """
    Light deflection angle (in radians) for point mass M and impact parameter b.
    """
    return 4 * G * M / (c**2 * b)

# Example 1: Sun
R_sun = 6.96e8  # m (solar radius)
alpha_sun = deflection_angle(M_sun, R_sun)
alpha_sun_arcsec = alpha_sun * 206265  # Convert to arcseconds

print(f"\n" + "=" * 60)
print("LIGHT DEFLECTION BY THE SUN")
print("=" * 60)
print(f"Solar mass: M = {M_sun:.3e} kg")
print(f"Solar radius: R = {R_sun:.3e} m")
print(f"Deflection angle at limb: α = {alpha_sun_arcsec:.2f} arcsec")
print(f"Einstein's prediction (1915): 1.75 arcsec")
print(f"Newtonian prediction: 0.875 arcsec (factor of 2 difference!)")

# Impact parameter range
b = np.logspace(np.log10(R_sun), np.log10(100*R_sun), 1000)
alpha = deflection_angle(M_sun, b)

# Example 2: Galaxy cluster (strong lensing)
M_cluster = 1e15 * M_sun  # Typical massive cluster
D_L = 1e9 * 9.461e15  # 1 Gpc in meters (lens distance)
D_S = 2e9 * 9.461e15  # 2 Gpc (source distance)
D_LS = D_S - D_L

# Einstein radius: θ_E = √(4GM/c² × D_LS/(D_L × D_S))
theta_E = np.sqrt(4*G*M_cluster/c**2 * D_LS/(D_L*D_S))
theta_E_arcsec = theta_E * 206265

print(f"\n" + "=" * 60)
print("GALAXY CLUSTER LENSING")
print("=" * 60)
print(f"Cluster mass: M = {M_cluster/M_sun:.2e} M☉")
print(f"Lens distance: D_L = {D_L/(9.461e15*1e9):.1f} Gpc")
print(f"Source distance: D_S = {D_S/(9.461e15*1e9):.1f} Gpc")
print(f"Einstein radius: θ_E = {theta_E_arcsec:.1f} arcsec")

# Visualization 1: Deflection angle vs impact parameter
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.loglog(b/R_sun, alpha * 206265, color=COLORS['blue'], linewidth=2)
ax1.axvline(x=1, color=COLORS['red'], linestyle='--', linewidth=2,
            alpha=0.5, label='Solar limb')
ax1.axhline(y=alpha_sun_arcsec, color=COLORS['gray'], linestyle=':',
            alpha=0.5, label=f'{alpha_sun_arcsec:.2f} arcsec')
ax1.set_xlabel('Impact parameter b / R☉', fontsize=11)
ax1.set_ylabel('Deflection angle α (arcsec)', fontsize=11)
ax1.set_title('Light Deflection by the Sun', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=10)
ax1.text(5, 0.5, 'α ∝ 1/b', fontsize=10, color=COLORS['blue'])

# Visualization 2: Lensing geometry
ax2 = axes[0, 1]

# Simple lensing setup
lens_x = 0
source_x_true = 0.5  # True angular position
source_x_apparent = 1.5  # Apparent position (lensed)

# Draw geometry
ax2.plot([0], [0], 'o', color=COLORS['orange'], markersize=15, label='Lens (galaxy)')
ax2.plot([source_x_true], [3], '*', color=COLORS['blue'], markersize=20,
         label='True source position')
ax2.plot([source_x_apparent], [3], '*', color=COLORS['red'], markersize=20,
         label='Apparent (lensed) image')

# Light rays
ax2.plot([0, source_x_apparent], [0, 3], color=COLORS['red'], linewidth=2,
         linestyle='--', alpha=0.7, label='Bent light path')
ax2.plot([0, source_x_true], [0, 3], color=COLORS['gray'], linewidth=1,
         linestyle=':', alpha=0.5, label='Straight line')

# Observer
ax2.plot([0], [-1], 's', color=COLORS['green'], markersize=12, label='Observer')
ax2.plot([0, 0], [-1, 0], 'k-', linewidth=2)

ax2.set_xlim(-2, 3)
ax2.set_ylim(-1.5, 3.5)
ax2.set_xlabel('Angular position (θ)', fontsize=11)
ax2.set_ylabel('Distance', fontsize=11)
ax2.set_title('Gravitational Lensing Geometry', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)

# Visualization 3: Einstein ring
ax3 = axes[1, 0]

# When source is perfectly aligned, forms a ring
theta_ring = np.linspace(0, 2*np.pi, 100)
r_einstein = 1.0  # Normalized Einstein radius

x_ring = r_einstein * np.cos(theta_ring)
y_ring = r_einstein * np.sin(theta_ring)

# Central lens
ax3.plot([0], [0], 'o', color=COLORS['orange'], markersize=20, label='Lens')

# Einstein ring
ax3.plot(x_ring, y_ring, color=COLORS['blue'], linewidth=3, label='Einstein ring')
ax3.fill(x_ring, y_ring, color=COLORS['blue'], alpha=0.2)

# Source (behind lens, perfectly aligned)
ax3.plot([0], [0], '*', color=COLORS['red'], markersize=15,
         label='Background source\n(aligned)', zorder=10)

ax3.set_xlim(-2, 2)
ax3.set_ylim(-2, 2)
ax3.set_aspect('equal')
ax3.set_xlabel('θ_x', fontsize=11)
ax3.set_ylabel('θ_y', fontsize=11)
ax3.set_title('Einstein Ring (Perfect Alignment)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)

# Visualization 4: Multiple images
ax4 = axes[1, 1]

# Slightly misaligned source creates multiple images
source_offset = 0.3

# Lens
ax4.plot([0], [0], 'o', color=COLORS['orange'], markersize=20, label='Lens')

# Arc (partial ring)
theta_arc = np.linspace(-0.8*np.pi, 0.8*np.pi, 100)
x_arc = r_einstein * np.cos(theta_arc)
y_arc = r_einstein * np.sin(theta_arc) + source_offset

ax4.plot(x_arc, y_arc, color=COLORS['blue'], linewidth=3, label='Lensed arc')

# True source position
ax4.plot([0], [source_offset], '*', color=COLORS['red'], markersize=15,
         label='True source', zorder=10)

# Additional images (for strong lensing)
ax4.plot([0.7], [-0.4], 's', color=COLORS['cyan'], markersize=10,
         label='Secondary image')
ax4.plot([-0.7], [-0.4], 's', color=COLORS['cyan'], markersize=10)

ax4.set_xlim(-2, 2)
ax4.set_ylim(-2, 2)
ax4.set_aspect('equal')
ax4.set_xlabel('θ_x', fontsize=11)
ax4.set_ylabel('θ_y', fontsize=11)
ax4.set_title('Multiple Images (Slight Misalignment)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

# Magnification
print(f"\n" + "=" * 60)
print("MAGNIFICATION")
print("=" * 60)
print("\nMagnification μ = (area of image) / (area of source)")
print("\nFor perfect alignment (Einstein ring):")
print("  μ → ∞ (infinite magnification!)")
print("\nFor slight offset β from alignment:")
print("  μ ≈ (θ_E/β)² for β << θ_E")

# Example magnifications
beta_range = np.logspace(-2, 0, 100)  # Source position (in units of θ_E)
mu = 1 / beta_range**2  # Simplified magnification

fig2, ax = plt.subplots(figsize=(10, 6))
ax.loglog(beta_range, mu, color=COLORS['purple'], linewidth=2)
ax.axhline(y=1, color=COLORS['gray'], linestyle='--', alpha=0.5,
           label='No magnification')
ax.set_xlabel('Source offset β / θ_E', fontsize=11)
ax.set_ylabel('Magnification μ', fontsize=11)
ax.set_title('Gravitational Lensing Magnification', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10)
ax.text(0.1, 50, 'μ ∝ 1/β²\n(closer to alignment\n→ brighter)',
        fontsize=10, color=COLORS['purple'], ha='center')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("TYPES OF LENSING")
print("=" * 60)
print("\n1. STRONG LENSING: Multiple images, arcs, Einstein rings")
print("   • Massive galaxy clusters")
print("   • Individual galaxies")
print("   • Observable distortion")
print("\n2. WEAK LENSING: Statistical shape distortions")
print("   • Used to map dark matter")
print("   • Requires analysis of many galaxies")
print("\n3. MICROLENSING: Temporary brightness changes")
print("   • Stars, planets, black holes")
print("   • No resolved images (too small)")
print("   • Time-domain astronomy")

print("\n" + "=" * 60)
print("KEY APPLICATIONS")
print("=" * 60)
print("• Measuring masses of galaxies and clusters")
print("• Mapping dark matter distribution")
print("• Magnifying distant galaxies (cosmic telescope)")
print("• Detecting exoplanets (microlensing)")
print("• Testing General Relativity")
print("• Measuring Hubble constant (time delays)")
print("=" * 60)
