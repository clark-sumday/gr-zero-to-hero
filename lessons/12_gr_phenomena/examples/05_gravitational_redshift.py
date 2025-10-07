#!/usr/bin/env python3
"""
Example: Gravitational Redshift
Demonstrates frequency/wavelength shift in gravitational fields
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("GRAVITATIONAL REDSHIFT")
print("=" * 60)
print("\nLight loses energy climbing out of gravitational field")
print("First measured by Pound-Rebka experiment (1959)")

# Physical constants
c = 2.998e8      # m/s
G = 6.674e-11    # m^3 kg^-1 s^-2
M_sun = 1.989e30  # kg
R_sun = 6.96e8    # m
M_earth = 5.972e24  # kg
R_earth = 6.371e6   # m

# Gravitational redshift formula (weak field)
# z = Δf/f = -(Δφ/c²) = (GM/c²)(1/r_emit - 1/r_obs)
# where φ is gravitational potential

def gravitational_redshift(M, r_emit, r_obs, G=G, c=c):
    """
    Gravitational redshift parameter z.
    Positive z = redshift (frequency decreases)
    Negative z = blueshift (frequency increases)
    """
    return (G * M / c**2) * (1/r_emit - 1/r_obs)

# Example 1: Pound-Rebka experiment (Earth)
h_tower = 22.5  # m (height of Harvard tower)
z_pound_rebka = gravitational_redshift(M_earth, R_earth, R_earth + h_tower)

print(f"\n" + "=" * 60)
print("POUND-REBKA EXPERIMENT (1959)")
print("=" * 60)
print(f"Location: Harvard University")
print(f"Height: h = {h_tower} m")
print(f"Redshift: z = {z_pound_rebka:.3e}")
print(f"Frequency shift: Δf/f = {z_pound_rebka:.3e}")
print(f"\nPhoton climbing 22.5 m loses:")
print(f"  Relative energy: {abs(z_pound_rebka)*100:.2e}%")
print(f"\nMeasured to ~1% precision - confirmed GR!")

# Example 2: GPS satellites
h_gps = 20200e3  # m
r_gps = R_earth + h_gps
z_gps = gravitational_redshift(M_earth, R_earth, r_gps)

print(f"\n" + "=" * 60)
print("GPS SATELLITES")
print("=" * 60)
print(f"Altitude: h = {h_gps/1000:.0f} km")
print(f"Redshift: z = {z_gps:.3e}")
print(f"\nPhotons from GPS are BLUESHIFTED (z < 0)")
print(f"  → Higher frequency at GPS altitude")
print(f"  → Weaker gravity = higher energy")

# Example 3: Solar surface
z_sun = gravitational_redshift(M_sun, R_sun, np.inf)  # To infinity

print(f"\n" + "=" * 60)
print("SOLAR SURFACE")
print("=" * 60)
print(f"Solar mass: M = {M_sun:.3e} kg")
print(f"Solar radius: R = {R_sun:.3e} m")
print(f"Redshift (to infinity): z = {z_sun:.3e}")
print(f"  ≈ {z_sun*1e6:.1f} parts per million")

# Example 4: White dwarf (Sirius B)
M_sirius_B = 1.02 * M_sun
R_sirius_B = 5800e3  # m (Earth-sized!)
z_sirius_B = gravitational_redshift(M_sirius_B, R_sirius_B, np.inf)

print(f"\n" + "=" * 60)
print("WHITE DWARF (SIRIUS B)")
print("=" * 60)
print(f"Mass: M = {M_sirius_B/M_sun:.2f} M☉")
print(f"Radius: R = {R_sirius_B/1000:.0f} km (Earth-sized)")
print(f"Redshift: z = {z_sirius_B:.3e}")
print(f"  ≈ {z_sirius_B*1e6:.0f} ppm")
print(f"\n→ Much stronger than Sun (more compact)")

# Example 5: Neutron star
M_ns = 1.4 * M_sun
R_ns = 12e3  # m (12 km radius)
z_ns = gravitational_redshift(M_ns, R_ns, np.inf)

print(f"\n" + "=" * 60)
print("NEUTRON STAR")
print("=" * 60)
print(f"Mass: M = {M_ns/M_sun:.1f} M☉")
print(f"Radius: R = {R_ns/1000:.0f} km")
print(f"Redshift: z = {z_ns:.3f}")
print(f"\n→ Photons lose ~{z_ns*100:.0f}% of energy escaping!")

# Visualization 1: Redshift vs radius for different objects
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Redshift from Earth
ax1 = axes[0, 0]
heights = np.linspace(0, 1000e3, 1000)  # Up to 1000 km
radii_earth = R_earth + heights
z_earth = gravitational_redshift(M_earth, R_earth, radii_earth)

ax1.plot(heights/1000, z_earth, color=COLORS['blue'], linewidth=2)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=h_tower/1000, color=COLORS['red'], linestyle='--',
            linewidth=2, alpha=0.5, label=f'Pound-Rebka ({h_tower} m)')
ax1.axvline(x=h_gps/1000, color=COLORS['orange'], linestyle='--',
            linewidth=2, alpha=0.5, label=f'GPS ({h_gps/1000:.0f} km)')
ax1.set_xlabel('Height above surface (km)', fontsize=11)
ax1.set_ylabel('Gravitational redshift z', fontsize=11)
ax1.set_title('Redshift from Earth Surface', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.fill_between(heights/1000, z_earth, 0, where=(z_earth < 0),
                 alpha=0.2, color=COLORS['blue'], label='Blueshift region')

# Plot 2: Comparison of different objects
ax2 = axes[0, 1]

objects = ['Earth\nsurface', 'Sun\nsurface', 'Sirius B\n(white dwarf)',
           'Neutron\nstar']
redshifts = [
    gravitational_redshift(M_earth, R_earth, np.inf),
    z_sun,
    z_sirius_B,
    z_ns
]

colors_bars = [COLORS['blue'], COLORS['orange'], COLORS['purple'], COLORS['red']]
ax2.bar(range(len(objects)), redshifts, color=colors_bars)
ax2.set_ylabel('Redshift z', fontsize=11)
ax2.set_title('Gravitational Redshift (to infinity)', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(objects)))
ax2.set_xticklabels(objects, fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_yscale('log')

# Plot 3: Wavelength shift
ax3 = axes[1, 0]

# Hydrogen alpha line (656.3 nm in vacuum)
lambda_0 = 656.3  # nm

# Observed wavelengths from different objects
lambda_observed = lambda_0 * (1 + np.array(redshifts))

ax3.plot(range(len(objects)), lambda_observed, 'o-', color=COLORS['green'],
         markersize=10, linewidth=2)
ax3.axhline(y=lambda_0, color=COLORS['gray'], linestyle='--',
            alpha=0.5, label=f'Rest wavelength ({lambda_0} nm)')
ax3.set_ylabel('Observed wavelength (nm)', fontsize=11)
ax3.set_title('Spectral Line Shift (H-alpha)', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(objects)))
ax3.set_xticklabels(objects, fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

# Plot 4: Redshift vs compactness (M/R)
ax4 = axes[1, 1]

# Range of compactness (geometric units: GM/(Rc²))
M_range = np.logspace(24, 31, 100)  # kg
R_range = np.logspace(5, 9, 100)    # m

M_grid, R_grid = np.meshgrid(M_range, R_range)
compactness = G * M_grid / (c**2 * R_grid)
z_grid = G * M_grid / (c**2 * R_grid)

# Plot contours
levels = [1e-9, 1e-6, 1e-3, 0.1, 0.3, 0.5]
contour = ax4.contourf(M_grid/M_sun, R_grid/1000, z_grid,
                       levels=levels, cmap='viridis', alpha=0.7)
cbar = plt.colorbar(contour, ax=ax4, label='Redshift z')

# Mark specific objects
ax4.plot([M_sun/M_sun], [R_sun/1000], 'o', color='yellow',
         markersize=12, markeredgecolor='black', markeredgewidth=2, label='Sun')
ax4.plot([M_sirius_B/M_sun], [R_sirius_B/1000], 's', color='white',
         markersize=10, markeredgecolor='black', markeredgewidth=2, label='Sirius B')
ax4.plot([M_ns/M_sun], [R_ns/1000], '^', color='red',
         markersize=10, markeredgecolor='black', markeredgewidth=2, label='Neutron star')

# Schwarzschild radius line
R_schwarzschild = 2 * G * M_range / c**2
ax4.plot(M_range/M_sun, R_schwarzschild/1000, '--', color='red',
         linewidth=2, label='Black hole limit')

ax4.set_xlabel('Mass (M☉)', fontsize=11)
ax4.set_ylabel('Radius (km)', fontsize=11)
ax4.set_title('Redshift vs Compactness', fontsize=12, fontweight='bold')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

# Time dilation equivalent
print(f"\n" + "=" * 60)
print("GRAVITATIONAL TIME DILATION")
print("=" * 60)
print("\nRedshift is equivalent to time dilation:")
print("  z = Δf/f = -Δt/t")
print("\nClock at lower gravitational potential runs SLOWER")

# Neutron star example
t_ns = 1.0  # 1 second on neutron star
t_infinity = t_ns * (1 + z_ns)

print(f"\nNeutron star surface:")
print(f"  1 second on NS surface = {t_infinity:.3f} seconds at infinity")
print(f"  Time runs {(t_infinity-t_ns)/t_ns*100:.1f}% slower on surface")

# Extreme case: near black hole
M_bh = 10 * M_sun
r_s = 2 * G * M_bh / c**2  # Schwarzschild radius
r_near = 1.5 * r_s  # Close to horizon

z_near_bh = gravitational_redshift(M_bh, r_near, np.inf)
t_near_bh = 1.0 * (1 + z_near_bh)

print(f"\nNear black hole (r = 1.5 r_s):")
print(f"  1 second near BH = {t_near_bh:.1f} seconds at infinity")
print(f"  Time runs {(t_near_bh-1)/1*100:.0f}% slower!")

print(f"\n" + "=" * 60)
print("OBSERVATIONAL TESTS")
print("=" * 60)
print("\n1. POUND-REBKA (1959): Laboratory test")
print("   • 22.5 m tower")
print("   • Measured z ~ 10⁻¹⁵")
print("   • Confirmed to ~1%")
print("\n2. GPS: Daily application")
print("   • Needed for accurate positioning")
print("   • ~45 μs/day time difference")
print("\n3. SOLAR SPECTRUM: Astrophysical test")
print("   • Spectral line shifts")
print("   • z ~ 2×10⁻⁶")
print("\n4. WHITE DWARFS: Strong field test")
print("   • Sirius B: z ~ 3×10⁻⁴")
print("   • Confirmed by spectroscopy")
print("\n5. NEUTRON STARS: Extreme gravity")
print("   • z ~ 0.2-0.3")
print("   • Observed in X-ray binaries")

print(f"\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("• Photons lose energy climbing out of gravity well")
print("• Redshift z = ΔE/E = Δf/f = Δλ/λ")
print("• Equivalent to gravitational time dilation")
print("• Weak field: z ≈ GM/(Rc²)")
print("• Stronger for more compact objects")
print("• At horizon (r=r_s): z → ∞ (infinite redshift)")
print("• Tested from lab (10⁻¹⁵) to neutron stars (0.3)")
print("• Essential for GPS accuracy")
print("=" * 60)
