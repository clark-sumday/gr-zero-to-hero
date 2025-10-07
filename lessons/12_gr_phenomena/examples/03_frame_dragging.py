#!/usr/bin/env python3
"""
Example: Frame Dragging (Lense-Thirring Effect)
Demonstrates how rotating masses drag spacetime
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("FRAME DRAGGING (LENSE-THIRRING EFFECT)")
print("=" * 60)
print("\nRotating masses drag spacetime around them")
print("Predicted by Lense and Thirring (1918)")
print("Confirmed by Gravity Probe B (2011)")

# Physical constants
G = 6.674e-11  # m^3 kg^-1 s^-2
c = 2.998e8    # m/s

# Earth parameters
M_earth = 5.972e24  # kg
R_earth = 6.371e6   # m
I_earth = 0.33 * M_earth * R_earth**2  # Moment of inertia (solid sphere approx)
T_earth = 86400     # s (rotation period)
omega_earth = 2 * np.pi / T_earth  # Angular velocity
J_earth = I_earth * omega_earth  # Angular momentum

print(f"\n" + "=" * 60)
print("EARTH'S ROTATION")
print("=" * 60)
print(f"Mass: M = {M_earth:.3e} kg")
print(f"Radius: R = {R_earth/1000:.0f} km")
print(f"Angular velocity: ω = {omega_earth:.3e} rad/s")
print(f"Angular momentum: J = {J_earth:.3e} kg·m²/s")

# Frame dragging angular velocity for test particle
# Ω_LT = 2GJ/(c²r³) at equator
# where J is angular momentum of central body

def frame_dragging_rate(r, J=J_earth, G=G, c=c):
    """
    Lense-Thirring precession rate (rad/s) at radius r.
    For equatorial orbit.
    """
    return 2 * G * J / (c**2 * r**3)

# Gravity Probe B orbit
h_gpb = 642e3  # m (altitude)
r_gpb = R_earth + h_gpb

omega_LT_gpb = frame_dragging_rate(r_gpb)

# Convert to arcseconds per year
arcsec_per_rad = 206265
seconds_per_year = 365.25 * 86400
omega_LT_arcsec_per_year = omega_LT_gpb * arcsec_per_rad * seconds_per_year

print(f"\n" + "=" * 60)
print("GRAVITY PROBE B MEASUREMENT")
print("=" * 60)
print(f"Orbital altitude: h = {h_gpb/1000:.0f} km")
print(f"Orbital radius: r = {r_gpb/1000:.0f} km")
print(f"Frame dragging rate: Ω_LT = {omega_LT_gpb:.3e} rad/s")
print(f"                          = {omega_LT_arcsec_per_year:.3f} arcsec/year")
print(f"\nGravity Probe B measured: ~39 milliarcsec/year")
print(f"Agreement with GR: < 1% error!")

# Radial dependence
altitudes = np.linspace(100e3, 40000e3, 1000)  # 100 km to 40,000 km
radii = R_earth + altitudes
omega_LT = frame_dragging_rate(radii)
omega_LT_arcsec_year = omega_LT * arcsec_per_rad * seconds_per_year

# Kerr black hole frame dragging (for comparison)
M_bh = 10 * 1.989e30  # 10 solar masses
a_bh = 0.9 * M_bh * G / c**2  # High spin parameter (geometric units)
r_s_bh = 2 * G * M_bh / c**2  # Schwarzschild radius

def kerr_frame_dragging(r, M, a, G=G, c=c):
    """
    Frame dragging rate for Kerr black hole (equatorial).
    ω = 2Mar/(r³ + a²r + 2Ma²) in geometric units
    """
    r_g = G * M / c**2  # Geometric mass
    a_g = a  # Already in geometric units
    omega = 2 * r_g * a_g * r / (r**3 + a_g**2 * r + 2 * r_g * a_g**2)
    return omega * c**3 / (G * M)  # Convert back to SI

r_bh = np.linspace(2.1*r_s_bh, 50*r_s_bh, 1000)
omega_kerr = kerr_frame_dragging(r_bh, M_bh, a_bh)

print(f"\n" + "=" * 60)
print("KERR BLACK HOLE (a=0.9M)")
print("=" * 60)
print(f"Mass: M = 10 M☉")
print(f"Schwarzschild radius: r_s = {r_s_bh/1000:.1f} km")
print(f"Spin parameter: a = 0.9 M")
print(f"\nAt r = 3r_s:")
idx_3rs = np.argmin(np.abs(r_bh - 3*r_s_bh))
print(f"  Frame dragging: Ω = {omega_kerr[idx_3rs]:.2e} rad/s")
print(f"  (Much stronger than Earth!)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Frame dragging vs altitude (Earth)
ax1 = axes[0, 0]
ax1.semilogy(altitudes/1000, omega_LT_arcsec_year, color=COLORS['blue'], linewidth=2)
ax1.axvline(x=h_gpb/1000, color=COLORS['red'], linestyle='--', linewidth=2,
            alpha=0.5, label=f'Gravity Probe B ({h_gpb/1000:.0f} km)')
ax1.axhline(y=omega_LT_arcsec_per_year, color=COLORS['red'], linestyle=':',
            alpha=0.5, label=f'{omega_LT_arcsec_per_year:.1f} mas/yr')
ax1.set_xlabel('Altitude (km)', fontsize=11)
ax1.set_ylabel('Frame dragging (arcsec/year)', fontsize=11)
ax1.set_title('Lense-Thirring Effect (Earth)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=9)
ax1.text(10000, 0.1, 'Ω ∝ 1/r³', fontsize=10, color=COLORS['blue'])

# Plot 2: Comparison Earth vs Kerr BH
ax2 = axes[0, 1]
ax2.loglog(radii/R_earth, omega_LT, color=COLORS['blue'], linewidth=2, label='Earth')
ax2.loglog(r_bh/r_s_bh, omega_kerr, color=COLORS['purple'], linewidth=2,
           label='Kerr BH (10M☉, a=0.9M)')
ax2.set_xlabel('Radius (r/R_earth or r/r_s)', fontsize=11)
ax2.set_ylabel('Frame dragging Ω (rad/s)', fontsize=11)
ax2.set_title('Frame Dragging: Earth vs Black Hole', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=9)

# Plot 3: Gyroscope precession visualization
ax3 = axes[1, 0]

# Show gyroscope precession over time
time_years = np.linspace(0, 5, 100)
precession_angle = omega_LT_arcsec_per_year * time_years / 1000  # Convert to degrees

# Draw gyroscope orientation
for t_idx in [0, 25, 50, 75, 99]:
    angle = np.radians(precession_angle[t_idx])
    # Gyroscope axis
    ax3.arrow(0, 0, 0.8*np.cos(angle), 0.8*np.sin(angle),
              head_width=0.1, head_length=0.1,
              fc=COLORS['blue'], ec=COLORS['blue'],
              alpha=0.3 + 0.7*t_idx/99, linewidth=2)
    if t_idx == 99:
        ax3.text(0.9*np.cos(angle), 0.9*np.sin(angle),
                f' {time_years[t_idx]:.0f} yr', fontsize=9, color=COLORS['blue'])

# Earth rotation axis
ax3.arrow(0, 0, 0, 1.2, head_width=0.15, head_length=0.1,
          fc=COLORS['green'], ec=COLORS['green'], linewidth=3)
ax3.text(0.05, 1.3, 'Earth rotation', fontsize=10, color=COLORS['green'])

ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-0.5, 1.5)
ax3.set_aspect('equal')
ax3.set_title('Gyroscope Precession (Gravity Probe B)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.text(0, -0.3, 'Precession: ~39 mas/year', fontsize=10,
         ha='center', color=COLORS['blue'])

# Plot 4: Spacetime dragging visualization
ax4 = axes[1, 1]

# Draw rotating Earth
earth_circle = plt.Circle((0, 0), 1, color=COLORS['blue'], alpha=0.3)
ax4.add_patch(earth_circle)

# Rotation arrow
ax4.annotate('', xy=(0.5, 0.5), xytext=(-0.5, 0.5),
            arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['green']))
ax4.text(0, 0.7, '↻ Rotation', fontsize=12, ha='center', color=COLORS['green'],
         fontweight='bold')

# Frame dragging field lines
r_field = np.array([1.5, 2.0, 2.5, 3.0])
for r in r_field:
    theta = np.linspace(0, 2*np.pi, 100)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax4.plot(x, y, color=COLORS['orange'], linewidth=1.5, alpha=0.6)

    # Add arrows to show dragging direction
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        x_arr = r * np.cos(angle)
        y_arr = r * np.sin(angle)
        dx = -0.2 * np.sin(angle)
        dy = 0.2 * np.cos(angle)
        ax4.arrow(x_arr, y_arr, dx, dy, head_width=0.15, head_length=0.1,
                 fc=COLORS['orange'], ec=COLORS['orange'], alpha=0.7)

ax4.set_xlim(-4, 4)
ax4.set_ylim(-4, 4)
ax4.set_aspect('equal')
ax4.set_title('Frame Dragging: Spacetime Rotation', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.text(0, -3.5, 'Rotating mass drags spacetime', fontsize=10,
         ha='center', color=COLORS['orange'])

plt.tight_layout()
plt.show()

# Satellites affected by frame dragging
print(f"\n" + "=" * 60)
print("FRAME DRAGGING FOR DIFFERENT SATELLITES")
print("=" * 60)

satellite_data = [
    ("Gravity Probe B", 642e3),
    ("LAGEOS", 5900e3),
    ("GPS", 20200e3),
    ("GEO", 35786e3)
]

for name, altitude in satellite_data:
    r = R_earth + altitude
    omega = frame_dragging_rate(r)
    omega_arcsec = omega * arcsec_per_rad * seconds_per_year
    print(f"\n{name} ({altitude/1000:.0f} km):")
    print(f"  Precession rate: {omega_arcsec:.2f} arcsec/year")
    print(f"                 = {omega_arcsec*1000:.1f} milliarcsec/year")

print(f"\n" + "=" * 60)
print("GEODETIC EFFECT (BONUS)")
print("=" * 60)
print("\nIn addition to frame dragging, gyroscopes also precess due to")
print("GEODETIC EFFECT (motion through curved spacetime)")
print(f"\nFor Gravity Probe B:")
print(f"  Geodetic precession: ~6600 milliarcsec/year")
print(f"  Frame dragging: ~39 milliarcsec/year")
print(f"\n  Geodetic effect is ~170× larger!")
print(f"  But both were measured by GPB")

print(f"\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("• Frame dragging: rotating mass drags spacetime")
print("• Effect is tiny for Earth (~39 mas/yr)")
print("• Much stronger near rotating black holes")
print("• Measured by Gravity Probe B (2004-2011)")
print("• Affects satellite orbits (LAGEOS, GPS)")
print("• Key prediction of General Relativity")
print("• Evidence for spacetime as a physical entity")
print("=" * 60)
