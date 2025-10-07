#!/usr/bin/env python3
"""
Example: Black Hole Properties
Explores event horizons, photon spheres, ISCO, and tidal forces
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("BLACK HOLE PROPERTIES")
print("=" * 60)

# Geometric units: G = c = 1
M = 1.0  # Black hole mass
r_s = 2 * M  # Schwarzschild radius (event horizon)
r_photon = 3 * M  # Photon sphere
r_isco = 6 * M  # Innermost stable circular orbit

print(f"\nBlack hole mass: M = {M}")
print(f"\nCritical radii:")
print(f"  • Event horizon: r_s = {r_s}M")
print(f"  • Photon sphere: r_ph = {r_photon}M")
print(f"  • ISCO: r_isco = {r_isco}M")

# Physical examples
masses_solar = np.array([1, 10, 1e6, 1e9])  # Solar masses
M_sun_kg = 1.989e30  # kg
G_SI = 6.674e-11
c_SI = 2.998e8

r_s_km = masses_solar * (2 * G_SI * M_sun_kg / c_SI**2) / 1000

print(f"\n" + "=" * 60)
print("SCHWARZSCHILD RADII FOR DIFFERENT MASSES")
print("=" * 60)
for i, M_ratio in enumerate(masses_solar):
    if M_ratio < 1e6:
        print(f"  {M_ratio:.0f} M☉: r_s = {r_s_km[i]:.2f} km")
    else:
        print(f"  {M_ratio:.0e} M☉: r_s = {r_s_km[i]:.2e} km")

# Orbital properties
r = np.linspace(r_s * 1.01, 20*M, 1000)

# Circular orbit angular velocity: Ω² = M/r³
omega_circular = np.sqrt(M / r**3)

# Orbital period: T = 2π/Ω
period = 2 * np.pi / omega_circular

# Specific orbital energy: E = √((r-2M)/r) / √(1-3M/r)
# Only defined for r > 3M (outside photon sphere)
r_orbit = r[r > 3*M]
E_orbit = np.sqrt((r_orbit - 2*M) / r_orbit) / np.sqrt(1 - 3*M/r_orbit)

# Specific angular momentum: L = √(Mr) / √(1-3M/r)
L_orbit = np.sqrt(M * r_orbit) / np.sqrt(1 - 3*M/r_orbit)

# Tidal forces (Riemann tensor component)
# R_trtθ ≈ M/r³ (for radial-tangential tidal stretching)
R_tidal = M / r**3

print(f"\n" + "=" * 60)
print("ORBITAL PROPERTIES AT KEY RADII")
print("=" * 60)

key_radii = [3*M, 6*M, 10*M, 20*M]
for r_val in key_radii:
    idx = np.argmin(np.abs(r - r_val))
    print(f"\nr = {r_val/M:.0f}M:")
    print(f"  Angular velocity Ω = {omega_circular[idx]:.4f} / M")
    print(f"  Period T = {period[idx]:.2f} M")
    print(f"  Tidal force ∝ {R_tidal[idx]:.4f}")
    if r_val >= 3*M:
        idx_orb = np.argmin(np.abs(r_orbit - r_val))
        print(f"  Orbital energy E = {E_orbit[idx_orb]:.4f}")
        print(f"  Angular momentum L = {L_orbit[idx_orb]:.4f} M")

# Escape velocity at different radii
# v_escape² = r_s/r (in geometric units where c=1)
v_escape = np.sqrt(r_s / r)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Critical radii visualization
ax1 = axes[0, 0]
theta = np.linspace(0, 2*np.pi, 100)

# Event horizon
x_h = r_s * np.cos(theta)
y_h = r_s * np.sin(theta)
ax1.fill(x_h, y_h, color=COLORS['black'], alpha=0.9, label='Black hole')
ax1.plot(x_h, y_h, color=COLORS['red'], linewidth=3, linestyle='-')

# Photon sphere
x_ph = r_photon * np.cos(theta)
y_ph = r_photon * np.sin(theta)
ax1.plot(x_ph, y_ph, color=COLORS['orange'], linewidth=2,
         linestyle='--', label='Photon sphere (3M)')

# ISCO
x_isco = r_isco * np.cos(theta)
y_isco = r_isco * np.sin(theta)
ax1.plot(x_isco, y_isco, color=COLORS['blue'], linewidth=2,
         linestyle=':', label='ISCO (6M)')

ax1.set_xlabel('x / M', fontsize=11)
ax1.set_ylabel('y / M', fontsize=11)
ax1.set_title('Critical Radii', fontsize=12, fontweight='bold')
ax1.set_aspect('equal')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.text(0, 0, 'Event\nHorizon', ha='center', va='center',
         fontsize=9, color='white', fontweight='bold')

# Plot 2: Orbital angular velocity
ax2 = axes[0, 1]
ax2.plot(r/M, omega_circular, color=COLORS['green'], linewidth=2)
ax2.axvline(x=r_s/M, color=COLORS['red'], linestyle='--', linewidth=2, alpha=0.5)
ax2.axvline(x=r_photon/M, color=COLORS['orange'], linestyle='--', linewidth=1, alpha=0.5)
ax2.axvline(x=r_isco/M, color=COLORS['blue'], linestyle=':', linewidth=1, alpha=0.5)
ax2.set_xlabel('r / M', fontsize=11)
ax2.set_ylabel('Angular velocity Ω (units of 1/M)', fontsize=11)
ax2.set_title('Orbital Angular Velocity', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(2, 20)

# Plot 3: Escape velocity
ax3 = axes[0, 2]
ax3.plot(r/M, v_escape, color=COLORS['purple'], linewidth=2)
ax3.axvline(x=r_s/M, color=COLORS['red'], linestyle='--', linewidth=2,
            alpha=0.5, label='Horizon')
ax3.axhline(y=1, color=COLORS['gray'], linestyle=':', alpha=0.5,
            label='v = c')
ax3.set_xlabel('r / M', fontsize=11)
ax3.set_ylabel('Escape velocity v/c', fontsize=11)
ax3.set_title('Escape Velocity', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)
ax3.set_xlim(2, 20)
ax3.set_ylim(0, 1.1)
ax3.text(4, 0.9, 'v_esc = c at horizon', fontsize=9, color=COLORS['purple'])

# Plot 4: Orbital energy
ax4 = axes[1, 0]
ax4.plot(r_orbit/M, E_orbit, color=COLORS['cyan'], linewidth=2)
ax4.axvline(x=r_photon/M, color=COLORS['orange'], linestyle='--',
            linewidth=2, alpha=0.5, label='Photon sphere')
ax4.axvline(x=r_isco/M, color=COLORS['blue'], linestyle=':',
            linewidth=2, alpha=0.5, label='ISCO')
ax4.axhline(y=1, color=COLORS['gray'], linestyle=':', alpha=0.5)
ax4.set_xlabel('r / M', fontsize=11)
ax4.set_ylabel('Specific energy E', fontsize=11)
ax4.set_title('Orbital Energy (Circular Orbits)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9)
ax4.set_ylim(0.8, 1.05)

# Plot 5: Angular momentum
ax5 = axes[1, 1]
ax5.plot(r_orbit/M, L_orbit/M, color=COLORS['pink'], linewidth=2)
ax5.axvline(x=r_photon/M, color=COLORS['orange'], linestyle='--',
            linewidth=2, alpha=0.5)
ax5.axvline(x=r_isco/M, color=COLORS['blue'], linestyle=':',
            linewidth=2, alpha=0.5)
ax5.set_xlabel('r / M', fontsize=11)
ax5.set_ylabel('Specific angular momentum L/M', fontsize=11)
ax5.set_title('Orbital Angular Momentum', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Tidal forces
ax6 = axes[1, 2]
ax6.semilogy(r/M, R_tidal, color=COLORS['red'], linewidth=2)
ax6.axvline(x=r_s/M, color=COLORS['red'], linestyle='--', linewidth=2,
            alpha=0.5, label='Horizon')
ax6.axvline(x=r_isco/M, color=COLORS['blue'], linestyle=':',
            linewidth=1, alpha=0.5)
ax6.set_xlabel('r / M', fontsize=11)
ax6.set_ylabel('Tidal force ∝ M/r³', fontsize=11)
ax6.set_title('Tidal Forces (Spaghettification)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, which='both')
ax6.legend(fontsize=9)
ax6.text(10, 1e-2, 'R ∝ 1/r³', fontsize=9, color=COLORS['red'])

plt.tight_layout()
plt.suptitle('Black Hole Physics: Schwarzschild Solution',
             fontsize=14, fontweight='bold', y=0.995)
plt.show()

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("• Event horizon (r=2M): Point of no return")
print("  - Escape velocity = c")
print("  - Coordinate singularity (not physical)")
print("\n• Photon sphere (r=3M): Unstable circular orbits for light")
print("  - Light can orbit the black hole")
print("  - Boundary between escape and capture")
print("\n• ISCO (r=6M): Innermost stable circular orbit")
print("  - Last stable orbit for massive particles")
print("  - Accretion disk inner edge")
print("\n• Tidal forces: ∝ M/r³")
print("  - Stronger near smaller black holes!")
print("  - 'Spaghettification' effect")
print("=" * 60)
