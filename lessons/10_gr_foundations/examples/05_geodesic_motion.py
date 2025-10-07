#!/usr/bin/env python3
"""
Example: Geodesic Motion in Curved Spacetime
Demonstrates how particles follow geodesics (curved paths) in curved spacetime
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("GEODESIC EQUATION IN GENERAL RELATIVITY")
print("=" * 60)
print("\nGeodesic equation: d²x^μ/dτ² + Γ^μ_αβ (dx^α/dτ)(dx^β/dτ) = 0")
print("\nFree particles follow geodesics in curved spacetime")

# Physical setup: Schwarzschild metric in geometric units (G=c=1)
M = 1.0  # Mass of central body
r_s = 2 * M  # Schwarzschild radius

print(f"\nSchwarzschild black hole:")
print(f"  Mass M = {M} (geometric units)")
print(f"  Schwarzschild radius r_s = {r_s}")

def christoffel_schwarzschild(r):
    """
    Compute key Christoffel symbols for Schwarzschild metric.
    Using (t, r, θ, φ) coordinates, equatorial plane (θ=π/2, dθ=0).
    """
    f = 1 - r_s/r  # g_tt = -f, g_rr = 1/f

    # Key Christoffel symbols (non-zero components)
    Gamma_t_tr = (r_s / (2*r)) / (r - r_s)  # Γ^t_tr
    Gamma_r_tt = (r_s * f) / (2 * r**2)      # Γ^r_tt
    Gamma_r_rr = -Gamma_r_tt / f             # Γ^r_rr
    Gamma_r_phiphi = -(r - r_s)              # Γ^r_φφ
    Gamma_phi_rphi = 1/r                     # Γ^φ_rφ

    return {
        't_tr': Gamma_t_tr,
        'r_tt': Gamma_r_tt,
        'r_rr': Gamma_r_rr,
        'r_phiphi': Gamma_r_phiphi,
        'phi_rphi': Gamma_phi_rphi
    }

def geodesic_equations(tau, y):
    """
    Geodesic equations in Schwarzschild spacetime (equatorial plane).
    y = [t, r, phi, dt/dtau, dr/dtau, dphi/dtau]
    """
    t, r, phi, dt_dtau, dr_dtau, dphi_dtau = y

    # Avoid singularity
    if r <= r_s * 1.01:
        r = r_s * 1.01

    Gamma = christoffel_schwarzschild(r)

    # Geodesic equations: d²x^μ/dτ² = -Γ^μ_αβ (dx^α/dτ)(dx^β/dτ)
    d2t_dtau2 = -2 * Gamma['t_tr'] * dt_dtau * dr_dtau

    d2r_dtau2 = (-Gamma['r_tt'] * dt_dtau**2
                 - Gamma['r_rr'] * dr_dtau**2
                 - Gamma['r_phiphi'] * dphi_dtau**2)

    d2phi_dtau2 = -2 * Gamma['phi_rphi'] * dr_dtau * dphi_dtau

    return [dt_dtau, dr_dtau, dphi_dtau, d2t_dtau2, d2r_dtau2, d2phi_dtau2]

# Initial conditions for different orbits
# Orbit 1: Stable circular orbit
r0_stable = 6 * M  # r > 3r_s (stable orbits exist)
angular_momentum_stable = np.sqrt(M * r0_stable / (1 - 3*M/r0_stable))
energy_stable = np.sqrt((1 - 2*M/r0_stable) / (1 - 3*M/r0_stable))

# Initial state: [t, r, phi, dt/dtau, dr/dtau, dphi/dtau]
y0_stable = [0, r0_stable, 0, energy_stable, 0, angular_momentum_stable/r0_stable**2]

# Orbit 2: Precessing elliptical orbit
r0_ellipse = 8 * M
L_ellipse = 3.9 * M  # Angular momentum
E_ellipse = 0.98     # Energy < 1 (bound orbit)
dr_dtau0 = 0.1       # Small radial velocity

y0_ellipse = [0, r0_ellipse, 0, E_ellipse, dr_dtau0, L_ellipse/r0_ellipse**2]

# Orbit 3: Plunging orbit
r0_plunge = 5 * M
L_plunge = 3.5 * M   # Low angular momentum
E_plunge = 0.95
dr_dtau0_plunge = -0.2

y0_plunge = [0, r0_plunge, 0, E_plunge, dr_dtau0_plunge, L_plunge/r0_plunge**2]

# Solve geodesic equations
tau_max = 200
print("\nIntegrating geodesic equations...")

sol_stable = solve_ivp(geodesic_equations, [0, tau_max], y0_stable,
                       max_step=0.1, dense_output=True)
sol_ellipse = solve_ivp(geodesic_equations, [0, tau_max], y0_ellipse,
                        max_step=0.1, dense_output=True)
sol_plunge = solve_ivp(geodesic_equations, [0, tau_max], y0_plunge,
                       max_step=0.1, dense_output=True)

# Extract solutions
t_stable = sol_stable.y[0]
r_stable = sol_stable.y[1]
phi_stable = sol_stable.y[2]

t_ellipse = sol_ellipse.y[0]
r_ellipse = sol_ellipse.y[1]
phi_ellipse = sol_ellipse.y[2]

t_plunge = sol_plunge.y[0]
r_plunge = sol_plunge.y[1]
phi_plunge = sol_plunge.y[2]

# Convert to Cartesian coordinates for plotting
x_stable = r_stable * np.cos(phi_stable)
y_stable = r_stable * np.sin(phi_stable)

x_ellipse = r_ellipse * np.cos(phi_ellipse)
y_ellipse = r_ellipse * np.sin(phi_ellipse)

x_plunge = r_plunge * np.cos(phi_plunge)
y_plunge = r_plunge * np.sin(phi_plunge)

print("Integration complete!")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Plot 1: Orbits in space
ax1 = axes[0]

# Event horizon
theta_circle = np.linspace(0, 2*np.pi, 100)
ax1.fill(r_s * np.cos(theta_circle), r_s * np.sin(theta_circle),
         color=COLORS['black'], alpha=0.8, label='Black hole')
ax1.plot(r_s * np.cos(theta_circle), r_s * np.sin(theta_circle),
         color=COLORS['red'], linewidth=2, linestyle='--', label=f'Horizon (r={r_s})')

# ISCO (Innermost Stable Circular Orbit at r = 6M)
r_isco = 6 * M
ax1.plot(r_isco * np.cos(theta_circle), r_isco * np.sin(theta_circle),
         color=COLORS['gray'], linewidth=1, linestyle=':', alpha=0.5, label=f'ISCO (r=6M)')

# Plot orbits
ax1.plot(x_stable, y_stable, color=COLORS['blue'], linewidth=1.5,
         label='Circular orbit (r=6M)', alpha=0.8)
ax1.plot(x_ellipse, y_ellipse, color=COLORS['orange'], linewidth=1.5,
         label='Precessing ellipse', alpha=0.8)
ax1.plot(x_plunge, y_plunge, color=COLORS['purple'], linewidth=2,
         label='Plunging orbit', alpha=0.9)

# Mark starting points
ax1.plot(x_stable[0], y_stable[0], 'o', color=COLORS['blue'], markersize=8)
ax1.plot(x_ellipse[0], y_ellipse[0], 'o', color=COLORS['orange'], markersize=8)
ax1.plot(x_plunge[0], y_plunge[0], 'o', color=COLORS['purple'], markersize=8)

ax1.set_xlabel('x/M', fontsize=11)
ax1.set_ylabel('y/M', fontsize=11)
ax1.set_title('Geodesics in Schwarzschild Spacetime', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.legend(fontsize=9, loc='upper right')
ax1.set_xlim(-15, 15)
ax1.set_ylim(-15, 15)

# Plot 2: Radial coordinate vs proper time
ax2 = axes[1]
ax2.plot(sol_stable.t, r_stable, color=COLORS['blue'], linewidth=2,
         label='Circular orbit')
ax2.plot(sol_ellipse.t, r_ellipse, color=COLORS['orange'], linewidth=2,
         label='Precessing ellipse')
ax2.plot(sol_plunge.t, r_plunge, color=COLORS['purple'], linewidth=2,
         label='Plunging orbit')
ax2.axhline(y=r_s, color=COLORS['red'], linestyle='--', linewidth=2,
            label='Event horizon')
ax2.axhline(y=r_isco, color=COLORS['gray'], linestyle=':', linewidth=1,
            alpha=0.5, label='ISCO')
ax2.set_xlabel('Proper time τ/M', fontsize=11)
ax2.set_ylabel('Radial coordinate r/M', fontsize=11)
ax2.set_title('Radius vs Proper Time', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 15)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("KEY OBSERVATIONS")
print("=" * 60)
print(f"• Circular orbit at r=6M: Stable (ISCO)")
print(f"• Elliptical orbit: Precession (perihelion shift)")
print(f"• Plunging orbit: Falls into black hole (crosses r={r_s})")
print("\nGeodesic features:")
print("  • No external force needed - curvature guides motion")
print("  • Energy and angular momentum conserved")
print("  • Closer to black hole → stronger curvature effects")
print("  • Below ISCO (r<6M): no stable circular orbits!")
print("=" * 60)
