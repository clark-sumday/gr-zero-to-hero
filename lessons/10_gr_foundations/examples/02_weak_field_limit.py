#!/usr/bin/env python3
"""
Example: Weak Field Limit - GR → Newtonian Gravity
Demonstrates how General Relativity reduces to Newton's theory for weak fields
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Constants (SI units)
G = 6.674e-11  # m^3 kg^-1 s^-2
c = 2.998e8    # m/s
M_sun = 1.989e30  # kg
R_sun = 6.96e8    # m

# Schwarzschild radius
r_s = 2 * G * M_sun / c**2
print("=" * 60)
print("WEAK FIELD LIMIT: GR → NEWTONIAN GRAVITY")
print("=" * 60)
print(f"\nSun's mass: M = {M_sun:.3e} kg")
print(f"Sun's radius: R = {R_sun:.3e} m")
print(f"Schwarzschild radius: r_s = {r_s:.2f} m")
print(f"Ratio r_s/R = {r_s/R_sun:.2e} << 1")
print("\n→ Sun is in WEAK FIELD regime (r_s << R)")

# Radial distance array (from Sun's surface outward)
r = np.linspace(R_sun, 50*R_sun, 1000)

# Metric perturbation h_00 in weak field limit
# g_00 = -(1 + 2Φ/c²) where Φ is Newtonian potential
# For Schwarzschild: g_00 ≈ -(1 - r_s/r) ≈ -(1 - 2GM/(c²r))
# So h_00 = g_00 + 1 = -r_s/r = -2GM/(c²r) = 2Φ/c²

Phi_newtonian = -G * M_sun / r  # Newtonian potential
h_00 = 2 * Phi_newtonian / c**2  # Metric perturbation

# GR metric component
g_00_GR = -(1 - r_s / r)

# Weak field approximation
g_00_weak = -(1 + 2*Phi_newtonian/c**2)

# Relative difference
relative_error = np.abs((g_00_GR - g_00_weak) / g_00_GR)

print("\n" + "=" * 60)
print("METRIC COMPARISON")
print("=" * 60)
print(f"\nAt r = R_sun (Sun's surface):")
print(f"  g_00 (exact GR) = {g_00_GR[0]:.10f}")
print(f"  g_00 (weak field) = {g_00_weak[0]:.10f}")
print(f"  Relative error = {relative_error[0]:.2e}")
print(f"\nAt r = 10 R_sun:")
idx_10 = np.argmin(np.abs(r - 10*R_sun))
print(f"  g_00 (exact GR) = {g_00_GR[idx_10]:.12f}")
print(f"  g_00 (weak field) = {g_00_weak[idx_10]:.12f}")
print(f"  Relative error = {relative_error[idx_10]:.2e}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Metric perturbation
ax1 = axes[0, 0]
ax1.plot(r/R_sun, h_00, color=COLORS['blue'], linewidth=2, label='h₀₀ = 2Φ/c²')
ax1.set_xlabel('Radius (R☉)', fontsize=11)
ax1.set_ylabel('Metric Perturbation h₀₀', fontsize=11)
ax1.set_title('Weak Field Perturbation', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.axhline(y=0, color='k', linewidth=0.5)

# Plot 2: g_00 comparison
ax2 = axes[0, 1]
ax2.plot(r/R_sun, g_00_GR, color=COLORS['red'], linewidth=2, label='GR (exact)', linestyle='-')
ax2.plot(r/R_sun, g_00_weak, color=COLORS['orange'], linewidth=2, label='Weak field approx', linestyle='--')
ax2.set_xlabel('Radius (R☉)', fontsize=11)
ax2.set_ylabel('g₀₀', fontsize=11)
ax2.set_title('Metric Component g₀₀', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.axhline(y=-1, color='k', linewidth=0.5, linestyle=':')

# Plot 3: Relative error
ax3 = axes[1, 0]
ax3.semilogy(r/R_sun, relative_error, color=COLORS['green'], linewidth=2)
ax3.set_xlabel('Radius (R☉)', fontsize=11)
ax3.set_ylabel('Relative Error |ΔG₀₀/g₀₀|', fontsize=11)
ax3.set_title('Accuracy of Weak Field Approximation', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')
ax3.axhline(y=1e-6, color=COLORS['gray'], linestyle='--', alpha=0.5, label='10⁻⁶ precision')
ax3.legend(fontsize=10)

# Plot 4: Newtonian potential
ax4 = axes[1, 1]
ax4.plot(r/R_sun, Phi_newtonian/c**2, color=COLORS['purple'], linewidth=2)
ax4.set_xlabel('Radius (R☉)', fontsize=11)
ax4.set_ylabel('Φ/c² (dimensionless)', fontsize=11)
ax4.set_title('Newtonian Potential (normalized)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.text(30, Phi_newtonian[500]/c**2, 'Φ/c² << 1\n(weak field)',
         fontsize=10, color=COLORS['purple'], ha='center')

plt.tight_layout()
plt.suptitle('Weak Field Limit: General Relativity → Newtonian Gravity',
             fontsize=14, fontweight='bold', y=1.00)
plt.show()

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("When |Φ/c²| << 1 and v << c:")
print("  • GR metric: g_μν ≈ η_μν + h_μν")
print("  • Time component: g_00 ≈ -(1 + 2Φ/c²)")
print("  • Geodesic equation → Newton's F = ma")
print("  • Field equations → Poisson's equation ∇²Φ = 4πGρ")
print("\n→ General Relativity CONTAINS Newtonian gravity as a limit!")
print("=" * 60)
