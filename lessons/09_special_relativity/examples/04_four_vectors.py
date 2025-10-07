#!/usr/bin/env python3
"""
Example: Four-Vectors and Relativistic Energy-Momentum
Demonstrates 4-vector formalism and E=mc²
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("FOUR-VECTORS AND ENERGY-MOMENTUM")
print("="*60)

c = 1.0  # Natural units (c=1)

def gamma(v, c=1.0):
    """Lorentz factor"""
    return 1 / np.sqrt(1 - (v/c)**2)

# Minkowski metric
eta = np.diag([-1, 1, 1, 1])
print("\nMinkowski metric η_μν:")
print(eta)

# Example: 4-velocity
print("\n" + "="*60)
print("FOUR-VELOCITY")
print("="*60)

v = 0.6 * c  # 3-velocity
gamma_v = gamma(v, c)

# 4-velocity: u^μ = γ(1, v/c, 0, 0)
u = np.array([gamma_v, gamma_v * v/c, 0, 0])

print(f"3-velocity: v = {v/c:.1f}c")
print(f"γ = {gamma_v:.4f}")
print(f"4-velocity: u^μ = {u}")

# Normalization
u_lower = eta @ u
norm = u @ u_lower

print(f"\nNormalization: u^μ u_μ = {norm:.6f}")
print("Expected: -1 (timelike, massive particle) ✓")

# Example: 4-momentum
print("\n" + "="*60)
print("FOUR-MOMENTUM")
print("="*60)

m = 1.0  # Rest mass
print(f"Rest mass: m = {m}")

# 4-momentum: p^μ = m u^μ = (E/c, p_x, p_y, p_z)
p = m * u

E = p[0]  # Energy
p_x = p[1]  # Momentum x-component

print(f"4-momentum: p^μ = {p}")
print(f"Energy: E = γmc² = {E:.4f}")
print(f"Momentum: p_x = γmv = {p_x:.4f}")

# Mass-energy relation
p_lower = eta @ p
m_squared = -p @ p_lower

print(f"\nMass-shell condition: p^μ p_μ = -m²")
print(f"  Computed: {p @ p_lower:.6f}")
print(f"  Expected: {-m**2:.6f}")
print(f"  Match: {np.isclose(-m_squared, m**2)}")

# E² = p²c² + (mc²)²
E_check = np.sqrt(p_x**2 + m**2)
print(f"\nEnergy-momentum relation: E² = p² + m²")
print(f"  E from 4-momentum: {E:.4f}")
print(f"  E from √(p²+m²): {E_check:.4f}")

# Rest frame: E = mc²
gamma_rest = gamma(0, c)
E_rest = gamma_rest * m

print(f"\nRest frame (v=0):")
print(f"  γ = {gamma_rest}")
print(f"  E = mc² = {E_rest:.4f}")
print("  ✓ E = mc² (Einstein's famous equation!)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Energy vs velocity
ax1 = axes[0, 0]
v_range = np.linspace(0, 0.99*c, 100)
gamma_range = gamma(v_range, c)
E_range = gamma_range * m

ax1.plot(v_range/c, E_range, color=COLORS['blue'], linewidth=3, label='Total E')
ax1.axhline(m, color=COLORS['green'], linestyle='--', linewidth=2, label='Rest energy mc²')

# Kinetic energy
KE_range = E_range - m
ax1.plot(v_range/c, KE_range, color=COLORS['red'], linewidth=2, label='Kinetic energy', alpha=0.7)

ax1.set_xlabel('Velocity v/c', fontsize=12)
ax1.set_ylabel('Energy (mc² units)', fontsize=12)
ax1.set_title('Relativistic Energy vs Velocity\nE = γmc²', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 10])

# Plot 2: Momentum vs velocity
ax2 = axes[0, 1]
p_range = gamma_range * v_range / c * m

ax2.plot(v_range/c, p_range, color=COLORS['orange'], linewidth=3)

# Classical momentum (for comparison)
p_classical = v_range / c * m
ax2.plot(v_range/c, p_classical, '--', color=COLORS['gray'], linewidth=2,
        label='Classical p=mv', alpha=0.7)

ax2.set_xlabel('Velocity v/c', fontsize=12)
ax2.set_ylabel('Momentum (mc units)', fontsize=12)
ax2.set_title('Relativistic Momentum\np = γmv', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: E-p diagram
ax3 = axes[1, 0]

# Energy-momentum relation: E² = p² + m²
p_plot = np.linspace(0, 5, 100)
E_plot = np.sqrt(p_plot**2 + m**2)

ax3.plot(p_plot, E_plot, color=COLORS['purple'], linewidth=3, label='E² = p² + m²')

# Mark specific points
velocities = [0, 0.5*c, 0.8*c]
for v_mark in velocities:
    g = gamma(v_mark, c)
    p_mark = g * (v_mark/c) * m
    E_mark = g * m
    ax3.plot(p_mark, E_mark, 'o', markersize=10)
    ax3.text(p_mark + 0.2, E_mark, f'v={v_mark/c:.1f}c', fontsize=9)

# Massless particle (photon): E = p
p_photon = np.linspace(0, 5, 100)
E_photon = p_photon
ax3.plot(p_photon, E_photon, '--', color=COLORS['yellow'], linewidth=2, label='Massless (E=p)')

ax3.set_xlabel('Momentum p (mc units)', fontsize=12)
ax3.set_ylabel('Energy E (mc² units)', fontsize=12)
ax3.set_title('Energy-Momentum Relation\n(Mass shell)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 5])
ax3.set_ylim([0, 6])

# Plot 4: Summary
ax4 = axes[1, 1]
summary = """
Four-Vectors in Special Relativity

4-Position: x^μ = (t, x, y, z)

4-Velocity: u^μ = dx^μ/dτ = γ(1, v/c)
• Normalization: u^μ u_μ = -1
• Proper time: dτ = dt/γ

4-Momentum: p^μ = mu^μ = (E/c, p⃗)
• Energy: E = γmc²
• Momentum: p⃗ = γmv⃗
• Mass-shell: p^μ p_μ = -m²c²
• E² = (pc)² + (mc²)²

Special Cases:
• Rest frame (v=0):
  E = mc² (rest energy)
  p = 0
• Massless (m=0):
  E = pc (photon)
  u^μ u_μ = 0 (null)

Conservation Laws:
• Energy-momentum conserved in collisions
• p^μ before = p^μ after
• Manifestly covariant!

Applications:
• Particle physics
• Compton scattering
• Pair production/annihilation
"""

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
        fontsize=9, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))
ax4.axis('off')

plt.tight_layout()
plt.show()

print("\n✓ 4-vectors make relativity equations elegant and covariant!")
