#!/usr/bin/env python3
"""
Example: Einstein Tensor and Curvature
Demonstrates computation of Einstein tensor for simple metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, Matrix, simplify, cos, sin
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("EINSTEIN TENSOR COMPUTATION")
print("=" * 60)
print("\nEinstein Field Equations: G_μν = (8πG/c⁴) T_μν")
print("where Einstein tensor: G_μν = R_μν - (1/2) R g_μν")

# Define symbolic coordinates
t, r, theta, phi = symbols('t r theta phi', real=True)
M, c, G = symbols('M c G', positive=True, real=True)

# Example 1: Flat spacetime (Minkowski metric)
print("\n" + "=" * 60)
print("EXAMPLE 1: MINKOWSKI SPACETIME (flat)")
print("=" * 60)

eta = Matrix([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

print("\nMetric tensor η_μν:")
print(eta)
print("\nFor flat spacetime:")
print("  • Riemann tensor R^ρ_σμν = 0")
print("  • Ricci tensor R_μν = 0")
print("  • Ricci scalar R = 0")
print("  • Einstein tensor G_μν = 0")
print("\n→ No curvature, no gravity!")

# Example 2: 2D Sphere (curved space, not spacetime)
print("\n" + "=" * 60)
print("EXAMPLE 2: 2D SPHERE (S²)")
print("=" * 60)

R_sphere = symbols('R', positive=True, real=True)  # Sphere radius
theta_s, phi_s = symbols('theta phi', real=True)

g_sphere = Matrix([
    [R_sphere**2, 0],
    [0, R_sphere**2 * sin(theta_s)**2]
])

print("\nMetric: ds² = R² dθ² + R² sin²θ dφ²")
print("\nFor a 2-sphere of radius R:")
print("  • Ricci tensor: R_μν = (1/R²) g_μν")
print("  • Ricci scalar: R = 2/R²")
print("  • Gaussian curvature: K = 1/R²")
print("\n→ Positive curvature, independent of position")

# Numerical example: Compute curvature for different radii
radii = np.array([1, 2, 5, 10, 100])
curvatures = 1 / radii**2
ricci_scalars = 2 / radii**2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Curvature vs radius
ax1 = axes[0]
ax1.loglog(radii, curvatures, 'o-', color=COLORS['blue'],
           markersize=8, linewidth=2, label='Gaussian curvature K')
ax1.loglog(radii, ricci_scalars, 's-', color=COLORS['orange'],
           markersize=8, linewidth=2, label='Ricci scalar R')
ax1.set_xlabel('Sphere Radius R', fontsize=11)
ax1.set_ylabel('Curvature', fontsize=11)
ax1.set_title('Curvature vs Sphere Radius', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=10)
ax1.text(3, 0.05, 'K ~ 1/R²\n(smaller radius\n→ more curved)',
         fontsize=10, color=COLORS['blue'], ha='center')

# Plot 2: Visualize spheres with different curvatures
ax2 = axes[1]
# Draw circles representing sphere cross-sections
for idx, R in enumerate([1, 2, 4]):
    theta_plot = np.linspace(0, 2*np.pi, 100)
    x = R * np.cos(theta_plot)
    y = R * np.sin(theta_plot)
    color = [COLORS['blue'], COLORS['orange'], COLORS['green']][idx]
    K = 1/R**2
    ax2.plot(x, y, color=color, linewidth=2,
             label=f'R={R}, K={K:.2f}')

ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y', fontsize=11)
ax2.set_title('2D Spheres with Different Curvatures', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.legend(fontsize=10)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

# Example 3: Schwarzschild metric components
print("\n" + "=" * 60)
print("EXAMPLE 3: SCHWARZSCHILD SPACETIME")
print("=" * 60)

# Define Schwarzschild radius
r_s = 2*G*M/c**2

print("\nMetric: ds² = -(1-r_s/r)c²dt² + (1-r_s/r)⁻¹dr² + r²dΩ²")
print(f"\nwhere r_s = 2GM/c² (Schwarzschild radius)")
print("\nNon-zero Einstein tensor components (outside r=0):")
print("  • In vacuum (T_μν = 0): G_μν = 0")
print("  • At r = 0: G_μν → ∞ (singularity)")
print("\nKey properties:")
print("  • Ricci tensor R_μν = 0 (vacuum solution)")
print("  • Ricci scalar R = 0")
print("  • Riemann tensor R^ρ_σμν ≠ 0 (spacetime IS curved!)")

# Numerical: Components of Riemann tensor
# Using geometric units (G=c=1, M=1)
r_vals = np.linspace(2.5, 20, 100)  # Start outside r_s = 2M
r_s_num = 2.0  # M=1, G=c=1

# Kretschmann scalar K = R^αβγδ R_αβγδ (measures curvature)
# For Schwarzschild: K = 48 M² / r⁶ = 12 r_s² / r⁶
K_scalar = 12 * r_s_num**2 / r_vals**6

fig2, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(r_vals/r_s_num, K_scalar, color=COLORS['purple'], linewidth=2)
ax.axvline(x=1, color=COLORS['red'], linestyle='--', linewidth=2,
           label='Event horizon (r = r_s)')
ax.set_xlabel('r / r_s', fontsize=11)
ax.set_ylabel('Kretschmann Scalar K', fontsize=11)
ax.set_title('Spacetime Curvature around Schwarzschild Black Hole',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10)
ax.text(1.5, 1e-2, 'Curvature → ∞\nas r → r_s',
        fontsize=10, color=COLORS['purple'])
ax.text(10, 1e-8, 'K ~ 1/r⁶\n(falls rapidly)',
        fontsize=10, color=COLORS['purple'])

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("KEY INSIGHTS: EINSTEIN TENSOR")
print("=" * 60)
print("• Einstein tensor G_μν encodes spacetime curvature")
print("• G_μν = R_μν - (1/2)R g_μν (combines Ricci tensor & scalar)")
print("• Automatically conserved: ∇^μ G_μν = 0")
print("• In vacuum: G_μν = 0, but R^α_βγδ ≠ 0 (can still be curved!)")
print("• Sources gravity: G_μν = (8πG/c⁴) T_μν")
print("• 10 independent equations (4D spacetime)")
print("=" * 60)

print("\nNote: Full symbolic computation of G_μν requires:")
print("  1. Compute Christoffel symbols Γ^μ_νλ from metric")
print("  2. Compute Riemann tensor R^ρ_σμν from Γ")
print("  3. Contract to get Ricci tensor R_μν")
print("  4. Contract to get Ricci scalar R")
print("  5. Combine into Einstein tensor G_μν")
print("\n  (See Lessons 6-7 for detailed calculations)")
