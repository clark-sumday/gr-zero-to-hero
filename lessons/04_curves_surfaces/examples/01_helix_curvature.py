#!/usr/bin/env python3
"""
Example: Helix Curvature and Torsion
Demonstrates how curves bend (curvature) and twist (torsion) in 3D space
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Parametric helix: r(t) = (a·cos(t), a·sin(t), b·t)
# a = radius, b = pitch

def helix(t, a=1.0, b=0.3):
    """Helix curve"""
    x = a * np.cos(t)
    y = a * np.sin(t)
    z = b * t
    return np.array([x, y, z])

def helix_derivative(t, a=1.0, b=0.3):
    """First derivative r'(t) = velocity"""
    x = -a * np.sin(t)
    y = a * np.cos(t)
    z = b * np.ones_like(t)
    return np.array([x, y, z])

def helix_second_derivative(t, a=1.0, b=0.3):
    """Second derivative r''(t) = acceleration"""
    x = -a * np.cos(t)
    y = -a * np.sin(t)
    z = np.zeros_like(t)
    return np.array([x, y, z])

def compute_curvature(r_prime, r_double_prime):
    """
    Curvature κ = |r' × r''| / |r'|³
    Measures how fast the curve bends
    """
    cross = np.cross(r_prime.T, r_double_prime.T).T
    numerator = np.linalg.norm(cross, axis=0)
    denominator = np.linalg.norm(r_prime, axis=0)**3
    return numerator / denominator

def compute_torsion(r_prime, r_double_prime, r_triple_prime):
    """
    Torsion τ = (r' × r'') · r''' / |r' × r''|²
    Measures how fast the curve twists out of its osculating plane
    """
    cross = np.cross(r_prime.T, r_double_prime.T).T
    numerator = np.sum(cross * r_triple_prime, axis=0)
    denominator = np.sum(cross * cross, axis=0)
    return numerator / denominator

# Parameters
a, b = 1.0, 0.3

# Analytical formulas for helix
# κ = a / (a² + b²)
# τ = b / (a² + b²)

kappa_analytical = a / (a**2 + b**2)
tau_analytical = b / (a**2 + b**2)

print("Helix Curvature and Torsion")
print("=" * 60)
print(f"Helix: r(t) = ({a}·cos(t), {a}·sin(t), {b}·t)")
print()
print("Curvature κ measures how much the curve bends")
print("Torsion τ measures how much the curve twists")
print()
print("Analytical formulas:")
print(f"  κ = a/(a² + b²) = {kappa_analytical:.4f}")
print(f"  τ = b/(a² + b²) = {tau_analytical:.4f}")
print()
print("For a helix, both κ and τ are constant!")

# Compute helix
t = np.linspace(0, 4*np.pi, 500)
r = helix(t, a, b)
r_prime = helix_derivative(t, a, b)
r_double_prime = helix_second_derivative(t, a, b)

# Third derivative for torsion
r_triple_prime = np.array([
    a * np.sin(t),
    -a * np.cos(t),
    np.zeros_like(t)
])

# Compute curvature and torsion
kappa = compute_curvature(r_prime, r_double_prime)
tau = compute_torsion(r_prime, r_double_prime, r_triple_prime)

print(f"\nNumerical verification:")
print(f"  κ (mean) = {np.mean(kappa):.4f}")
print(f"  τ (mean) = {np.mean(tau):.4f}")

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: 3D helix
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(r[0], r[1], r[2], color=COLORS['blue'], linewidth=2, label='Helix')

# Mark a few points
t_points = np.linspace(0, 4*np.pi, 8)
r_points = helix(t_points, a, b)
ax1.scatter(r_points[0], r_points[1], r_points[2],
           color=COLORS['red'], s=100, zorder=5)

ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.set_zlabel('z', fontsize=11)
ax1.set_title('Helix in 3D Space', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)

# Plot 2: Helix with tangent vectors
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot(r[0], r[1], r[2], color=COLORS['blue'], linewidth=2, alpha=0.5)

# Show tangent vectors at selected points
t_selected = np.linspace(0, 4*np.pi, 12)
for ts in t_selected:
    r_s = helix(ts, a, b)
    v_s = helix_derivative(ts, a, b)
    # Normalize and scale for visualization
    v_norm = v_s / np.linalg.norm(v_s) * 0.5
    ax2.quiver(r_s[0], r_s[1], r_s[2],
              v_norm[0], v_norm[1], v_norm[2],
              color=COLORS['red'], arrow_length_ratio=0.3, linewidth=2)

ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y', fontsize=11)
ax2.set_zlabel('z', fontsize=11)
ax2.set_title('Tangent Vectors (velocity)', fontsize=12, fontweight='bold')

# Plot 3: Curvature along the curve
ax3 = plt.subplot(2, 3, 3)
ax3.plot(t, kappa, color=COLORS['green'], linewidth=2, label='Numerical')
ax3.axhline(y=kappa_analytical, color=COLORS['red'], linestyle='--',
           linewidth=2, label=f'Analytical κ = {kappa_analytical:.4f}')
ax3.set_xlabel('Parameter t', fontsize=12)
ax3.set_ylabel('Curvature κ', fontsize=12)
ax3.set_title('Curvature (constant for helix)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Plot 4: Torsion along the curve
ax4 = plt.subplot(2, 3, 4)
ax4.plot(t, tau, color=COLORS['purple'], linewidth=2, label='Numerical')
ax4.axhline(y=tau_analytical, color=COLORS['red'], linestyle='--',
           linewidth=2, label=f'Analytical τ = {tau_analytical:.4f}')
ax4.set_xlabel('Parameter t', fontsize=12)
ax4.set_ylabel('Torsion τ', fontsize=12)
ax4.set_title('Torsion (constant for helix)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# Plot 5: Different helices with different curvatures
ax5 = fig.add_subplot(2, 3, 5, projection='3d')

helix_params = [(1, 0.3), (1, 0.1), (0.5, 0.3)]
helix_colors = [COLORS['blue'], COLORS['green'], COLORS['orange']]
helix_labels = []

for (a_h, b_h), color in zip(helix_params, helix_colors):
    r_h = helix(t[:300], a_h, b_h)
    kappa_h = a_h / (a_h**2 + b_h**2)
    tau_h = b_h / (a_h**2 + b_h**2)
    ax5.plot(r_h[0], r_h[1], r_h[2], color=color, linewidth=2,
            label=f'a={a_h}, b={b_h}: κ={kappa_h:.3f}')

ax5.set_xlabel('x', fontsize=11)
ax5.set_ylabel('y', fontsize=11)
ax5.set_zlabel('z', fontsize=11)
ax5.set_title('Different Helices', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)

# Plot 6: Comparison with circle (zero torsion)
ax6 = fig.add_subplot(2, 3, 6, projection='3d')

# Circle: r(t) = (cos(t), sin(t), 0)
t_circle = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(t_circle), np.sin(t_circle), np.zeros_like(t_circle)])

ax6.plot(circle[0], circle[1], circle[2], color=COLORS['red'],
        linewidth=3, label='Circle (τ=0, planar)')
ax6.plot(r[0, :200], r[1, :200], r[2, :200], color=COLORS['blue'],
        linewidth=3, label=f'Helix (τ={tau_analytical:.3f}, twisted)')

ax6.set_xlabel('x', fontsize=11)
ax6.set_ylabel('y', fontsize=11)
ax6.set_zlabel('z', fontsize=11)
ax6.set_title('Circle vs Helix', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Geometric Interpretation:")
print("="*60)
print("Curvature κ:")
print("  • κ = 0: Straight line")
print("  • κ > 0: Curve bends")
print("  • Larger κ → tighter bend")
print("  • Circle of radius R has κ = 1/R")
print()
print("Torsion τ:")
print("  • τ = 0: Curve lies in a plane")
print("  • τ > 0: Curve twists out of plane (right-handed)")
print("  • τ < 0: Curve twists out of plane (left-handed)")
print()
print("✓ In GR: Curvature generalizes to spacetime!")
print("✓ Try: Change a and b to see different helices")
