#!/usr/bin/env python3
"""
Example: Tensor Transformations Between Coordinate Systems
Demonstrates how contravariant and covariant tensors transform under coordinate changes
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

def cartesian_to_polar(x, y):
    """Convert Cartesian to polar coordinates"""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def jacobian_cart_to_polar(x, y):
    """
    Jacobian matrix ∂(r,θ)/∂(x,y)
    Used to transform contravariant tensors
    """
    r = np.sqrt(x**2 + y**2)
    if r < 1e-10:
        return np.eye(2)

    dr_dx = x / r
    dr_dy = y / r
    dtheta_dx = -y / r**2
    dtheta_dy = x / r**2

    return np.array([
        [dr_dx, dr_dy],
        [dtheta_dx, dtheta_dy]
    ])

def inverse_jacobian_cart_to_polar(x, y):
    """
    Inverse Jacobian ∂(x,y)/∂(r,θ)
    Used to transform covariant tensors
    """
    r, theta = cartesian_to_polar(x, y)

    dx_dr = x / r if r > 1e-10 else 1
    dx_dtheta = -y
    dy_dr = y / r if r > 1e-10 else 0
    dy_dtheta = x

    return np.array([
        [dx_dr, dx_dtheta],
        [dy_dr, dy_dtheta]
    ])

# Test point
x, y = 3.0, 4.0
r, theta = cartesian_to_polar(x, y)

print("="*60)
print("COORDINATE TRANSFORMATION: CARTESIAN → POLAR")
print("="*60)
print(f"\nPoint in Cartesian: (x, y) = ({x}, {y})")
print(f"Point in polar: (r, θ) = ({r:.3f}, {theta:.3f} rad = {np.degrees(theta):.1f}°)")

# Contravariant vector (like velocity)
v_cartesian = np.array([1.0, 2.0])
print(f"\n{'='*60}")
print("CONTRAVARIANT VECTOR (velocity-like)")
print("="*60)
print(f"Vector in Cartesian: v^μ = {v_cartesian}")

J = jacobian_cart_to_polar(x, y)
v_polar = J @ v_cartesian

print(f"\nJacobian ∂(r,θ)/∂(x,y):")
print(J)
print(f"\nTransformed: v'^μ = (∂x'^μ/∂x^ν) v^ν")
print(f"Vector in polar: v' = {v_polar}")
print(f"  v^r = {v_polar[0]:.4f}")
print(f"  v^θ = {v_polar[1]:.4f}")

# Covariant vector (like gradient)
w_cartesian = np.array([5.0, 6.0])
print(f"\n{'='*60}")
print("COVARIANT VECTOR (gradient-like)")
print("="*60)
print(f"One-form in Cartesian: w_μ = {w_cartesian}")

J_inv = inverse_jacobian_cart_to_polar(x, y)
w_polar = J_inv.T @ w_cartesian  # Transpose for covariant

print(f"\nInverse Jacobian ∂(x,y)/∂(r,θ):")
print(J_inv)
print(f"\nTransformed: w'_μ = (∂x^ν/∂x'^μ) w_ν")
print(f"One-form in polar: w' = {w_polar}")
print(f"  w_r = {w_polar[0]:.4f}")
print(f"  w_θ = {w_polar[1]:.4f}")

# Verify: dot product is invariant
dot_cartesian = np.dot(v_cartesian, w_cartesian)
dot_polar = np.dot(v_polar, w_polar)
print(f"\n{'='*60}")
print("INVARIANCE CHECK")
print("="*60)
print(f"Dot product in Cartesian: v^μ w_μ = {dot_cartesian:.6f}")
print(f"Dot product in polar: v'^μ w'_μ = {dot_polar:.6f}")
print(f"Difference: {abs(dot_cartesian - dot_polar):.2e}")
print(f"✓ Invariant!" if np.isclose(dot_cartesian, dot_polar) else "✗ Error!")

# Visualization
fig = plt.figure(figsize=(14, 6))

# Left: Cartesian basis
ax1 = plt.subplot(1, 2, 1)
ax1.arrow(0, 0, 1, 0, head_width=0.2, head_length=0.15,
         fc=COLORS['blue'], ec=COLORS['blue'], linewidth=2, label='e_x')
ax1.arrow(0, 0, 0, 1, head_width=0.2, head_length=0.15,
         fc=COLORS['orange'], ec=COLORS['orange'], linewidth=2, label='e_y')

# Show the point
ax1.plot(x, y, 'ko', markersize=10, label=f'Point ({x}, {y})')

# Show the vector
ax1.arrow(x, y, v_cartesian[0], v_cartesian[1], head_width=0.3, head_length=0.2,
         fc=COLORS['green'], ec=COLORS['green'], linewidth=3, alpha=0.7, label='v (Cartesian)')

ax1.text(x + v_cartesian[0]/2, y + v_cartesian[1]/2 + 0.5,
         f'v = ({v_cartesian[0]:.1f}, {v_cartesian[1]:.1f})',
         fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax1.set_xlim([-1, 6])
ax1.set_ylim([-1, 8])
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Cartesian Coordinates', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='k', linewidth=0.5)
ax1.axvline(0, color='k', linewidth=0.5)
ax1.set_aspect('equal')

# Right: Polar basis at the point
ax2 = plt.subplot(1, 2, 2, projection='polar')

# Show radial circles and angular lines
theta_grid = np.linspace(0, 2*np.pi, 100)
for rad in [1, 2, 3, 4, 5]:
    ax2.plot(theta_grid, np.ones_like(theta_grid) * rad, 'gray', alpha=0.3, linewidth=0.5)

# Show the point
ax2.plot(theta, r, 'ko', markersize=10, label=f'Point (r={r:.1f}, θ={np.degrees(theta):.0f}°)')

# Show basis vectors at the point (in polar representation)
# e_r points radially, e_θ is tangent to circle
ax2.arrow(theta, r, 0, 0.8, head_width=0.15, head_length=0.2,
         fc=COLORS['blue'], ec=COLORS['blue'], linewidth=2, alpha=0.6)
ax2.text(theta, r + 1.2, 'e_r', fontsize=11, ha='center', color=COLORS['blue'])

# e_θ (tangent)
dtheta = 0.2
ax2.plot([theta, theta + dtheta], [r, r], color=COLORS['orange'], linewidth=3, alpha=0.6)
ax2.text(theta + dtheta/2, r + 0.3, 'e_θ', fontsize=11, ha='center', color=COLORS['orange'])

# Show the transformed vector (approximately)
# Note: In polar plot, we show v^r e_r + v^θ e_θ
ax2.arrow(theta, r, 0, v_polar[0], head_width=0.15, head_length=0.2,
         fc=COLORS['green'], ec=COLORS['green'], linewidth=3, alpha=0.7)

ax2.text(theta - 0.3, r + v_polar[0] + 1,
         f"v' = ({v_polar[0]:.2f}, {v_polar[1]:.2f})",
         fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax2.set_title('Polar Coordinates\n(Same vector, different components)',
             fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax2.set_ylim([0, 8])

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY INSIGHT")
print("="*60)
print("The vector itself doesn't change - it's a geometric object.")
print("But its COMPONENTS change when we use different coordinates.")
print("Contravariant components transform with ∂x'/∂x (Jacobian)")
print("Covariant components transform with ∂x/∂x' (inverse Jacobian transpose)")
print("\n✓ This ensures physical laws remain the same in all coordinate systems!")
