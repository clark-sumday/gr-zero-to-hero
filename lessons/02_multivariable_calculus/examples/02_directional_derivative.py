#!/usr/bin/env python3
"""
Example: Directional Derivative Visualization
Shows how the rate of change depends on direction
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Define function f(x, y) = x² + 2y²
def f(x, y):
    return x**2 + 2*y**2

# Gradient: ∇f = (2x, 4y)
def gradient_f(x, y):
    return np.array([2*x, 4*y])

# Directional derivative: D_u f = ∇f · u (where u is unit vector)
def directional_derivative(x, y, direction):
    grad = gradient_f(x, y)
    # Normalize direction
    u = direction / np.linalg.norm(direction)
    return np.dot(grad, u)

# Point of interest
point = np.array([1.0, 1.0])
z_point = f(point[0], point[1])

# Test different directions
angles = np.linspace(0, 2*np.pi, 100)
directions = np.array([[np.cos(theta), np.sin(theta)] for theta in angles])
derivatives = [directional_derivative(point[0], point[1], d) for d in directions]

# Find max and min
max_idx = np.argmax(derivatives)
min_idx = np.argmin(derivatives)

print("Directional Derivative Example: f(x, y) = x² + 2y²")
print("=" * 60)
print(f"At point {point}:")
print(f"  f({point[0]}, {point[1]}) = {z_point}")
print(f"  ∇f = {gradient_f(point[0], point[1])}")
print(f"\nMaximum rate of change:")
print(f"  Direction: {directions[max_idx]}")
print(f"  D_u f = {derivatives[max_idx]:.3f}")
print(f"  (This is in the gradient direction!)")
print(f"\nMinimum rate of change:")
print(f"  Direction: {directions[min_idx]}")
print(f"  D_u f = {derivatives[min_idx]:.3f}")
print(f"  (Opposite to gradient)")

# Create visualization
fig = plt.figure(figsize=(14, 5))

# Left: Contour plot with direction vectors
ax1 = plt.subplot(1, 3, 1)
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

contour = ax1.contour(X, Y, Z, levels=20, colors=COLORS['gray'], alpha=0.4)
ax1.clabel(contour, inline=True, fontsize=8)

# Show gradient direction
grad = gradient_f(point[0], point[1])
grad_norm = grad / np.linalg.norm(grad)
ax1.quiver(point[0], point[1], grad_norm[0], grad_norm[1],
           angles='xy', scale_units='xy', scale=2,
           color=COLORS['red'], width=0.01, label='∇f (max rate)')

# Show perpendicular direction (zero derivative)
perp = np.array([-grad_norm[1], grad_norm[0]])
ax1.quiver(point[0], point[1], perp[0], perp[1],
           angles='xy', scale_units='xy', scale=2,
           color=COLORS['blue'], width=0.01, label='Perpendicular (zero rate)')

# Show arbitrary direction
arb_dir = np.array([1, 0.5])
arb_norm = arb_dir / np.linalg.norm(arb_dir)
ax1.quiver(point[0], point[1], arb_norm[0], arb_norm[1],
           angles='xy', scale_units='xy', scale=2,
           color=COLORS['orange'], width=0.01, label='Arbitrary direction')

ax1.scatter([point[0]], [point[1]], color='black', s=100, zorder=5)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.set_title('Different Directions at Point (1, 1)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.set_aspect('equal')
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 2.5)

# Middle: Polar plot of directional derivatives
ax2 = plt.subplot(1, 3, 2, projection='polar')
ax2.plot(angles, derivatives, color=COLORS['blue'], linewidth=2)
ax2.fill(angles, derivatives, color=COLORS['blue'], alpha=0.2)
ax2.plot(angles[max_idx], derivatives[max_idx], 'o', color=COLORS['red'],
         markersize=10, label='Max (gradient dir)')
ax2.plot(angles[min_idx], derivatives[min_idx], 'o', color=COLORS['orange'],
         markersize=10, label='Min (opposite)')
ax2.set_title('Directional Derivative D_u f\nvs Direction', fontsize=12, pad=20)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Right: 3D surface with cross-sections
ax3 = plt.subplot(1, 3, 3, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4, edgecolor='none')

# Show gradient direction cross-section
t = np.linspace(-0.5, 0.5, 50)
grad_path_x = point[0] + t * grad_norm[0]
grad_path_y = point[1] + t * grad_norm[1]
grad_path_z = f(grad_path_x, grad_path_y)
ax3.plot(grad_path_x, grad_path_y, grad_path_z, color=COLORS['red'],
         linewidth=3, label='Gradient direction')

# Show perpendicular cross-section
perp_path_x = point[0] + t * perp[0]
perp_path_y = point[1] + t * perp[1]
perp_path_z = f(perp_path_x, perp_path_y)
ax3.plot(perp_path_x, perp_path_y, perp_path_z, color=COLORS['blue'],
         linewidth=3, label='Perpendicular (tangent to level curve)')

ax3.scatter([point[0]], [point[1]], [z_point], color='black', s=100)
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('y', fontsize=10)
ax3.set_zlabel('f(x, y)', fontsize=10)
ax3.set_title('Cross-sections through Point', fontsize=12)
ax3.legend(fontsize=9)

plt.tight_layout()
plt.show()

print("\n✓ Key insight: Maximum rate of change = |∇f|")
print("✓ Direction of max rate = direction of ∇f")
print("✓ Zero rate of change perpendicular to ∇f (along level curve)")
