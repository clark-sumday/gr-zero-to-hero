#!/usr/bin/env python3
"""
Example: Gradient Field Visualization
Demonstrates how the gradient points in the direction of steepest ascent
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Define a scalar field f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Compute the gradient ∇f = (∂f/∂x, ∂f/∂y) = (2x, 2y)
def gradient_f(x, y):
    return np.array([2*x, 2*y])

# Create grid
x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)

# Compute function values and gradient at each point
Z = f(X, Y)
U, V = 2*X, 2*Y

# Normalize for better visualization
magnitude = np.sqrt(U**2 + V**2)
U_norm = U / (magnitude + 1e-10)
V_norm = V / (magnitude + 1e-10)

print("Gradient Field Example: f(x, y) = x² + y²")
print("=" * 50)
print(f"∇f = (∂f/∂x, ∂f/∂y) = (2x, 2y)")
print(f"\nAt point (1, 1):")
print(f"  f(1, 1) = {f(1, 1)}")
print(f"  ∇f(1, 1) = {gradient_f(1, 1)}")
print(f"\nThe gradient points radially outward (steepest ascent)!")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Contour plot with gradient vectors
contour = ax1.contour(X, Y, Z, levels=15, colors=COLORS['gray'], alpha=0.4)
ax1.clabel(contour, inline=True, fontsize=8)
ax1.quiver(X[::3, ::3], Y[::3, ::3], U_norm[::3, ::3], V_norm[::3, ::3],
           color=COLORS['blue'], alpha=0.7, width=0.003)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Gradient Field ∇f (perpendicular to level curves)', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_aspect('equal')

# Right: 3D surface with gradient at a point
ax2 = plt.subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# Show gradient vector at point (1, 1)
point = np.array([1, 1])
grad = gradient_f(point[0], point[1])
z_val = f(point[0], point[1])

ax2.scatter([point[0]], [point[1]], [z_val], color=COLORS['red'], s=100, label='Point (1,1)')
ax2.quiver(point[0], point[1], z_val, grad[0], grad[1], 0,
           color=COLORS['red'], arrow_length_ratio=0.3, linewidth=2,
           label='∇f(1,1)')

ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.set_zlabel('f(x, y)', fontsize=10)
ax2.set_title('Surface f(x, y) = x² + y²', fontsize=13)
ax2.legend()

plt.tight_layout()
plt.show()

print("\n✓ Notice: The gradient is always perpendicular to level curves!")
print("✓ Try: Modify f(x,y) to x² - y² (saddle point)")
