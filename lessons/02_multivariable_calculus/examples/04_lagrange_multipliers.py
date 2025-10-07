#!/usr/bin/env python3
"""
Example: Lagrange Multipliers for Constrained Optimization
Find extrema of f(x,y) subject to constraint g(x,y) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Problem: Maximize f(x, y) = xy subject to constraint x² + y² = 2

def objective(x, y):
    """Function to optimize: f(x,y) = xy"""
    return x * y

def constraint(x, y):
    """Constraint: g(x,y) = x² + y² - 2 = 0"""
    return x**2 + y**2 - 2

def gradient_f(x, y):
    """∇f = (y, x)"""
    return np.array([y, x])

def gradient_g(x, y):
    """∇g = (2x, 2y)"""
    return np.array([2*x, 2*y])

# Lagrange condition: ∇f = λ∇g
# (y, x) = λ(2x, 2y)
# y = 2λx and x = 2λy
# From these: y = 2λx and x = 2λy => y = 4λ²y => λ = ±1/2
# If λ = 1/2: y = x, so x² + x² = 2 => x = ±1, y = ±1
# If λ = -1/2: y = -x, so x² + x² = 2 => x = ±1, y = ∓1

critical_points = [
    (1, 1),    # f = 1 (maximum)
    (-1, -1),  # f = 1 (maximum)
    (1, -1),   # f = -1 (minimum)
    (-1, 1),   # f = -1 (minimum)
]

print("Lagrange Multipliers Example")
print("=" * 60)
print("Problem: Optimize f(x, y) = xy")
print("Subject to: x² + y² = 2")
print()
print("Method: ∇f = λ∇g")
print("  ∇f = (y, x)")
print("  ∇g = (2x, 2y)")
print()
print("Critical points and values:")
for pt in critical_points:
    f_val = objective(pt[0], pt[1])
    grad_f = gradient_f(pt[0], pt[1])
    grad_g = gradient_g(pt[0], pt[1])
    # Compute λ from condition
    if abs(grad_g[0]) > 1e-6:
        lam = grad_f[0] / grad_g[0]
    else:
        lam = grad_f[1] / grad_g[1]
    print(f"  ({pt[0]:4.1f}, {pt[1]:4.1f}): f = {f_val:5.1f}, λ = {lam:5.2f}")

print(f"\nMaximum: f = 1 at (±1, ±1)")
print(f"Minimum: f = -1 at (±1, ∓1)")

# Visualization
fig = plt.figure(figsize=(15, 5))

# Left: Contour plot with constraint
ax1 = plt.subplot(1, 3, 1)
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = objective(X, Y)

# Plot objective function contours
contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
ax1.clabel(contour, inline=True, fontsize=8)

# Plot constraint circle
theta = np.linspace(0, 2*np.pi, 100)
constraint_x = np.sqrt(2) * np.cos(theta)
constraint_y = np.sqrt(2) * np.sin(theta)
ax1.plot(constraint_x, constraint_y, color=COLORS['red'], linewidth=3,
         label='Constraint: x² + y² = 2')

# Mark critical points
for pt in critical_points:
    f_val = objective(pt[0], pt[1])
    if f_val > 0:
        ax1.scatter([pt[0]], [pt[1]], color=COLORS['green'], s=150,
                   marker='*', edgecolor='black', linewidth=1, zorder=5)
    else:
        ax1.scatter([pt[0]], [pt[1]], color=COLORS['blue'], s=150,
                   marker='v', edgecolor='black', linewidth=1, zorder=5)

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Level curves of f(x,y) = xy\nwith constraint (red circle)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_aspect('equal')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Middle: Gradient alignment at critical point
ax2 = plt.subplot(1, 3, 2)

# Focus on one maximum point
pt = critical_points[0]  # (1, 1)
ax2.plot(constraint_x, constraint_y, color=COLORS['red'], linewidth=3,
         label='Constraint')

# Plot gradients at the critical point
grad_f = gradient_f(pt[0], pt[1])
grad_g = gradient_g(pt[0], pt[1])

# Normalize for visualization
grad_f_norm = grad_f / np.linalg.norm(grad_f) * 0.5
grad_g_norm = grad_g / np.linalg.norm(grad_g) * 0.5

ax2.quiver(pt[0], pt[1], grad_f_norm[0], grad_f_norm[1],
          angles='xy', scale_units='xy', scale=1,
          color=COLORS['blue'], width=0.015, label='∇f',
          headwidth=4, headlength=5)
ax2.quiver(pt[0], pt[1], grad_g_norm[0], grad_g_norm[1],
          angles='xy', scale_units='xy', scale=1,
          color=COLORS['orange'], width=0.015, label='∇g',
          headwidth=4, headlength=5)

ax2.scatter([pt[0]], [pt[1]], color=COLORS['green'], s=200,
           marker='*', edgecolor='black', linewidth=2, zorder=5)
ax2.text(pt[0]+0.2, pt[1]+0.2, f'Max at ({pt[0]}, {pt[1]})',
        fontsize=11, fontweight='bold')

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Gradient Alignment: ∇f = λ∇g\nat critical point', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_aspect('equal')
ax2.set_xlim(-0.5, 2)
ax2.set_ylim(-0.5, 2)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# Right: 3D surface with constraint
ax3 = plt.subplot(1, 3, 3, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')

# Plot constraint curve on surface
constraint_z = objective(constraint_x, constraint_y)
ax3.plot(constraint_x, constraint_y, constraint_z,
        color=COLORS['red'], linewidth=3, label='Constrained path')

# Mark critical points
for pt in critical_points:
    f_val = objective(pt[0], pt[1])
    if f_val > 0:
        ax3.scatter([pt[0]], [pt[1]], [f_val], color=COLORS['green'],
                   s=200, marker='*', edgecolor='black', linewidth=2)
    else:
        ax3.scatter([pt[0]], [pt[1]], [f_val], color=COLORS['blue'],
                   s=200, marker='v', edgecolor='black', linewidth=2)

ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('y', fontsize=10)
ax3.set_zlabel('f(x, y)', fontsize=10)
ax3.set_title('Surface with\nConstrained Extrema', fontsize=12)
ax3.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Key Insight:")
print("="*60)
print("At a constrained extremum, ∇f is parallel to ∇g")
print("(both perpendicular to the constraint curve)")
print()
print("This means: ∇f = λ∇g for some scalar λ (Lagrange multiplier)")
print()
print("✓ In GR: Used to find geodesics (extremal paths in spacetime)")
print("✓ Try: Minimize x² + y² subject to x + y = 1")
