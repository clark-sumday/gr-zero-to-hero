#!/usr/bin/env python3
"""
Example: Multivariable Chain Rule
Demonstrates how to compute derivatives along parametric curves
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Scalar field: f(x, y) = x² + y²
def f(x, y):
    return x**2 + y**2

# Gradient: ∇f = (2x, 2y)
def gradient_f(x, y):
    return np.array([2*x, 2*y])

# Parametric curve: r(t) = (cos(t), sin(t)) [unit circle]
def curve(t):
    return np.array([np.cos(t), np.sin(t)])

# Velocity: r'(t) = (-sin(t), cos(t))
def velocity(t):
    return np.array([-np.sin(t), np.cos(t)])

# Function along curve: h(t) = f(r(t))
def f_along_curve(t):
    x, y = curve(t)
    return f(x, y)

# Derivative by chain rule: h'(t) = ∇f(r(t)) · r'(t)
def df_dt_chain_rule(t):
    pos = curve(t)
    grad = gradient_f(pos[0], pos[1])
    vel = velocity(t)
    return np.dot(grad, vel)

# Direct derivative: h(t) = cos²(t) + sin²(t) = 1 => h'(t) = 0
def df_dt_direct(t):
    return 0  # Since f is constant on the unit circle!

print("Multivariable Chain Rule Example")
print("=" * 60)
print("Function: f(x, y) = x² + y²")
print("Curve: r(t) = (cos(t), sin(t)) [unit circle]")
print()
print("Chain Rule: df/dt = ∇f · (dr/dt)")
print("  ∇f = (2x, 2y)")
print("  dr/dt = (-sin(t), cos(t))")
print()

# Test at specific points
test_times = [0, np.pi/4, np.pi/2, np.pi]
print("Verification at different points:")
print(f"{'t':>10} {'x,y':>15} {'∇f':>20} {'dr/dt':>20} {'df/dt':>10}")
print("-" * 80)
for t in test_times:
    pos = curve(t)
    grad = gradient_f(pos[0], pos[1])
    vel = velocity(t)
    deriv = df_dt_chain_rule(t)
    print(f"{t:10.3f} ({pos[0]:5.2f},{pos[1]:5.2f}) "
          f"({grad[0]:5.2f},{grad[1]:5.2f}) "
          f"({vel[0]:6.2f},{vel[1]:6.2f}) "
          f"{deriv:10.3f}")

print(f"\nNotice: df/dt = 0 everywhere (f is constant on the circle!)")

# Example 2: Curve that climbs the surface
print("\n" + "="*60)
print("Example 2: Spiral curve climbing the surface")
print("=" * 60)

def spiral(t):
    """r(t) = (t·cos(t), t·sin(t)) - outward spiral"""
    return np.array([t*np.cos(t), t*np.sin(t)])

def spiral_velocity(t):
    """dr/dt using product rule"""
    return np.array([np.cos(t) - t*np.sin(t), np.sin(t) + t*np.cos(t)])

def f_along_spiral(t):
    x, y = spiral(t)
    return f(x, y)

def df_dt_spiral(t):
    pos = spiral(t)
    grad = gradient_f(pos[0], pos[1])
    vel = spiral_velocity(t)
    return np.dot(grad, vel)

print("Curve: r(t) = (t·cos(t), t·sin(t))")
print("Chain Rule: df/dt = ∇f · (dr/dt)")
print()
print(f"{'t':>10} {'f(r(t))':>10} {'df/dt':>10}")
print("-" * 35)
for t in np.linspace(0.1, 2*np.pi, 5):
    pos = spiral(t)
    f_val = f(pos[0], pos[1])
    deriv = df_dt_spiral(t)
    print(f"{t:10.3f} {f_val:10.3f} {deriv:10.3f}")

# Visualization
fig = plt.figure(figsize=(15, 5))

# Left: Contour plot with both curves
ax1 = plt.subplot(1, 3, 1)
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

contour = ax1.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
ax1.clabel(contour, inline=True, fontsize=8)

# Plot unit circle (constant f)
t_circle = np.linspace(0, 2*np.pi, 100)
circle_points = np.array([curve(t) for t in t_circle])
ax1.plot(circle_points[:, 0], circle_points[:, 1],
        color=COLORS['red'], linewidth=3, label='Circle (df/dt = 0)')

# Plot spiral (increasing f)
t_spiral = np.linspace(0, 2*np.pi, 100)
spiral_points = np.array([spiral(t) for t in t_spiral])
ax1.plot(spiral_points[:, 0], spiral_points[:, 1],
        color=COLORS['blue'], linewidth=3, label='Spiral (df/dt > 0)')

# Show gradient and velocity at a point on circle
t_point = np.pi/4
pos = curve(t_point)
grad = gradient_f(pos[0], pos[1])
vel = velocity(t_point)

# Normalize for visualization
grad_norm = grad / np.linalg.norm(grad) * 0.3
vel_norm = vel / np.linalg.norm(vel) * 0.3

ax1.quiver(pos[0], pos[1], grad_norm[0], grad_norm[1],
          angles='xy', scale_units='xy', scale=1,
          color=COLORS['orange'], width=0.015, label='∇f (radial)')
ax1.quiver(pos[0], pos[1], vel_norm[0], vel_norm[1],
          angles='xy', scale_units='xy', scale=1,
          color=COLORS['green'], width=0.015, label='dr/dt (tangent)')
ax1.scatter([pos[0]], [pos[1]], color='black', s=100, zorder=5)

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Level Curves of f(x,y) = x² + y²', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.set_aspect('equal')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Middle: f(t) along both curves
ax2 = plt.subplot(1, 3, 2)

t_vals = np.linspace(0, 2*np.pi, 200)
f_circle = [f_along_curve(t) for t in t_vals]
f_spiral_vals = [f_along_spiral(t) for t in t_vals]

ax2.plot(t_vals, f_circle, color=COLORS['red'], linewidth=2,
        label='Circle: f(t) = 1 (constant)')
ax2.plot(t_vals, f_spiral_vals, color=COLORS['blue'], linewidth=2,
        label='Spiral: f(t) = t² (increasing)')
ax2.set_xlabel('t', fontsize=12)
ax2.set_ylabel('f(r(t))', fontsize=12)
ax2.set_title('Function Value Along Curves', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Right: df/dt along both curves
ax3 = plt.subplot(1, 3, 3)

df_circle = [df_dt_chain_rule(t) for t in t_vals]
df_spiral_vals = [df_dt_spiral(t) for t in t_vals]

ax3.plot(t_vals, df_circle, color=COLORS['red'], linewidth=2,
        label='Circle: df/dt = 0')
ax3.plot(t_vals, df_spiral_vals, color=COLORS['blue'], linewidth=2,
        label='Spiral: df/dt = ∇f · (dr/dt)')
ax3.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
ax3.set_xlabel('t', fontsize=12)
ax3.set_ylabel('df/dt', fontsize=12)
ax3.set_title('Rate of Change Along Curves\n(Chain Rule)', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Key Insights:")
print("="*60)
print("1. df/dt = ∇f · (dr/dt) - dot product of gradient and velocity")
print("2. When perpendicular: df/dt = 0 (moving along level curve)")
print("3. When parallel: df/dt = maximum (steepest ascent)")
print()
print("✓ In GR: This generalizes to covariant derivatives along curves!")
print("✓ Try: Use curve r(t) = (t, t²) - parabola")
