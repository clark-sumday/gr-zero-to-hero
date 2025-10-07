#!/usr/bin/env python3
"""
Example: Action Principle - Path Integrals and Variational Calculus
Demonstrates how nature chooses paths that extremize the action
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("ACTION PRINCIPLE: NATURE'S OPTIMIZATION")
print("="*60)

print("\nAction: S[q] = ∫ L(q, q̇, t) dt")
print("Principle: δS = 0 (action is stationary)")
print("Result: Euler-Lagrange equations")

# Example: Projectile motion
g = 9.8
print(f"\nProjectile under gravity (g={g} m/s²)")
print("L = ½m(ẋ² + ẏ²) - mgy")

# True trajectory (parabola)
v0_x, v0_y = 10.0, 15.0
x0, y0 = 0.0, 0.0
t_flight = 2 * v0_y / g
t = np.linspace(0, t_flight, 100)

x_true = x0 + v0_x * t
y_true = y0 + v0_y * t - 0.5 * g * t**2

# Compute action
def compute_action(x, y, t, m=1.0, g=9.8):
    """Compute action for a path"""
    dt = t[1] - t[0]
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    L = 0.5 * m * (vx**2 + vy**2) - m * g * y
    return np.trapz(L, t)

S_true = compute_action(x_true, y_true, t)

# Varied paths
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot paths
axes[0].plot(x_true, y_true, color=COLORS['green'], linewidth=3, label=f'True path (S={S_true:.2f})')

S_values = [S_true]
for amp in [0.5, 1.0, 2.0]:
    y_varied = y_true + amp * np.sin(np.pi * t / t_flight)
    S_var = compute_action(x_true, y_varied, t)
    S_values.append(S_var)
    axes[0].plot(x_true, y_varied, '--', linewidth=2, alpha=0.7, label=f'Varied (S={S_var:.2f})')

axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].set_title('Projectile Paths\n(True path minimizes action)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot actions
axes[1].bar(range(len(S_values)), S_values,
           color=[COLORS['green']] + [COLORS['blue']]*3,
           edgecolor='black', linewidth=2)
axes[1].set_xticks(range(len(S_values)))
axes[1].set_xticklabels(['True', 'Var 1', 'Var 2', 'Var 3'])
axes[1].set_ylabel('Action S')
axes[1].set_title('Action Comparison')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n✓ Hamilton's principle: Foundation of modern physics!")
