#!/usr/bin/env python3
"""
Example: Phase Portraits for Different Systems
Demonstrates stable/unstable equilibria, saddles, spirals, and centers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Different 2D systems with different types of equilibria

def stable_node(state, t):
    """
    dx/dt = -x
    dy/dt = -2y
    Stable node at origin
    """
    x, y = state
    return [-x, -2*y]

def unstable_node(state, t):
    """
    dx/dt = x
    dy/dt = 2y
    Unstable node at origin
    """
    x, y = state
    return [x, 2*y]

def saddle_point(state, t):
    """
    dx/dt = x
    dy/dt = -y
    Saddle point at origin
    """
    x, y = state
    return [x, -y]

def center(state, t):
    """
    dx/dt = -y
    dy/dt = x
    Center at origin (periodic orbits)
    """
    x, y = state
    return [-y, x]

def stable_spiral(state, t):
    """
    dx/dt = -x - y
    dy/dt = x - y
    Stable spiral at origin
    """
    x, y = state
    return [-x - y, x - y]

def unstable_spiral(state, t):
    """
    dx/dt = x - y
    dy/dt = x + y
    Unstable spiral at origin
    """
    x, y = state
    return [x - y, x + y]

# Time array
t = np.linspace(0, 5, 500)

# Create grid for vector fields
x_range = np.linspace(-3, 3, 20)
y_range = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x_range, y_range)

# Initial conditions for trajectories
angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
radius = 2.0
initial_conditions = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]

print("Phase Portrait Classification")
print("=" * 60)
print("\nTypes of equilibrium points:")
print("  1. Stable Node: All trajectories approach equilibrium")
print("  2. Unstable Node: All trajectories move away from equilibrium")
print("  3. Saddle: Stable in one direction, unstable in another")
print("  4. Center: Periodic orbits around equilibrium")
print("  5. Stable Spiral: Trajectories spiral into equilibrium")
print("  6. Unstable Spiral: Trajectories spiral away from equilibrium")

# Visualization
fig = plt.figure(figsize=(18, 12))

# System 1: Stable Node
ax1 = plt.subplot(2, 3, 1)
U, V = -X, -2*Y
magnitude = np.sqrt(U**2 + V**2)
ax1.quiver(X, Y, U/(magnitude+1e-10), V/(magnitude+1e-10),
          color=COLORS['gray'], alpha=0.4, width=0.003)

for ic in initial_conditions[:6]:
    sol = odeint(stable_node, ic, t)
    ax1.plot(sol[:, 0], sol[:, 1], color=COLORS['blue'], linewidth=1.5, alpha=0.7)
    ax1.scatter([ic[0]], [ic[1]], color=COLORS['blue'], s=30, zorder=5)

ax1.scatter([0], [0], color=COLORS['red'], s=200, marker='o',
           edgecolor='black', linewidth=2, zorder=6)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.set_title('Stable Node\ndx/dt = -x, dy/dt = -2y', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_aspect('equal')

# System 2: Unstable Node
ax2 = plt.subplot(2, 3, 2)
U, V = X, 2*Y
magnitude = np.sqrt(U**2 + V**2)
ax2.quiver(X, Y, U/(magnitude+1e-10), V/(magnitude+1e-10),
          color=COLORS['gray'], alpha=0.4, width=0.003)

# Use shorter time and smaller initial conditions
t_short = np.linspace(0, 2, 300)
ic_small = [(0.5*np.cos(a), 0.5*np.sin(a)) for a in angles]

for ic in ic_small[:6]:
    sol = odeint(unstable_node, ic, t_short)
    ax2.plot(sol[:, 0], sol[:, 1], color=COLORS['orange'], linewidth=1.5, alpha=0.7)
    ax2.scatter([ic[0]], [ic[1]], color=COLORS['orange'], s=30, zorder=5)

ax2.scatter([0], [0], color=COLORS['red'], s=200, marker='o',
           edgecolor='black', linewidth=2, zorder=6)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y', fontsize=11)
ax2.set_title('Unstable Node\ndx/dt = x, dy/dt = 2y', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_aspect('equal')

# System 3: Saddle Point
ax3 = plt.subplot(2, 3, 3)
U, V = X, -Y
magnitude = np.sqrt(U**2 + V**2)
ax3.quiver(X, Y, U/(magnitude+1e-10), V/(magnitude+1e-10),
          color=COLORS['gray'], alpha=0.4, width=0.003)

# Special initial conditions for saddle
ic_saddle = [(2, 0.5), (2, -0.5), (-2, 0.5), (-2, -0.5),
             (0.5, 2), (-0.5, 2), (0.5, -2), (-0.5, -2)]

for ic in ic_saddle:
    sol = odeint(saddle_point, ic, t_short)
    ax3.plot(sol[:, 0], sol[:, 1], color=COLORS['green'], linewidth=1.5, alpha=0.7)
    ax3.scatter([ic[0]], [ic[1]], color=COLORS['green'], s=30, zorder=5)

# Draw stable/unstable manifolds
ax3.plot([-3, 3], [0, 0], color=COLORS['red'], linewidth=2,
        linestyle='--', label='Unstable manifold')
ax3.plot([0, 0], [-3, 3], color=COLORS['blue'], linewidth=2,
        linestyle='--', label='Stable manifold')

ax3.scatter([0], [0], color=COLORS['red'], s=200, marker='X',
           edgecolor='black', linewidth=2, zorder=6)
ax3.set_xlabel('x', fontsize=11)
ax3.set_ylabel('y', fontsize=11)
ax3.set_title('Saddle Point\ndx/dt = x, dy/dt = -y', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)
ax3.set_xlim(-3, 3)
ax3.set_ylim(-3, 3)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_aspect('equal')

# System 4: Center
ax4 = plt.subplot(2, 3, 4)
U, V = -Y, X
magnitude = np.sqrt(U**2 + V**2)
ax4.quiver(X, Y, U/(magnitude+1e-10), V/(magnitude+1e-10),
          color=COLORS['gray'], alpha=0.4, width=0.003)

radii = [0.5, 1.0, 1.5, 2.0, 2.5]
for r in radii:
    ic = (r, 0)
    sol = odeint(center, ic, np.linspace(0, 6.3, 200))
    ax4.plot(sol[:, 0], sol[:, 1], color=COLORS['purple'], linewidth=1.5, alpha=0.7)

ax4.scatter([0], [0], color=COLORS['red'], s=200, marker='o',
           edgecolor='black', linewidth=2, zorder=6)
ax4.set_xlabel('x', fontsize=11)
ax4.set_ylabel('y', fontsize=11)
ax4.set_title('Center\ndx/dt = -y, dy/dt = x', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-3, 3)
ax4.set_ylim(-3, 3)
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)
ax4.set_aspect('equal')

# System 5: Stable Spiral
ax5 = plt.subplot(2, 3, 5)
U, V = -X - Y, X - Y
magnitude = np.sqrt(U**2 + V**2)
ax5.quiver(X, Y, U/(magnitude+1e-10), V/(magnitude+1e-10),
          color=COLORS['gray'], alpha=0.4, width=0.003)

t_spiral = np.linspace(0, 8, 800)
for ic in initial_conditions[:4]:
    sol = odeint(stable_spiral, ic, t_spiral)
    ax5.plot(sol[:, 0], sol[:, 1], color=COLORS['cyan'], linewidth=1.5, alpha=0.7)
    ax5.scatter([ic[0]], [ic[1]], color=COLORS['cyan'], s=30, zorder=5)

ax5.scatter([0], [0], color=COLORS['red'], s=200, marker='o',
           edgecolor='black', linewidth=2, zorder=6)
ax5.set_xlabel('x', fontsize=11)
ax5.set_ylabel('y', fontsize=11)
ax5.set_title('Stable Spiral\ndx/dt = -x - y, dy/dt = x - y', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(-3, 3)
ax5.set_ylim(-3, 3)
ax5.axhline(y=0, color='k', linewidth=0.5)
ax5.axvline(x=0, color='k', linewidth=0.5)
ax5.set_aspect('equal')

# System 6: Unstable Spiral
ax6 = plt.subplot(2, 3, 6)
U, V = X - Y, X + Y
magnitude = np.sqrt(U**2 + V**2)
ax6.quiver(X, Y, U/(magnitude+1e-10), V/(magnitude+1e-10),
          color=COLORS['gray'], alpha=0.4, width=0.003)

t_spiral_short = np.linspace(0, 3, 500)
for ic in ic_small[:4]:
    sol = odeint(unstable_spiral, ic, t_spiral_short)
    ax6.plot(sol[:, 0], sol[:, 1], color=COLORS['pink'], linewidth=1.5, alpha=0.7)
    ax6.scatter([ic[0]], [ic[1]], color=COLORS['pink'], s=30, zorder=5)

ax6.scatter([0], [0], color=COLORS['red'], s=200, marker='o',
           edgecolor='black', linewidth=2, zorder=6)
ax6.set_xlabel('x', fontsize=11)
ax6.set_ylabel('y', fontsize=11)
ax6.set_title('Unstable Spiral\ndx/dt = x - y, dy/dt = x + y', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(-3, 3)
ax6.set_ylim(-3, 3)
ax6.axhline(y=0, color='k', linewidth=0.5)
ax6.axvline(x=0, color='k', linewidth=0.5)
ax6.set_aspect('equal')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Classification by Linear Stability Analysis:")
print("="*60)
print("\nFor system dx/dt = f(x,y), dy/dt = g(x,y):")
print("Jacobian matrix at equilibrium:")
print("    J = | ∂f/∂x  ∂f/∂y |")
print("        | ∂g/∂x  ∂g/∂y |")
print()
print("Eigenvalues λ₁, λ₂ determine behavior:")
print("  • Both negative real: Stable node")
print("  • Both positive real: Unstable node")
print("  • Opposite signs: Saddle point")
print("  • Complex with negative real part: Stable spiral")
print("  • Complex with positive real part: Unstable spiral")
print("  • Pure imaginary: Center")
print()
print("✓ In GR: Geodesic deviation analyzed via stability!")
print("✓ Try: Modify the systems to create new phase portraits")
