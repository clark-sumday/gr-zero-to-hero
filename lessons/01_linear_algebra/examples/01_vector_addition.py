#!/usr/bin/env python3
"""
Example: Vector Addition Visualization
Demonstrates tip-to-tail method and parallelogram construction
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Define two vectors with distinct directions
u = np.array([3, 1])
v = np.array([1, 3])

print(f"u = {u}")
print(f"v = {v}")
print(f"u + v = {u + v}")
print(f"|u| = {np.linalg.norm(u):.2f}")
print(f"|v| = {np.linalg.norm(v):.2f}")
print(f"|u + v| = {np.linalg.norm(u + v):.2f}")

# Visualization
plt.figure(figsize=(10, 5))

# Left: Standard view (all from origin)
plt.subplot(1, 2, 1)
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
           color=COLORS['blue'], width=0.006, label='u')
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color=COLORS['orange'], width=0.006, label='v')
plt.quiver(0, 0, (u+v)[0], (u+v)[1], angles='xy', scale_units='xy', scale=1,
           color=COLORS['green'], width=0.006, label='u+v')
plt.text(u[0], u[1], ' u', fontsize=12, color=COLORS['blue'])
plt.text(v[0], v[1], ' v', fontsize=12, color=COLORS['orange'])
plt.text((u+v)[0], (u+v)[1], ' u+v', fontsize=12, color=COLORS['green'])
plt.xlim(-1, 5)
plt.ylim(-1, 7)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Vectors from Origin")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.gca().set_aspect('equal')

# Right: Tip-to-tail construction
plt.subplot(1, 2, 2)
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
           color=COLORS['blue'], width=0.006, label='u')
plt.quiver(u[0], u[1], v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color=COLORS['orange'], width=0.006, label='v (from u tip)')
plt.quiver(0, 0, (u+v)[0], (u+v)[1], angles='xy', scale_units='xy', scale=1,
           color=COLORS['green'], width=0.006, label='u+v')
plt.plot([0, v[0]], [0, v[1]], '--', color=COLORS['gray'], alpha=0.5)
plt.plot([u[0], (u+v)[0]], [u[1], (u+v)[1]], '--', color=COLORS['gray'], alpha=0.5)
plt.text(u[0]/2, u[1]/2, ' u', fontsize=12, color=COLORS['blue'])
plt.text(u[0] + v[0]/2, u[1] + v[1]/2, ' v', fontsize=12, color=COLORS['orange'])
plt.text((u+v)[0], (u+v)[1], ' u+v', fontsize=12, color=COLORS['green'])
plt.xlim(-1, 5)
plt.ylim(-1, 7)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Tip-to-Tail (Parallelogram) Method")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.gca().set_aspect('equal')

plt.tight_layout()
plt.show()

print("\n✓ Try changing u and v to see different vector additions!")
