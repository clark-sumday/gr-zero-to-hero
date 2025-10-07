#!/usr/bin/env python3
"""
Example: Divergence and Curl Visualization
Demonstrates vector field properties: sources/sinks and rotation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Vector field with positive divergence (source)
def radial_field(x, y):
    """F = (x, y) - radial outward"""
    return x, y

# Vector field with curl (rotation)
def rotational_field(x, y):
    """F = (-y, x) - counterclockwise rotation"""
    return -y, x

# Vector field with both divergence and curl
def mixed_field(x, y):
    """F = (x - y, x + y)"""
    return x - y, x + y

# Compute divergence: ∇·F = ∂F_x/∂x + ∂F_y/∂y
def divergence_radial(x, y):
    # F = (x, y), so ∂F_x/∂x = 1, ∂F_y/∂y = 1
    return 2 * np.ones_like(x)

def divergence_rotational(x, y):
    # F = (-y, x), so ∂F_x/∂x = 0, ∂F_y/∂y = 0
    return np.zeros_like(x)

def divergence_mixed(x, y):
    # F = (x - y, x + y), so ∂F_x/∂x = 1, ∂F_y/∂y = 1
    return 2 * np.ones_like(x)

# Compute curl (z-component): (∇×F)_z = ∂F_y/∂x - ∂F_x/∂y
def curl_radial(x, y):
    # F = (x, y), so ∂F_y/∂x = 0, ∂F_x/∂y = 0
    return np.zeros_like(x)

def curl_rotational(x, y):
    # F = (-y, x), so ∂F_y/∂x = 1, ∂F_x/∂y = -1
    return 2 * np.ones_like(x)

def curl_mixed(x, y):
    # F = (x - y, x + y), so ∂F_y/∂x = 1, ∂F_x/∂y = -1
    return 2 * np.ones_like(x)

print("Divergence and Curl Examples")
print("=" * 60)
print("\n1. Radial Field F = (x, y):")
print("   Divergence: ∇·F = 2 (positive everywhere - source)")
print("   Curl: (∇×F)_z = 0 (no rotation)")
print("\n2. Rotational Field F = (-y, x):")
print("   Divergence: ∇·F = 0 (incompressible)")
print("   Curl: (∇×F)_z = 2 (counterclockwise rotation)")
print("\n3. Mixed Field F = (x - y, x + y):")
print("   Divergence: ∇·F = 2 (has source)")
print("   Curl: (∇×F)_z = 2 (has rotation)")

# Create grid
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Compute fields
U1, V1 = radial_field(X, Y)
U2, V2 = rotational_field(X, Y)
U3, V3 = mixed_field(X, Y)

# Visualization
fig = plt.figure(figsize=(15, 10))

# Row 1: Vector fields
ax1 = plt.subplot(2, 3, 1)
ax1.quiver(X, Y, U1, V1, color=COLORS['blue'], alpha=0.7, width=0.003)
ax1.set_title('Radial Field F = (x, y)\n(Source)', fontsize=12, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_aspect('equal')

ax2 = plt.subplot(2, 3, 2)
ax2.quiver(X, Y, U2, V2, color=COLORS['orange'], alpha=0.7, width=0.003)
ax2.set_title('Rotational Field F = (-y, x)\n(Vortex)', fontsize=12, fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_aspect('equal')

ax3 = plt.subplot(2, 3, 3)
ax3.quiver(X, Y, U3, V3, color=COLORS['green'], alpha=0.7, width=0.003)
ax3.set_title('Mixed Field F = (x-y, x+y)\n(Source + Vortex)', fontsize=12, fontweight='bold')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_aspect('equal')

# Row 2: Divergence and Curl
ax4 = plt.subplot(2, 3, 4)
div1 = divergence_radial(X, Y)
im1 = ax4.contourf(X, Y, div1, levels=20, cmap='RdBu_r', alpha=0.7)
ax4.quiver(X[::2, ::2], Y[::2, ::2], U1[::2, ::2], V1[::2, ::2],
           color=COLORS['blue'], alpha=0.5, width=0.003)
plt.colorbar(im1, ax=ax4, label='Divergence')
ax4.set_title('∇·F = 2\n(Uniform expansion)', fontsize=11)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

ax5 = plt.subplot(2, 3, 5)
curl2 = curl_rotational(X, Y)
im2 = ax5.contourf(X, Y, curl2, levels=20, cmap='RdBu_r', alpha=0.7)
ax5.quiver(X[::2, ::2], Y[::2, ::2], U2[::2, ::2], V2[::2, ::2],
           color=COLORS['orange'], alpha=0.5, width=0.003)
plt.colorbar(im2, ax=ax5, label='Curl (z-component)')
ax5.set_title('(∇×F)_z = 2\n(Uniform rotation)', fontsize=11)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.grid(True, alpha=0.3)
ax5.set_aspect('equal')

ax6 = plt.subplot(2, 3, 6)
div3 = divergence_mixed(X, Y)
curl3 = curl_mixed(X, Y)
# Show both as overlays
im3 = ax6.contourf(X, Y, div3, levels=20, cmap='RdBu_r', alpha=0.4)
ax6.contour(X, Y, curl3, levels=10, colors='black', linewidths=0.5, alpha=0.6)
ax6.quiver(X[::2, ::2], Y[::2, ::2], U3[::2, ::2], V3[::2, ::2],
           color=COLORS['green'], alpha=0.5, width=0.003)
plt.colorbar(im3, ax=ax6, label='Divergence')
ax6.set_title('∇·F = 2, (∇×F)_z = 2\n(Both present)', fontsize=11)
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.grid(True, alpha=0.3)
ax6.set_aspect('equal')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Physical Interpretations:")
print("="*60)
print("Divergence > 0: Source (fluid flowing out)")
print("Divergence < 0: Sink (fluid flowing in)")
print("Divergence = 0: Incompressible flow")
print()
print("Curl > 0: Counterclockwise rotation")
print("Curl < 0: Clockwise rotation")
print("Curl = 0: Irrotational flow")
print()
print("✓ In GR: Divergence relates to spacetime curvature!")
print("✓ Try: Create your own vector field and compute div/curl")
