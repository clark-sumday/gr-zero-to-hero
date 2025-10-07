#!/usr/bin/env python3
"""
Example: Surface Curvature (Gaussian and Mean)
Demonstrates how surfaces bend in 3D space
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

def sphere(u, v, R=1.0):
    """
    Sphere of radius R
    Gaussian curvature K = 1/R²
    Mean curvature H = 1/R
    """
    x = R * np.sin(u) * np.cos(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(u)
    return x, y, z

def cylinder(u, v, R=1.0):
    """
    Cylinder of radius R
    Gaussian curvature K = 0 (flat!)
    Mean curvature H = 1/(2R)
    """
    x = R * np.cos(u)
    y = R * np.sin(u)
    z = v
    return x, y, z

def saddle(u, v, a=1.0):
    """
    Hyperbolic paraboloid z = x²/a - y²/a
    Gaussian curvature K < 0 (negative)
    """
    x = u
    y = v
    z = (u**2 - v**2) / a
    return x, y, z

def torus(u, v, R=2.0, r=0.5):
    """
    Torus with major radius R and minor radius r
    Gaussian curvature K = cos(u) / (r(R + r·cos(u)))
    (varies: positive on outer side, negative on inner side)
    """
    x = (R + r*np.cos(u)) * np.cos(v)
    y = (R + r*np.cos(u)) * np.sin(v)
    z = r * np.sin(u)
    return x, y, z

def gaussian_curvature_saddle(u, v, a=1.0):
    """
    Analytical Gaussian curvature for hyperbolic paraboloid
    K = -4/(a²(1 + 4u²/a² + 4v²/a²)²)
    """
    denominator = a**2 * (1 + 4*u**2/a**2 + 4*v**2/a**2)**2
    return -4 / denominator

print("Surface Curvature")
print("=" * 60)
print("Surfaces can bend in two independent directions:")
print()
print("Principal curvatures κ₁, κ₂:")
print("  Maximum and minimum normal curvatures at a point")
print()
print("Gaussian curvature K = κ₁ · κ₂:")
print("  • K > 0: Sphere-like (both directions curve same way)")
print("  • K = 0: Flat or cylindrical (one direction flat)")
print("  • K < 0: Saddle-like (directions curve opposite ways)")
print()
print("Mean curvature H = (κ₁ + κ₂)/2:")
print("  • Average curvature")
print("  • Related to surface area minimization")
print()
print("Examples:")
print("  Sphere (R=1):    K = 1,  H = 1")
print("  Cylinder (R=1):  K = 0,  H = 1/2")
print("  Saddle:          K < 0,  H ≈ 0")
print("  Plane:           K = 0,  H = 0")

# Create parameter grids
u = np.linspace(0, np.pi, 50)
v = np.linspace(0, 2*np.pi, 50)
U, V = np.meshgrid(u, v)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Sphere (K > 0)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
X_sphere, Y_sphere, Z_sphere = sphere(U, V, R=1.0)
surf1 = ax1.plot_surface(X_sphere, Y_sphere, Z_sphere, cmap='viridis',
                         alpha=0.8, edgecolor='none')
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.set_zlabel('z', fontsize=10)
ax1.set_title('Sphere\nK = 1 (positive)\nH = 1', fontsize=12, fontweight='bold')
ax1.set_box_aspect([1,1,1])

# Plot 2: Cylinder (K = 0)
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
u_cyl = np.linspace(0, 2*np.pi, 50)
v_cyl = np.linspace(-2, 2, 50)
U_cyl, V_cyl = np.meshgrid(u_cyl, v_cyl)
X_cyl, Y_cyl, Z_cyl = cylinder(U_cyl, V_cyl, R=1.0)
surf2 = ax2.plot_surface(X_cyl, Y_cyl, Z_cyl, cmap='plasma',
                         alpha=0.8, edgecolor='none')
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.set_zlabel('z', fontsize=10)
ax2.set_title('Cylinder\nK = 0 (flat!)\nH = 1/2', fontsize=12, fontweight='bold')
ax2.set_box_aspect([1,1,1])

# Plot 3: Saddle (K < 0)
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
u_sad = np.linspace(-2, 2, 50)
v_sad = np.linspace(-2, 2, 50)
U_sad, V_sad = np.meshgrid(u_sad, v_sad)
X_sad, Y_sad, Z_sad = saddle(U_sad, V_sad, a=1.0)
surf3 = ax3.plot_surface(X_sad, Y_sad, Z_sad, cmap='coolwarm',
                         alpha=0.8, edgecolor='none')
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('y', fontsize=10)
ax3.set_zlabel('z', fontsize=10)
ax3.set_title('Hyperbolic Paraboloid\nK < 0 (negative)\nH ≈ 0', fontsize=12, fontweight='bold')
ax3.set_box_aspect([1,1,1])

# Plot 4: Torus (K varies)
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
u_tor = np.linspace(0, 2*np.pi, 50)
v_tor = np.linspace(0, 2*np.pi, 50)
U_tor, V_tor = np.meshgrid(u_tor, v_tor)
X_tor, Y_tor, Z_tor = torus(U_tor, V_tor, R=2.0, r=0.5)

# Compute approximate Gaussian curvature for torus
R, r = 2.0, 0.5
K_tor = np.cos(U_tor) / (r * (R + r*np.cos(U_tor)))

surf4 = ax4.plot_surface(X_tor, Y_tor, Z_tor, facecolors=plt.cm.RdBu(
                        (K_tor - K_tor.min())/(K_tor.max() - K_tor.min())),
                        alpha=0.8, edgecolor='none')
ax4.set_xlabel('x', fontsize=10)
ax4.set_ylabel('y', fontsize=10)
ax4.set_zlabel('z', fontsize=10)
ax4.set_title('Torus\nK varies (+ outside, - inside)', fontsize=12, fontweight='bold')
ax4.set_box_aspect([1,1,1])

# Plot 5: Gaussian curvature distribution for saddle
ax5 = plt.subplot(2, 3, 5)
K_saddle = gaussian_curvature_saddle(U_sad, V_sad, a=1.0)
contour = ax5.contourf(U_sad, V_sad, K_saddle, levels=20, cmap='coolwarm')
plt.colorbar(contour, ax=ax5, label='K')
ax5.set_xlabel('u', fontsize=12)
ax5.set_ylabel('v', fontsize=12)
ax5.set_title('Gaussian Curvature of Saddle\n(all negative)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_aspect('equal')

# Plot 6: Comparison of principal curvatures
ax6 = plt.subplot(2, 3, 6)

# Create schematic comparison
surfaces = ['Sphere', 'Cylinder', 'Saddle', 'Plane']
kappa1 = [1, 1, 1, 0]
kappa2 = [1, 0, -1, 0]
K_values = [k1*k2 for k1, k2 in zip(kappa1, kappa2)]
H_values = [(k1+k2)/2 for k1, k2 in zip(kappa1, kappa2)]

x_pos = np.arange(len(surfaces))
width = 0.35

bars1 = ax6.bar(x_pos - width/2, kappa1, width, label='κ₁',
               color=COLORS['blue'], alpha=0.7)
bars2 = ax6.bar(x_pos + width/2, kappa2, width, label='κ₂',
               color=COLORS['orange'], alpha=0.7)

# Add K values as text
for i, (K, H) in enumerate(zip(K_values, H_values)):
    ax6.text(i, max(kappa1[i], kappa2[i]) + 0.2,
            f'K={K:.1f}\nH={H:.1f}',
            ha='center', fontsize=10, fontweight='bold')

ax6.set_xlabel('Surface Type', fontsize=12)
ax6.set_ylabel('Principal Curvatures', fontsize=12)
ax6.set_title('Principal Curvatures Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(surfaces)
ax6.axhline(y=0, color='k', linewidth=0.5)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Gauss's Theorema Egregium:")
print("="*60)
print("Gaussian curvature K is *intrinsic*:")
print("  • Can be computed using only distances on the surface")
print("  • Doesn't depend on how surface is embedded in 3D")
print("  • Preserved under isometric deformations (bending without stretching)")
print()
print("Example: A cylinder can be 'unrolled' into a flat plane")
print("  → Both have K = 0 (flat intrinsically)")
print()
print("But a sphere CANNOT be flattened without distortion")
print("  → K = 1/R² ≠ 0 (curved intrinsically)")
print()
print("Mean curvature H is *extrinsic*:")
print("  • Depends on embedding in 3D space")
print("  • Changes when you bend the surface")
print()
print("✓ In GR: Spacetime has intrinsic curvature K!")
print("✓ This is what the Einstein field equations describe!")
print("✓ Try: Visualize how K changes for different surfaces")
