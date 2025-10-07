#!/usr/bin/env python3
"""
Example: Tangent Spaces and Tangent Vectors
Demonstrates the tangent space T_p M at each point p on a manifold
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

def sphere_surface(u, v, R=1.0):
    """Parametric sphere"""
    x = R * np.sin(u) * np.cos(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(u)
    return x, y, z

def sphere_tangent_vectors(u, v, R=1.0):
    """
    Compute basis vectors for tangent space at point (u,v)
    ∂r/∂u and ∂r/∂v form a basis for T_p S²
    """
    # ∂r/∂u
    du_x = R * np.cos(u) * np.cos(v)
    du_y = R * np.cos(u) * np.sin(v)
    du_z = -R * np.sin(u)

    # ∂r/∂v
    dv_x = -R * np.sin(u) * np.sin(v)
    dv_y = R * np.sin(u) * np.cos(v)
    dv_z = 0

    return np.array([du_x, du_y, du_z]), np.array([dv_x, dv_y, dv_z])

def normal_vector(u, v, R=1.0):
    """
    Normal vector to sphere at point (u,v)
    Perpendicular to tangent space
    """
    # For sphere, normal is just radial direction
    x, y, z = sphere_surface(u, v, R)
    norm = np.sqrt(x**2 + y**2 + z**2)
    return np.array([x/norm, y/norm, z/norm])

def torus_surface(u, v, R=2.0, r=0.5):
    """Parametric torus"""
    x = (R + r*np.cos(u)) * np.cos(v)
    y = (R + r*np.cos(u)) * np.sin(v)
    z = r * np.sin(u)
    return x, y, z

def torus_tangent_vectors(u, v, R=2.0, r=0.5):
    """Tangent basis for torus"""
    # ∂r/∂u
    du_x = -r * np.sin(u) * np.cos(v)
    du_y = -r * np.sin(u) * np.sin(v)
    du_z = r * np.cos(u)

    # ∂r/∂v
    dv_x = -(R + r*np.cos(u)) * np.sin(v)
    dv_y = (R + r*np.cos(u)) * np.cos(v)
    dv_z = 0

    return np.array([du_x, du_y, du_z]), np.array([dv_x, dv_y, dv_z])

print("Tangent Spaces on Manifolds")
print("=" * 60)
print("At each point p on a manifold M, there is a tangent space T_p M")
print()
print("Tangent space T_p M:")
print("  • Vector space of all tangent vectors at p")
print("  • Dimension = dimension of manifold")
print("  • Different for each point p!")
print()
print("For a surface in ℝ³:")
print("  • Tangent space is a 2D plane touching the surface at p")
print("  • Tangent vectors are derivatives of curves through p")
print()
print("Coordinate basis:")
print("  If manifold has coordinates (u, v), then")
print("  {∂/∂u, ∂/∂v} form a basis for T_p M")
print()
print("Example: Sphere S²")
print("  • At each point: T_p S² is a 2D plane")
print("  • Tangent plane perpendicular to radial direction")
print("  • Basis vectors: ∂r/∂θ and ∂r/∂φ")

# Create sphere
u_sph = np.linspace(0.1, np.pi-0.1, 30)
v_sph = np.linspace(0, 2*np.pi, 30)
U_sph, V_sph = np.meshgrid(u_sph, v_sph)
X_sph, Y_sph, Z_sph = sphere_surface(U_sph, V_sph, R=1.0)

# Create torus
u_tor = np.linspace(0, 2*np.pi, 30)
v_tor = np.linspace(0, 2*np.pi, 30)
U_tor, V_tor = np.meshgrid(u_tor, v_tor)
X_tor, Y_tor, Z_tor = torus_surface(U_tor, V_tor, R=2.0, r=0.5)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Tangent plane to sphere at a point
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot_surface(X_sph, Y_sph, Z_sph, alpha=0.3, color=COLORS['gray'])

# Choose a point
u_point = np.pi/3
v_point = np.pi/4
x_p, y_p, z_p = sphere_surface(u_point, v_point)

# Get tangent vectors
e_u, e_v = sphere_tangent_vectors(u_point, v_point)

# Scale for visualization
scale = 0.5
e_u_scaled = e_u * scale
e_v_scaled = e_v * scale

# Draw tangent plane
t1 = np.linspace(-0.8, 0.8, 10)
t2 = np.linspace(-0.8, 0.8, 10)
T1, T2 = np.meshgrid(t1, t2)
plane_x = x_p + T1 * e_u[0] + T2 * e_v[0]
plane_y = y_p + T1 * e_u[1] + T2 * e_v[1]
plane_z = z_p + T1 * e_u[2] + T2 * e_v[2]
ax1.plot_surface(plane_x, plane_y, plane_z, alpha=0.5, color=COLORS['yellow'])

# Draw basis vectors
ax1.quiver(x_p, y_p, z_p, e_u_scaled[0], e_u_scaled[1], e_u_scaled[2],
          color=COLORS['red'], arrow_length_ratio=0.3, linewidth=3, label='∂r/∂u')
ax1.quiver(x_p, y_p, z_p, e_v_scaled[0], e_v_scaled[1], e_v_scaled[2],
          color=COLORS['blue'], arrow_length_ratio=0.3, linewidth=3, label='∂r/∂v')

# Draw normal vector
n = normal_vector(u_point, v_point)
n_scaled = n * scale
ax1.quiver(x_p, y_p, z_p, n_scaled[0], n_scaled[1], n_scaled[2],
          color=COLORS['green'], arrow_length_ratio=0.3, linewidth=3, label='normal')

ax1.scatter([x_p], [y_p], [z_p], color='black', s=100, zorder=5)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.set_zlabel('z', fontsize=10)
ax1.set_title('Tangent Space T_p S²\n(plane tangent to sphere)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_box_aspect([1,1,1])

# Plot 2: Multiple tangent spaces on sphere
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot_surface(X_sph, Y_sph, Z_sph, alpha=0.2, color=COLORS['gray'])

# Show tangent spaces at several points
points = [(np.pi/4, 0), (np.pi/4, np.pi/2), (np.pi/2, np.pi/4),
          (3*np.pi/4, np.pi), (np.pi/2, 3*np.pi/2)]
colors_points = [COLORS['red'], COLORS['blue'], COLORS['green'],
                COLORS['orange'], COLORS['purple']]

for (up, vp), color in zip(points, colors_points):
    xp, yp, zp = sphere_surface(up, vp)
    e_u, e_v = sphere_tangent_vectors(up, vp)

    # Draw basis vectors
    scale_small = 0.3
    ax2.quiver(xp, yp, zp, e_u[0]*scale_small, e_u[1]*scale_small, e_u[2]*scale_small,
              color=color, arrow_length_ratio=0.3, linewidth=2, alpha=0.7)
    ax2.quiver(xp, yp, zp, e_v[0]*scale_small, e_v[1]*scale_small, e_v[2]*scale_small,
              color=color, arrow_length_ratio=0.3, linewidth=2, alpha=0.7)

    ax2.scatter([xp], [yp], [zp], color=color, s=80, zorder=5)

ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.set_zlabel('z', fontsize=10)
ax2.set_title('Tangent Spaces at Different Points\n(each point has its own T_p M)', fontsize=12, fontweight='bold')
ax2.set_box_aspect([1,1,1])

# Plot 3: Tangent vector as derivative of curve
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.plot_surface(X_sph, Y_sph, Z_sph, alpha=0.3, color=COLORS['gray'])

# Define a curve on the sphere: longitude line at φ = π/4
t = np.linspace(np.pi/6, 5*np.pi/6, 50)
u_curve = t
v_curve = np.pi/4 * np.ones_like(t)
x_curve, y_curve, z_curve = sphere_surface(u_curve, v_curve)

ax3.plot(x_curve, y_curve, z_curve, color=COLORS['blue'], linewidth=3, label='Curve γ(t)')

# Point on curve
t_point = np.pi/3
idx = np.argmin(np.abs(u_curve - t_point))
xc = x_curve[idx]
yc = y_curve[idx]
zc = z_curve[idx]

# Tangent vector = derivative of curve
e_u, e_v = sphere_tangent_vectors(t_point, v_curve[idx])
# γ'(t) = (∂r/∂u)·(du/dt) + (∂r/∂v)·(dv/dt)
# For our curve: du/dt = 1, dv/dt = 0
tangent = e_u * 0.5

ax3.quiver(xc, yc, zc, tangent[0], tangent[1], tangent[2],
          color=COLORS['red'], arrow_length_ratio=0.3, linewidth=3,
          label="γ'(t) ∈ T_p M")
ax3.scatter([xc], [yc], [zc], color='black', s=100, zorder=5)

ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('y', fontsize=10)
ax3.set_zlabel('z', fontsize=10)
ax3.set_title('Tangent Vector as\nDerivative of Curve', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.set_box_aspect([1,1,1])

# Plot 4: Tangent spaces on torus
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.plot_surface(X_tor, Y_tor, Z_tor, alpha=0.3, color=COLORS['gray'])

# Show tangent space at a point
u_t = np.pi/3
v_t = np.pi/4
x_t, y_t, z_t = torus_surface(u_t, v_t)
e_u_t, e_v_t = torus_tangent_vectors(u_t, v_t)

# Draw tangent plane
scale_t = 0.4
t1_t = np.linspace(-1, 1, 10)
t2_t = np.linspace(-1, 1, 10)
T1_t, T2_t = np.meshgrid(t1_t, t2_t)
plane_x_t = x_t + T1_t * e_u_t[0] * scale_t + T2_t * e_v_t[0] * scale_t
plane_y_t = y_t + T1_t * e_u_t[1] * scale_t + T2_t * e_v_t[1] * scale_t
plane_z_t = z_t + T1_t * e_u_t[2] * scale_t + T2_t * e_v_t[2] * scale_t
ax4.plot_surface(plane_x_t, plane_y_t, plane_z_t, alpha=0.5, color=COLORS['yellow'])

# Draw basis vectors
ax4.quiver(x_t, y_t, z_t, e_u_t[0]*scale_t, e_u_t[1]*scale_t, e_u_t[2]*scale_t,
          color=COLORS['red'], arrow_length_ratio=0.3, linewidth=3, label='∂r/∂u')
ax4.quiver(x_t, y_t, z_t, e_v_t[0]*scale_t, e_v_t[1]*scale_t, e_v_t[2]*scale_t,
          color=COLORS['blue'], arrow_length_ratio=0.3, linewidth=3, label='∂r/∂v')

ax4.scatter([x_t], [y_t], [z_t], color='black', s=100, zorder=5)
ax4.set_xlabel('x', fontsize=10)
ax4.set_ylabel('y', fontsize=10)
ax4.set_zlabel('z', fontsize=10)
ax4.set_title('Tangent Space on Torus', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.set_box_aspect([1,1,1])

# Plot 5: Basis vector magnitudes (metric)
ax5 = plt.subplot(2, 3, 5)

# Compute metric components g_ij = e_i · e_j for sphere
u_vals = np.linspace(0.1, np.pi-0.1, 50)
v_val = np.pi/4

g_uu = []
g_uv = []
g_vv = []

for u_val in u_vals:
    e_u, e_v = sphere_tangent_vectors(u_val, v_val)
    g_uu.append(np.dot(e_u, e_u))
    g_uv.append(np.dot(e_u, e_v))
    g_vv.append(np.dot(e_v, e_v))

ax5.plot(u_vals, g_uu, color=COLORS['red'], linewidth=2, label='g_uu = ⟨∂_u, ∂_u⟩')
ax5.plot(u_vals, g_uv, color=COLORS['blue'], linewidth=2, label='g_uv = ⟨∂_u, ∂_v⟩')
ax5.plot(u_vals, g_vv, color=COLORS['green'], linewidth=2, label='g_vv = ⟨∂_v, ∂_v⟩')

ax5.set_xlabel('u (colatitude)', fontsize=12)
ax5.set_ylabel('Metric component', fontsize=12)
ax5.set_title('Metric Tensor Components\n(dot products of basis vectors)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

# Plot 6: Dimension comparison
ax6 = plt.subplot(2, 3, 6)

manifolds = ['Circle\nS¹', 'Sphere\nS²', 'Torus\nT²', 'Spacetime\nM⁴']
dim_manifold = [1, 2, 2, 4]
dim_tangent = [1, 2, 2, 4]
dim_ambient = [2, 3, 3, 4]

x_pos = np.arange(len(manifolds))
width = 0.25

bars1 = ax6.bar(x_pos - width, dim_manifold, width, label='Manifold dim',
               color=COLORS['blue'], alpha=0.7)
bars2 = ax6.bar(x_pos, dim_tangent, width, label='Tangent space dim',
               color=COLORS['orange'], alpha=0.7)
bars3 = ax6.bar(x_pos + width, dim_ambient, width, label='Ambient space dim',
               color=COLORS['green'], alpha=0.7)

# Add value labels
for i, (dm, dt, da) in enumerate(zip(dim_manifold, dim_tangent, dim_ambient)):
    ax6.text(i, max(dm, dt, da) + 0.2, f'{dt}D',
            ha='center', fontsize=11, fontweight='bold')

ax6.set_ylabel('Dimension', fontsize=12)
ax6.set_title('Manifold vs Tangent Space Dimension\n(always equal!)', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(manifolds, fontsize=10)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Key Properties of Tangent Spaces:")
print("="*60)
print("1. Each point p has its own tangent space T_p M")
print("2. dim(T_p M) = dim(M) always")
print("3. T_p M is a vector space (can add vectors, multiply by scalars)")
print("4. Tangent vectors are derivatives of curves through p")
print("5. Coordinate basis {∂/∂x^i} spans T_p M")
print()
print("Metric tensor g:")
print("  g_ij = ⟨∂_i, ∂_j⟩ (dot products of basis vectors)")
print("  Determines lengths and angles in T_p M")
print("  Essential for geometry!")
print()
print("✓ In GR: Spacetime is a 4D manifold")
print("✓ At each event p, tangent space T_p M is 4-dimensional")
print("✓ Tangent vectors are 4-velocities of particles!")
print("✓ Metric g_μν determines spacetime geometry")
print()
print("Try: Visualize tangent bundles (all tangent spaces together)")
