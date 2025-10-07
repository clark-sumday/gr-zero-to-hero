#!/usr/bin/env python3
"""
Example: Vector Fields on Manifolds
Demonstrates smooth assignment of tangent vectors to each point
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

def sphere_tangent_basis(u, v, R=1.0):
    """Coordinate basis vectors ∂_u and ∂_v"""
    # ∂r/∂u
    e_u = np.array([
        R * np.cos(u) * np.cos(v),
        R * np.cos(u) * np.sin(v),
        -R * np.sin(u)
    ])

    # ∂r/∂v
    e_v = np.array([
        -R * np.sin(u) * np.sin(v),
        R * np.sin(u) * np.cos(v),
        0
    ])

    return e_u, e_v

def vector_field_meridian(u, v, R=1.0):
    """
    Vector field pointing along meridians (north-south)
    V = ∂/∂u
    """
    e_u, e_v = sphere_tangent_basis(u, v, R)
    return e_u

def vector_field_latitude(u, v, R=1.0):
    """
    Vector field pointing along latitude circles (east-west)
    V = ∂/∂v
    """
    e_u, e_v = sphere_tangent_basis(u, v, R)
    return e_v

def vector_field_spiral(u, v, R=1.0):
    """
    Spiral vector field
    V = ∂/∂u + ∂/∂v
    """
    e_u, e_v = sphere_tangent_basis(u, v, R)
    return e_u + e_v

def vector_field_radial_2d(x, y):
    """Radial vector field in plane"""
    return np.array([x, y])

def vector_field_rotational_2d(x, y):
    """Rotational vector field in plane"""
    return np.array([-y, x])

def torus_surface(u, v, R=2.0, r=0.5):
    """Parametric torus"""
    x = (R + r*np.cos(u)) * np.cos(v)
    y = (R + r*np.cos(u)) * np.sin(v)
    z = r * np.sin(u)
    return x, y, z

def torus_tangent_basis(u, v, R=2.0, r=0.5):
    """Coordinate basis for torus"""
    e_u = np.array([
        -r * np.sin(u) * np.cos(v),
        -r * np.sin(u) * np.sin(v),
        r * np.cos(u)
    ])

    e_v = np.array([
        -(R + r*np.cos(u)) * np.sin(v),
        (R + r*np.cos(u)) * np.cos(v),
        0
    ])

    return e_u, e_v

def vector_field_torus_poloidal(u, v, R=2.0, r=0.5):
    """Poloidal (around small circle)"""
    e_u, e_v = torus_tangent_basis(u, v, R, r)
    return e_u

def vector_field_torus_toroidal(u, v, R=2.0, r=0.5):
    """Toroidal (around large circle)"""
    e_u, e_v = torus_tangent_basis(u, v, R, r)
    return e_v

print("Vector Fields on Manifolds")
print("=" * 60)
print("A vector field is a smooth assignment of a tangent vector")
print("to each point on the manifold")
print()
print("Formally: V: M → TM")
print("  For each p ∈ M, V(p) ∈ T_p M")
print()
print("In coordinates: V = V^i ∂/∂x^i")
print("  V^i = components (smooth functions)")
print("  ∂/∂x^i = coordinate basis vectors")
print()
print("Examples on sphere S²:")
print("  1. Meridional: V = ∂/∂u (points north-south)")
print("  2. Latitudinal: V = ∂/∂v (points east-west)")
print("  3. Spiral: V = ∂/∂u + ∂/∂v")
print()
print("Vector fields define flows (integral curves)")

# Create sphere
u_sph = np.linspace(0.1, np.pi-0.1, 20)
v_sph = np.linspace(0, 2*np.pi, 20)
U_sph, V_sph = np.meshgrid(u_sph, v_sph)
X_sph, Y_sph, Z_sph = sphere_surface(U_sph, V_sph)

# Create torus
u_tor = np.linspace(0, 2*np.pi, 20)
v_tor = np.linspace(0, 2*np.pi, 20)
U_tor, V_tor = np.meshgrid(u_tor, v_tor)
X_tor, Y_tor, Z_tor = torus_surface(U_tor, V_tor)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Meridional vector field on sphere
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot_surface(X_sph, Y_sph, Z_sph, alpha=0.3, color=COLORS['gray'])

# Sample points
u_sample = np.linspace(np.pi/6, 5*np.pi/6, 6)
v_sample = np.linspace(0, 2*np.pi, 8, endpoint=False)

scale = 0.2
for u_s in u_sample:
    for v_s in v_sample:
        x_s, y_s, z_s = sphere_surface(u_s, v_s)
        vec = vector_field_meridian(u_s, v_s)
        vec_norm = vec / np.linalg.norm(vec) * scale

        ax1.quiver(x_s, y_s, z_s, vec_norm[0], vec_norm[1], vec_norm[2],
                  color=COLORS['red'], arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8)

ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.set_zlabel('z', fontsize=10)
ax1.set_title('Meridional Field V = ∂/∂u\n(points toward poles)', fontsize=12, fontweight='bold')
ax1.set_box_aspect([1,1,1])

# Plot 2: Latitudinal vector field on sphere
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot_surface(X_sph, Y_sph, Z_sph, alpha=0.3, color=COLORS['gray'])

for u_s in u_sample:
    for v_s in v_sample:
        x_s, y_s, z_s = sphere_surface(u_s, v_s)
        vec = vector_field_latitude(u_s, v_s)
        if np.linalg.norm(vec) > 1e-6:
            vec_norm = vec / np.linalg.norm(vec) * scale
            ax2.quiver(x_s, y_s, z_s, vec_norm[0], vec_norm[1], vec_norm[2],
                      color=COLORS['blue'], arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8)

ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.set_zlabel('z', fontsize=10)
ax2.set_title('Latitudinal Field V = ∂/∂v\n(circles around z-axis)', fontsize=12, fontweight='bold')
ax2.set_box_aspect([1,1,1])

# Plot 3: Spiral vector field on sphere
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.plot_surface(X_sph, Y_sph, Z_sph, alpha=0.3, color=COLORS['gray'])

for u_s in u_sample:
    for v_s in v_sample:
        x_s, y_s, z_s = sphere_surface(u_s, v_s)
        vec = vector_field_spiral(u_s, v_s)
        vec_norm = vec / np.linalg.norm(vec) * scale

        ax3.quiver(x_s, y_s, z_s, vec_norm[0], vec_norm[1], vec_norm[2],
                  color=COLORS['green'], arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8)

# Draw integral curve (following the vector field)
u_curve = np.linspace(np.pi/4, 3*np.pi/4, 50)
v_curve = np.linspace(0, 4*np.pi, 50)  # Spirals around
x_curve, y_curve, z_curve = sphere_surface(u_curve, v_curve % (2*np.pi))
ax3.plot(x_curve, y_curve, z_curve, color=COLORS['orange'], linewidth=3,
        label='Integral curve')

ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('y', fontsize=10)
ax3.set_zlabel('z', fontsize=10)
ax3.set_title('Spiral Field V = ∂/∂u + ∂/∂v\n(with integral curve)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_box_aspect([1,1,1])

# Plot 4: Vector fields on plane (for comparison)
ax4 = plt.subplot(2, 3, 4)

x_plane = np.linspace(-2, 2, 15)
y_plane = np.linspace(-2, 2, 15)
X_plane, Y_plane = np.meshgrid(x_plane, y_plane)

# Radial field
U_rad = X_plane
V_rad = Y_plane
magnitude_rad = np.sqrt(U_rad**2 + V_rad**2)
U_rad_norm = U_rad / (magnitude_rad + 1e-10)
V_rad_norm = V_rad / (magnitude_rad + 1e-10)

ax4.quiver(X_plane, Y_plane, U_rad_norm, V_rad_norm,
          color=COLORS['red'], alpha=0.7, width=0.004)

ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.set_title('Radial Field in Plane\nV = x∂/∂x + y∂/∂y', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)

# Plot 5: Rotational field on plane
ax5 = plt.subplot(2, 3, 5)

U_rot = -Y_plane
V_rot = X_plane
magnitude_rot = np.sqrt(U_rot**2 + V_rot**2)
U_rot_norm = U_rot / (magnitude_rot + 1e-10)
V_rot_norm = V_rot / (magnitude_rot + 1e-10)

ax5.quiver(X_plane, Y_plane, U_rot_norm, V_rot_norm,
          color=COLORS['blue'], alpha=0.7, width=0.004)

# Draw circular integral curves
theta_circle = np.linspace(0, 2*np.pi, 100)
for r_circle in [0.5, 1.0, 1.5]:
    x_circle = r_circle * np.cos(theta_circle)
    y_circle = r_circle * np.sin(theta_circle)
    ax5.plot(x_circle, y_circle, color=COLORS['orange'], linewidth=2, alpha=0.7)

ax5.set_xlabel('x', fontsize=12)
ax5.set_ylabel('y', fontsize=12)
ax5.set_title('Rotational Field in Plane\nV = -y∂/∂x + x∂/∂y', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_aspect('equal')
ax5.axhline(y=0, color='k', linewidth=0.5)
ax5.axvline(x=0, color='k', linewidth=0.5)

# Plot 6: Vector fields on torus
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
ax6.plot_surface(X_tor, Y_tor, Z_tor, alpha=0.3, color=COLORS['gray'])

u_sample_tor = np.linspace(0, 2*np.pi, 6, endpoint=False)
v_sample_tor = np.linspace(0, 2*np.pi, 8, endpoint=False)

scale_tor = 0.15

# Show both poloidal and toroidal
for i, u_s in enumerate(u_sample_tor):
    for j, v_s in enumerate(v_sample_tor):
        x_s, y_s, z_s = torus_surface(u_s, v_s)

        # Alternate between poloidal and toroidal
        if (i + j) % 2 == 0:
            vec = vector_field_torus_poloidal(u_s, v_s)
            color = COLORS['red']
        else:
            vec = vector_field_torus_toroidal(u_s, v_s)
            color = COLORS['blue']

        if np.linalg.norm(vec) > 1e-6:
            vec_norm = vec / np.linalg.norm(vec) * scale_tor
            ax6.quiver(x_s, y_s, z_s, vec_norm[0], vec_norm[1], vec_norm[2],
                      color=color, arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8)

# Add legend
ax6.plot([], [], color=COLORS['red'], linewidth=2, label='Poloidal (∂/∂u)')
ax6.plot([], [], color=COLORS['blue'], linewidth=2, label='Toroidal (∂/∂v)')

ax6.set_xlabel('x', fontsize=10)
ax6.set_ylabel('y', fontsize=10)
ax6.set_zlabel('z', fontsize=10)
ax6.set_title('Vector Fields on Torus\n(alternating)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.set_box_aspect([1,1,1])

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Types of Vector Fields:")
print("="*60)
print("1. Coordinate basis fields: ∂/∂x^i")
print("   Natural basis at each point")
print()
print("2. Killing vector fields:")
print("   Generate symmetries (isometries)")
print("   Example: ∂/∂v on sphere (rotational symmetry)")
print()
print("3. Gradient fields: ∇f")
print("   Perpendicular to level sets of function f")
print()
print("4. Divergence-free fields: ∇·V = 0")
print("   Example: Rotational field in plane")
print()
print("Integral Curves:")
print("  Curves γ(t) whose tangent equals the vector field")
print("  γ'(t) = V(γ(t))")
print("  Example: Latitude circles for ∂/∂v field on sphere")
print()
print("✓ In GR: Vector fields are crucial!")
print("✓ 4-velocity field: describes matter flow")
print("✓ Killing vectors: encode spacetime symmetries")
print("✓ Timelike Killing vector: defines conserved energy")
print()
print("Try: Create vector field with both poloidal and toroidal components")
