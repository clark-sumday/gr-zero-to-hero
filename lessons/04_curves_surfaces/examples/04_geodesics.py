#!/usr/bin/env python3
"""
Example: Geodesics on Surfaces
Demonstrates shortest paths on curved surfaces (great circles, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

def sphere_surface(u, v, R=1.0):
    """Parametric sphere"""
    x = R * np.sin(u) * np.cos(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(u)
    return x, y, z

def sphere_geodesic_equations(state, t, R=1.0):
    """
    Geodesic equations on a sphere
    d²u/dt² - sin(u)cos(u)(dv/dt)² = 0
    d²v/dt² + 2cot(u)(du/dt)(dv/dt) = 0

    Rewrite as first-order system:
    state = [u, v, u', v']
    """
    u, v, u_dot, v_dot = state

    # Avoid singularities at poles
    if np.abs(np.sin(u)) < 1e-6:
        u = 1e-6 if u < np.pi/2 else np.pi - 1e-6

    u_ddot = np.sin(u) * np.cos(u) * v_dot**2
    v_ddot = -2 * np.cos(u) / np.sin(u) * u_dot * v_dot

    return [u_dot, v_dot, u_ddot, v_ddot]

def cylinder_surface(u, v, R=1.0):
    """Parametric cylinder"""
    x = R * np.cos(u)
    y = R * np.sin(u)
    z = v
    return x, y, z

print("Geodesics on Surfaces")
print("=" * 60)
print("A geodesic is the shortest path between two points on a surface")
print()
print("Examples:")
print("  • Plane: Straight lines")
print("  • Sphere: Great circles (like equator or meridians)")
print("  • Cylinder: Helices (or straight lines parallel to axis)")
print()
print("Geodesics are solutions to the geodesic equation:")
print("  d²xᵏ/dt² + Γᵏᵢⱼ(dxⁱ/dt)(dxʲ/dt) = 0")
print()
print("where Γᵏᵢⱼ are Christoffel symbols (derived from the metric)")

# Create sphere
u_sph = np.linspace(0.1, np.pi-0.1, 50)
v_sph = np.linspace(0, 2*np.pi, 50)
U_sph, V_sph = np.meshgrid(u_sph, v_sph)
X_sph, Y_sph, Z_sph = sphere_surface(U_sph, V_sph, R=1.0)

# Create cylinder
u_cyl = np.linspace(0, 2*np.pi, 50)
v_cyl = np.linspace(-2, 2, 50)
U_cyl, V_cyl = np.meshgrid(u_cyl, v_cyl)
X_cyl, Y_cyl, Z_cyl = cylinder_surface(U_cyl, V_cyl, R=1.0)

# Compute geodesics on sphere
print("\nComputing geodesics on sphere...")

# Great circle from north pole to equator
t = np.linspace(0, np.pi, 200)

# Initial conditions for different geodesics
geodesic_ics = [
    [np.pi/4, 0, 1, 0],      # Meridian (longitude line)
    [np.pi/2, 0, 0, 1],      # Equator
    [np.pi/4, 0, 0.7, 0.7],  # Great circle at angle
]

geodesic_labels = ['Meridian', 'Equator', 'Tilted Great Circle']
geodesic_colors = [COLORS['red'], COLORS['blue'], COLORS['green']]

geodesics = []
for ic in geodesic_ics:
    sol = odeint(sphere_geodesic_equations, ic, t)
    geodesics.append(sol)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Geodesics on sphere
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot_surface(X_sph, Y_sph, Z_sph, alpha=0.3, color=COLORS['gray'])

for geo, label, color in zip(geodesics, geodesic_labels, geodesic_colors):
    u_geo = geo[:, 0]
    v_geo = geo[:, 1]
    x_geo, y_geo, z_geo = sphere_surface(u_geo, v_geo)
    ax1.plot(x_geo, y_geo, z_geo, color=color, linewidth=3, label=label)

ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.set_zlabel('z', fontsize=10)
ax1.set_title('Geodesics on Sphere\n(Great Circles)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_box_aspect([1,1,1])

# Plot 2: Non-geodesic path (small circle of latitude)
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot_surface(X_sph, Y_sph, Z_sph, alpha=0.3, color=COLORS['gray'])

# Great circle (geodesic)
u_equator = np.pi/2 * np.ones(100)
v_equator = np.linspace(0, 2*np.pi, 100)
x_eq, y_eq, z_eq = sphere_surface(u_equator, v_equator)
ax2.plot(x_eq, y_eq, z_eq, color=COLORS['blue'], linewidth=3, label='Geodesic (equator)')

# Small circle (not a geodesic)
u_small = np.pi/4 * np.ones(100)
v_small = np.linspace(0, 2*np.pi, 100)
x_sm, y_sm, z_sm = sphere_surface(u_small, v_small)
ax2.plot(x_sm, y_sm, z_sm, color=COLORS['red'], linewidth=3,
        linestyle='--', label='Not geodesic (latitude)')

ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.set_zlabel('z', fontsize=10)
ax2.set_title('Geodesic vs Non-Geodesic', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_box_aspect([1,1,1])

# Plot 3: Geodesics on cylinder
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.3, color=COLORS['gray'])

# Straight vertical line (geodesic)
u_vert = np.zeros(100)
v_vert = np.linspace(-2, 2, 100)
x_v, y_v, z_v = cylinder_surface(u_vert, v_vert)
ax3.plot(x_v, y_v, z_v, color=COLORS['blue'], linewidth=3, label='Vertical geodesic')

# Horizontal circle (geodesic)
u_horiz = np.linspace(0, 2*np.pi, 100)
v_horiz = np.ones(100) * 0.5
x_h, y_h, z_h = cylinder_surface(u_horiz, v_horiz)
ax3.plot(x_h, y_h, z_h, color=COLORS['green'], linewidth=3, label='Horizontal geodesic')

# Helix (geodesic!)
t_helix = np.linspace(0, 4*np.pi, 200)
u_helix = t_helix
v_helix = 0.3 * t_helix
x_hx, y_hx, z_hx = cylinder_surface(u_helix, v_helix)
ax3.plot(x_hx, y_hx, z_hx, color=COLORS['red'], linewidth=3, label='Helical geodesic')

ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('y', fontsize=10)
ax3.set_zlabel('z', fontsize=10)
ax3.set_title('Geodesics on Cylinder\n(Lines and Helices)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_box_aspect([1,1,1])

# Plot 4: Geodesic in parameter space
ax4 = plt.subplot(2, 3, 4)
for geo, label, color in zip(geodesics, geodesic_labels, geodesic_colors):
    u_geo = geo[:, 0]
    v_geo = geo[:, 1]
    ax4.plot(v_geo, u_geo, color=color, linewidth=2, label=label)

ax4.set_xlabel('v (longitude)', fontsize=12)
ax4.set_ylabel('u (colatitude)', fontsize=12)
ax4.set_title('Geodesics in Parameter Space', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.set_xlim(0, 2*np.pi)
ax4.set_ylim(0, np.pi)

# Plot 5: Speed along geodesic
ax5 = plt.subplot(2, 3, 5)
for geo, label, color in zip(geodesics, geodesic_labels, geodesic_colors):
    u_dot = geo[:, 2]
    v_dot = geo[:, 3]
    # Speed in parameter space
    speed = np.sqrt(u_dot**2 + np.sin(geo[:, 0])**2 * v_dot**2)
    ax5.plot(t, speed, color=color, linewidth=2, label=label)

ax5.set_xlabel('t', fontsize=12)
ax5.set_ylabel('|dr/dt|', fontsize=12)
ax5.set_title('Speed Along Geodesic\n(constant for geodesics)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

# Plot 6: Triangle on sphere (geodesic triangle)
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
ax6.plot_surface(X_sph, Y_sph, Z_sph, alpha=0.3, color=COLORS['gray'])

# Vertices of geodesic triangle (on unit sphere)
vertex1 = np.array([1, 0, 0])  # On equator
vertex2 = np.array([0, 1, 0])  # On equator
vertex3 = np.array([0, 0, 1])  # North pole

# Great circle edges
def great_circle_between(p1, p2, n_points=50):
    """Great circle arc between two points on unit sphere"""
    # Angle between points
    theta = np.arccos(np.dot(p1, p2))
    t_vals = np.linspace(0, 1, n_points)

    # Spherical linear interpolation
    points = []
    for t_val in t_vals:
        if theta < 1e-6:
            p = p1
        else:
            p = (np.sin((1-t_val)*theta) * p1 + np.sin(t_val*theta) * p2) / np.sin(theta)
        points.append(p)
    return np.array(points)

edge1 = great_circle_between(vertex1, vertex2)
edge2 = great_circle_between(vertex2, vertex3)
edge3 = great_circle_between(vertex3, vertex1)

ax6.plot(edge1[:, 0], edge1[:, 1], edge1[:, 2],
        color=COLORS['red'], linewidth=3, label='Geodesic edges')
ax6.plot(edge2[:, 0], edge2[:, 1], edge2[:, 2],
        color=COLORS['red'], linewidth=3)
ax6.plot(edge3[:, 0], edge3[:, 1], edge3[:, 2],
        color=COLORS['red'], linewidth=3)

ax6.scatter(*vertex1, color=COLORS['blue'], s=100, zorder=5)
ax6.scatter(*vertex2, color=COLORS['blue'], s=100, zorder=5)
ax6.scatter(*vertex3, color=COLORS['blue'], s=100, zorder=5)

ax6.set_xlabel('x', fontsize=10)
ax6.set_ylabel('y', fontsize=10)
ax6.set_zlabel('z', fontsize=10)
ax6.set_title('Geodesic Triangle on Sphere\n(angles sum > 180°!)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.set_box_aspect([1,1,1])

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Properties of Geodesics:")
print("="*60)
print("1. Locally shortest path between nearby points")
print("2. 'Straight as possible' on the surface")
print("3. Zero geodesic curvature (doesn't bend within surface)")
print("4. Parallel transports its own tangent vector")
print()
print("On a sphere:")
print("  • Great circles are geodesics")
print("  • Small circles (latitude lines) are NOT geodesics")
print("  • Geodesic triangles have angle sum > 180° (positive curvature)")
print()
print("On a cylinder:")
print("  • Straight lines (vertical and horizontal) are geodesics")
print("  • Helices are also geodesics!")
print("  • Can be 'unrolled' to see straight lines on flat paper")
print()
print("✓ In GR: Freely falling particles follow geodesics in spacetime!")
print("✓ This is the generalization of Newton's first law!")
print("✓ Try: Compute geodesics on other surfaces (torus, saddle)")
