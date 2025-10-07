#!/usr/bin/env python3
"""
Example: The Metric Tensor - Measuring Distances in Different Geometries
Demonstrates how the metric tensor defines geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("THE METRIC TENSOR: DEFINING GEOMETRY")
print("="*60)

# 1. EUCLIDEAN METRIC (flat space)
print("\n1. EUCLIDEAN METRIC (Flat 3D Space)")
print("-" * 60)
g_euclidean = np.eye(3)
print("Metric g_ij =")
print(g_euclidean)
print("\nLine element: ds² = dx² + dy² + dz²")

# Compute distance between two points
point_A = np.array([0, 0, 0])
point_B = np.array([3, 4, 0])
displacement = point_B - point_A
distance_squared = displacement @ g_euclidean @ displacement
distance = np.sqrt(distance_squared)

print(f"\nPoint A: {point_A}")
print(f"Point B: {point_B}")
print(f"Distance: √(3² + 4²) = {distance:.2f}")

# 2. MINKOWSKI METRIC (flat spacetime)
print("\n2. MINKOWSKI METRIC (Flat Spacetime)")
print("-" * 60)
eta = np.diag([-1, 1, 1, 1])
print("Minkowski metric η_μν =")
print(eta)
print("\nLine element: ds² = -dt² + dx² + dy² + dz²")
print("Signature: (-,+,+,+) or 'mostly plus'")

# Spacetime interval
event_1 = np.array([0, 0, 0, 0])  # (t, x, y, z)
event_2 = np.array([5, 3, 4, 0])
separation = event_2 - event_1
interval_squared = separation @ eta @ separation

print(f"\nEvent 1: (t,x,y,z) = {event_1}")
print(f"Event 2: (t,x,y,z) = {event_2}")
print(f"Interval²: s² = -Δt² + Δx² + Δy² + Δz²")
print(f"         = -(5)² + (3)² + (4)² + (0)²")
print(f"         = -25 + 9 + 16 = {interval_squared}")

if interval_squared < 0:
    proper_time = np.sqrt(-interval_squared)
    print(f"✓ Timelike separation (proper time τ = {proper_time:.3f})")
elif interval_squared > 0:
    proper_distance = np.sqrt(interval_squared)
    print(f"✓ Spacelike separation (proper distance = {proper_distance:.3f})")
else:
    print("✓ Null/Lightlike separation (photon path)")

# 3. POLAR METRIC (flat space, curved coordinates)
print("\n3. POLAR METRIC (Flat 2D Space in Polar Coords)")
print("-" * 60)

def metric_polar(r):
    """Metric in polar coordinates"""
    return np.array([[1, 0],
                     [0, r**2]])

r = 2.0
g_polar = metric_polar(r)
print(f"At r = {r}:")
print("g_ij =")
print(g_polar)
print(f"\nLine element: ds² = dr² + r² dθ²")
print("\nNotice: g_θθ = r² (circumference grows with radius)")

# Arc length
theta_start = 0
theta_end = np.pi / 4
arc_length = r * (theta_end - theta_start)
print(f"\nArc length from θ=0 to θ=π/4 at r={r}:")
print(f"s = r Δθ = {r} × {theta_end:.4f} = {arc_length:.4f}")

# 4. SPHERICAL METRIC (surface of sphere)
print("\n4. SPHERE METRIC (Surface of 2-Sphere)")
print("-" * 60)

def metric_sphere(R, theta):
    """Metric on 2-sphere of radius R"""
    return np.array([[R**2, 0],
                     [0, R**2 * np.sin(theta)**2]])

R = 1.0  # Unit sphere
theta_point = np.pi / 4  # 45 degrees from north pole
g_sphere = metric_sphere(R, theta_point)

print(f"Sphere radius R = {R}")
print(f"At θ = π/4 (45° from north pole):")
print("g_ij =")
print(g_sphere)
print(f"\nLine element: ds² = R²(dθ² + sin²θ dφ²)")
print("\nThis is INTRINSICALLY CURVED (Gaussian curvature K = 1/R²)")

# Visualization
fig = plt.figure(figsize=(14, 10))

# Plot 1: Euclidean metric - straight line is shortest
ax1 = plt.subplot(2, 2, 1)
x_line = np.linspace(0, 3, 100)
y_line = (4/3) * x_line  # Straight line from (0,0) to (3,4)

ax1.plot(x_line, y_line, color=COLORS['green'], linewidth=3, label='Geodesic (straight line)')
ax1.plot([0, 3], [0, 4], 'o', color=COLORS['blue'], markersize=10)
ax1.text(0, 0, ' A(0,0)', fontsize=11, verticalalignment='top')
ax1.text(3, 4, ' B(3,4)', fontsize=11, verticalalignment='bottom')

# Alternative curved path (longer)
t = np.linspace(0, 1, 100)
x_curved = 3 * t
y_curved = 4 * t + np.sin(3*np.pi*t)
ax1.plot(x_curved, y_curved, '--', color=COLORS['red'], linewidth=2, alpha=0.7, label='Curved path (longer)')

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Euclidean Metric: ds² = dx² + dy²\nShortest = Straight Line', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: Minkowski light cone
ax2 = plt.subplot(2, 2, 2)
t_vals = np.linspace(-5, 5, 100)

ax2.fill_between(t_vals, -np.abs(t_vals), np.abs(t_vals), alpha=0.2, color=COLORS['yellow'], label='Timelike')
ax2.plot(t_vals, t_vals, color=COLORS['orange'], linewidth=3, label='Light cone (ds²=0)')
ax2.plot(t_vals, -t_vals, color=COLORS['orange'], linewidth=3)

# Example timelike path
tau = np.linspace(0, 4, 50)
x_particle = 0.5 * tau
t_particle = np.sqrt(tau**2 + x_particle**2)
ax2.plot(x_particle, t_particle, color=COLORS['blue'], linewidth=3, linestyle='--', label='Massive particle')

ax2.set_xlabel('x (space)', fontsize=12)
ax2.set_ylabel('t (time)', fontsize=12)
ax2.set_title('Minkowski Metric: ds² = -dt² + dx²\nLight Cones & Causality', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-5, 5])
ax2.set_ylim([-1, 5])
ax2.axhline(0, color='k', linewidth=0.5)
ax2.axvline(0, color='k', linewidth=0.5)

# Plot 3: Polar metric - circles at different radii
ax3 = plt.subplot(2, 2, 3)
theta_circle = np.linspace(0, 2*np.pi, 100)

for r_val in [1, 2, 3]:
    x_circle = r_val * np.cos(theta_circle)
    y_circle = r_val * np.sin(theta_circle)
    circumference = 2 * np.pi * r_val
    ax3.plot(x_circle, y_circle, linewidth=2, label=f'r={r_val}, C={circumference:.1f}')

# Show metric components vary with position
r_test = 2
theta_test = np.pi/4
g_test = metric_polar(r_test)
ax3.plot(r_test * np.cos(theta_test), r_test * np.sin(theta_test), 'ro', markersize=10)
ax3.text(r_test * np.cos(theta_test) + 0.3, r_test * np.sin(theta_test),
        f'g_θθ = {g_test[1,1]:.0f}', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Polar Metric: ds² = dr² + r²dθ²\nMetric Components Vary', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

# Plot 4: Sphere (intrinsically curved)
ax4 = plt.subplot(2, 2, 4, projection='3d')

u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2*np.pi, 30)
U, V = np.meshgrid(u, v)

X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

ax4.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', edgecolor='none')

# Draw a geodesic (great circle)
phi_geodesic = np.linspace(0, np.pi, 100)
theta_geodesic = np.pi / 4
x_geo = np.sin(phi_geodesic) * np.cos(theta_geodesic)
y_geo = np.sin(phi_geodesic) * np.sin(theta_geodesic)
z_geo = np.cos(phi_geodesic)
ax4.plot(x_geo, y_geo, z_geo, color=COLORS['red'], linewidth=3, label='Geodesic (great circle)')

ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('z')
ax4.set_title('Sphere: ds² = R²(dθ² + sin²θ dφ²)\nIntrinsic Curvature K=1/R²', fontsize=11, fontweight='bold')
ax4.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. The metric g_μν completely determines the geometry")
print("2. Same space can have different metrics in different coordinates")
print("3. Euclidean: positive definite (all eigenvalues > 0)")
print("4. Minkowski: indefinite signature (1 negative, 3 positive)")
print("5. Curved spaces: metric varies from point to point")
print("6. Geodesics (shortest paths) depend on the metric")
print("\n✓ In GR, Einstein's equations determine how matter curves g_μν!")
