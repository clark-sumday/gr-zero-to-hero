#!/usr/bin/env python3
"""
Example: Sphere Coordinate Systems
Demonstrates multiple coordinate charts on S² and their transition functions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Sphere S² - 2-dimensional manifold in 3D space

def spherical_coords(theta, phi, R=1.0):
    """
    Standard spherical coordinates
    θ ∈ [0, π] (colatitude from north pole)
    φ ∈ [0, 2π) (longitude)
    """
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return x, y, z

def stereographic_north(x, y, z):
    """
    Stereographic projection from north pole (0, 0, 1)
    Maps S² \\ {north pole} → ℝ²
    """
    # Project from (0,0,1) onto z=0 plane
    if np.abs(z - 1) < 1e-10:  # At north pole
        return np.inf, np.inf
    u = x / (1 - z)
    v = y / (1 - z)
    return u, v

def stereographic_south(x, y, z):
    """
    Stereographic projection from south pole (0, 0, -1)
    Maps S² \\ {south pole} → ℝ²
    """
    if np.abs(z + 1) < 1e-10:  # At south pole
        return np.inf, np.inf
    u = x / (1 + z)
    v = y / (1 + z)
    return u, v

def inverse_stereographic_north(u, v):
    """Map from ℝ² back to sphere (north chart)"""
    r_squared = u**2 + v**2
    x = 2*u / (1 + r_squared)
    y = 2*v / (1 + r_squared)
    z = (r_squared - 1) / (1 + r_squared)
    return x, y, z

def inverse_stereographic_south(u, v):
    """Map from ℝ² back to sphere (south chart)"""
    r_squared = u**2 + v**2
    x = 2*u / (1 + r_squared)
    y = 2*v / (1 + r_squared)
    z = (1 - r_squared) / (1 + r_squared)
    return x, y, z

print("Sphere S² as a Manifold")
print("=" * 60)
print("The sphere is a 2-dimensional manifold")
print("(2D surface living in 3D space)")
print()
print("Cannot use single coordinate chart to cover entire sphere!")
print()
print("Standard spherical coordinates (θ, φ):")
print("  • Problem at poles: φ undefined")
print("  • Problem at φ=0,2π: Discontinuity")
print()
print("Stereographic projection provides better atlas:")
print()
print("Chart U_N (from north pole):")
print("  Covers S² \\ {(0,0,1)}")
print("  (u_N, v_N) ∈ ℝ²")
print()
print("Chart U_S (from south pole):")
print("  Covers S² \\ {(0,0,-1)}")
print("  (u_S, v_S) ∈ ℝ²")
print()
print("Together: Complete atlas covering all of S²")

# Create sphere
theta = np.linspace(0.05, np.pi-0.05, 40)
phi = np.linspace(0, 2*np.pi, 40)
THETA, PHI = np.meshgrid(theta, phi)
X, Y, Z = spherical_coords(THETA, PHI)

# Sample points on sphere
n_samples = 200
theta_sample = np.random.uniform(0.1, np.pi-0.1, n_samples)
phi_sample = np.random.uniform(0, 2*np.pi, n_samples)
x_sample, y_sample, z_sample = spherical_coords(theta_sample, phi_sample)

# Compute stereographic coordinates
u_north = []
v_north = []
u_south = []
v_south = []

for xs, ys, zs in zip(x_sample, y_sample, z_sample):
    un, vn = stereographic_north(xs, ys, zs)
    us, vs = stereographic_south(xs, ys, zs)
    if np.isfinite(un) and np.isfinite(vn):
        u_north.append(un)
        v_north.append(vn)
    if np.isfinite(us) and np.isfinite(vs):
        u_south.append(us)
        v_south.append(vs)

u_north = np.array(u_north)
v_north = np.array(v_north)
u_south = np.array(u_south)
v_south = np.array(v_south)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Sphere with poles marked
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot_surface(X, Y, Z, alpha=0.3, color=COLORS['gray'])

# Mark poles
ax1.scatter([0], [0], [1], color=COLORS['red'], s=300, marker='^',
           edgecolor='black', linewidth=2, label='North pole', zorder=10)
ax1.scatter([0], [0], [-1], color=COLORS['blue'], s=300, marker='v',
           edgecolor='black', linewidth=2, label='South pole', zorder=10)

# Mark equator
phi_eq = np.linspace(0, 2*np.pi, 100)
x_eq, y_eq, z_eq = spherical_coords(np.pi/2 * np.ones_like(phi_eq), phi_eq)
ax1.plot(x_eq, y_eq, z_eq, color=COLORS['green'], linewidth=3, label='Equator')

ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.set_zlabel('z', fontsize=10)
ax1.set_title('Sphere S²', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_box_aspect([1,1,1])

# Plot 2: Stereographic projection from north pole (visualization)
ax2 = fig.add_subplot(2, 3, 2, projection='3d')

# Show projection lines from north pole to equatorial plane
n_proj_lines = 12
theta_proj = np.linspace(0.3, np.pi-0.3, n_proj_lines)
phi_proj = np.linspace(0, 2*np.pi, n_proj_lines, endpoint=False)

for tp in theta_proj[:6]:
    for pp in phi_proj[:6]:
        # Point on sphere
        xp, yp, zp = spherical_coords(tp, pp)
        # Projection point
        up, vp = stereographic_north(xp, yp, zp)
        # Draw line from north pole through sphere point to plane
        ax2.plot([0, xp, up], [0, yp, vp], [1, zp, 0],
                color=COLORS['blue'], alpha=0.3, linewidth=0.5)

ax2.plot_surface(X, Y, Z, alpha=0.2, color=COLORS['gray'])
ax2.scatter([0], [0], [1], color=COLORS['red'], s=200, marker='^')

# Draw equatorial plane
plane_range = np.linspace(-3, 3, 10)
U_plane, V_plane = np.meshgrid(plane_range, plane_range)
Z_plane = np.zeros_like(U_plane)
ax2.plot_surface(U_plane, V_plane, Z_plane, alpha=0.2, color=COLORS['yellow'])

ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.set_zlabel('z', fontsize=10)
ax2.set_title('Stereographic Projection\n(from North Pole)', fontsize=12, fontweight='bold')
ax2.set_box_aspect([1,1,1])

# Plot 3: North chart in coordinate plane
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(u_north, v_north, c=COLORS['blue'], s=5, alpha=0.5)

# Draw circle representing equator in stereographic coords
theta_eq_circle = np.linspace(0, 2*np.pi, 100)
r_eq = 1  # Equator maps to circle of radius 1
u_eq_circle = r_eq * np.cos(theta_eq_circle)
v_eq_circle = r_eq * np.sin(theta_eq_circle)
ax3.plot(u_eq_circle, v_eq_circle, color=COLORS['green'], linewidth=2, label='Equator')

ax3.set_xlabel('u_N', fontsize=12)
ax3.set_ylabel('v_N', fontsize=12)
ax3.set_title('North Chart U_N\n(stereographic coordinates)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_aspect('equal')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)

# Plot 4: South chart in coordinate plane
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(u_south, v_south, c=COLORS['orange'], s=5, alpha=0.5)
ax4.plot(u_eq_circle, v_eq_circle, color=COLORS['green'], linewidth=2, label='Equator')

ax4.set_xlabel('u_S', fontsize=12)
ax4.set_ylabel('v_S', fontsize=12)
ax4.set_title('South Chart U_S\n(stereographic coordinates)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.set_aspect('equal')
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)

# Plot 5: Transition function
ax5 = plt.subplot(2, 3, 5)

# For points in overlap (not near poles), compute both coordinates
# Select points with |z| < 0.9
overlap_mask = np.abs(z_sample) < 0.9
x_overlap = x_sample[overlap_mask]
y_overlap = y_sample[overlap_mask]
z_overlap = z_sample[overlap_mask]

u_n_overlap = []
u_s_overlap = []
for xo, yo, zo in zip(x_overlap, y_overlap, z_overlap):
    un, vn = stereographic_north(xo, yo, zo)
    us, vs = stereographic_south(xo, yo, zo)
    if np.isfinite(un) and np.isfinite(us):
        # Just look at radial coordinate r = sqrt(u² + v²)
        r_n = np.sqrt(un**2 + vn**2)
        r_s = np.sqrt(us**2 + vs**2)
        u_n_overlap.append(r_n)
        u_s_overlap.append(r_s)

u_n_overlap = np.array(u_n_overlap)
u_s_overlap = np.array(u_s_overlap)

ax5.scatter(u_n_overlap, u_s_overlap, c=COLORS['purple'], s=10, alpha=0.5)

# Analytical transition: In overlap, (u_S, v_S) = (u_N, v_N) / (u_N² + v_N²)
# For radial coordinate: r_S = 1/r_N
r_theory = np.linspace(0.1, 5, 100)
r_s_theory = 1 / r_theory
ax5.plot(r_theory, r_s_theory, color=COLORS['red'], linewidth=2,
        label='r_S = 1/r_N', linestyle='--')

ax5.set_xlabel('r_N = √(u_N² + v_N²)', fontsize=12)
ax5.set_ylabel('r_S = √(u_S² + v_S²)', fontsize=12)
ax5.set_title('Transition Function\n(in overlap region)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

# Plot 6: Coverage illustration
ax6 = plt.subplot(2, 3, 6)

# Show which chart covers which region
z_range = np.linspace(-1, 1, 100)
coverage_north = []
coverage_south = []

for z_val in z_range:
    # North chart covers everything except north pole
    if z_val < 0.99:
        coverage_north.append(1)
    else:
        coverage_north.append(0)

    # South chart covers everything except south pole
    if z_val > -0.99:
        coverage_south.append(1)
    else:
        coverage_south.append(0)

ax6.fill_between(z_range, 0, coverage_north, color=COLORS['blue'],
                alpha=0.3, label='North chart')
ax6.fill_between(z_range, 0, coverage_south, color=COLORS['orange'],
                alpha=0.3, label='South chart')

# Show overlap
overlap_region = [min(n, s) for n, s in zip(coverage_north, coverage_south)]
ax6.fill_between(z_range, 0, overlap_region, color=COLORS['green'],
                alpha=0.5, label='Overlap')

ax6.set_xlabel('z coordinate on sphere', fontsize=12)
ax6.set_ylabel('Coverage', fontsize=12)
ax6.set_title('Chart Coverage vs z-coordinate', fontsize=12, fontweight='bold')
ax6.set_ylim(0, 1.2)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Stereographic Projection Properties:")
print("="*60)
print("1. Conformal: Preserves angles (but not distances)")
print("2. Maps circles to circles (or lines)")
print("3. North hemisphere → inside unit circle")
print("4. South hemisphere → outside unit circle")
print("5. Equator → unit circle")
print()
print("Transition function in overlap:")
print("  (u_S, v_S) = (u_N, v_N) / (u_N² + v_N²)")
print("  This is called an 'inversion'")
print()
print("Why two charts are necessary:")
print("  • Each chart misses one point (projection pole)")
print("  • Together they cover entire sphere")
print("  • Smooth transition in overlap region")
print()
print("✓ In GR: Need multiple charts for spacetime manifold!")
print("✓ Black hole spacetimes need several coordinate patches")
print("✓ Try: Add more stereographic charts from different points")
