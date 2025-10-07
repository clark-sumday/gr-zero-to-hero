#!/usr/bin/env python3
"""
Example: Geodesics on a Sphere - Shortest Paths in Curved Space
Demonstrates great circles as geodesics on a sphere
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("GEODESICS: SHORTEST PATHS IN CURVED SPACE")
print("="*60)

print("\nGeodesic equation (general form):")
print("d²x^μ/dλ² + Γ^μ_νρ (dx^ν/dλ)(dx^ρ/dλ) = 0")

print("\nOn a sphere: geodesics are GREAT CIRCLES")
print("(Circles of maximum radius that go through two points)")

# Sphere parameters
R = 1.0  # Unit sphere

print(f"\n Sphere radius: R = {R}")
print("Metric: ds² = R²(dθ² + sin²θ dφ²)")

# Christoffel symbols for sphere
def christoffel_sphere(theta):
    """
    Non-zero Christoffel symbols on sphere:
    Γ^θ_φφ = -sin(θ)cos(θ)
    Γ^φ_θφ = Γ^φ_φθ = cot(θ) = cos(θ)/sin(θ)
    """
    Gamma = np.zeros((2, 2, 2))

    # Avoid singularity at poles
    if abs(np.sin(theta)) < 1e-10:
        return Gamma

    Gamma[0, 1, 1] = -np.sin(theta) * np.cos(theta)  # Γ^θ_φφ
    Gamma[1, 0, 1] = np.cos(theta) / np.sin(theta)   # Γ^φ_θφ
    Gamma[1, 1, 0] = np.cos(theta) / np.sin(theta)   # Γ^φ_φθ

    return Gamma

def geodesic_equations_sphere(state, lam):
    """
    Geodesic equations on sphere
    state = [theta, phi, dtheta/dlam, dphi/dlam]
    """
    theta, phi, theta_dot, phi_dot = state

    Gamma = christoffel_sphere(theta)

    # d²theta/dlam² = -Γ^θ_νρ (dx^ν/dlam)(dx^ρ/dlam)
    # = -Γ^θ_φφ (dphi/dlam)²
    theta_ddot = -Gamma[0, 1, 1] * phi_dot**2

    # d²phi/dlam² = -Γ^φ_νρ (dx^ν/dlam)(dx^ρ/dlam)
    # = -Γ^φ_θφ (dtheta/dlam)(dphi/dlam) - Γ^φ_φθ (dphi/dlam)(dtheta/dlam)
    # = -2 Γ^φ_θφ (dtheta/dlam)(dphi/dlam)
    phi_ddot = -2 * Gamma[1, 0, 1] * theta_dot * phi_dot

    return [theta_dot, phi_dot, theta_ddot, phi_ddot]

# Example geodesics
print("\n" + "="*60)
print("COMPUTING GEODESICS")
print("="*60)

# Geodesic 1: Meridian (phi = constant)
print("\n1. MERIDIAN (longitude line)")
theta0_1 = 0.1  # Start near north pole
phi0_1 = 0.0
theta_dot0_1 = 1.0  # Moving in theta direction
phi_dot0_1 = 0.0    # Not moving in phi

initial_state_1 = [theta0_1, phi0_1, theta_dot0_1, phi_dot0_1]
lambda_vals_1 = np.linspace(0, np.pi - 0.2, 100)

solution_1 = odeint(geodesic_equations_sphere, initial_state_1, lambda_vals_1)

print(f"Initial: θ={theta0_1:.2f}, φ={phi0_1:.2f}")
print(f"Final: θ={solution_1[-1, 0]:.2f}, φ={solution_1[-1, 1]:.2f}")
print("Result: φ stays constant → meridian is a geodesic ✓")

# Geodesic 2: Equator (theta = pi/2)
print("\n2. EQUATOR")
theta0_2 = np.pi / 2
phi0_2 = 0.0
theta_dot0_2 = 0.0
phi_dot0_2 = 1.0

initial_state_2 = [theta0_2, phi0_2, theta_dot0_2, phi_dot0_2]
lambda_vals_2 = np.linspace(0, 2*np.pi, 100)

solution_2 = odeint(geodesic_equations_sphere, initial_state_2, lambda_vals_2)

print(f"Initial: θ={theta0_2:.2f}, φ={phi0_2:.2f}")
print(f"Final: θ={solution_2[-1, 0]:.2f}, φ={solution_2[-1, 1]:.2f}")
print("Result: θ stays at π/2 → equator is a geodesic ✓")

# Geodesic 3: General great circle
print("\n3. GENERAL GREAT CIRCLE")
theta0_3 = np.pi / 4
phi0_3 = 0.0
theta_dot0_3 = 0.5
phi_dot0_3 = 0.5

initial_state_3 = [theta0_3, phi0_3, theta_dot0_3, phi_dot0_3]
lambda_vals_3 = np.linspace(0, 5, 200)

solution_3 = odeint(geodesic_equations_sphere, initial_state_3, lambda_vals_3)

print(f"Initial: θ={theta0_3:.2f}, φ={phi0_3:.2f}")
print(f"Initial velocity: dθ/dλ={theta_dot0_3}, dφ/dλ={phi_dot0_3}")
print("Result: Traces out a great circle")

# Comparison: non-geodesic path
print("\n4. NON-GEODESIC (latitude line, not equator)")
print("Small circles (latitude ≠ equator) are NOT geodesics")
print("They have non-zero geodesic curvature")

# Visualization
fig = plt.figure(figsize=(14, 11))

# Plot 1: Sphere with multiple geodesics
ax1 = plt.subplot(2, 2, 1, projection='3d')

# Draw sphere
u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2*np.pi, 30)
U, V = np.meshgrid(u, v)
X_sphere = R * np.sin(U) * np.cos(V)
Y_sphere = R * np.sin(U) * np.sin(V)
Z_sphere = R * np.cos(U)

ax1.plot_surface(X_sphere, Y_sphere, Z_sphere, alpha=0.3, color=COLORS['blue'], edgecolor='none')

# Plot geodesic 1: Meridian
theta_1 = solution_1[:, 0]
phi_1 = solution_1[:, 1]
x1 = R * np.sin(theta_1) * np.cos(phi_1)
y1 = R * np.sin(theta_1) * np.sin(phi_1)
z1 = R * np.cos(theta_1)
ax1.plot(x1, y1, z1, color=COLORS['red'], linewidth=3, label='Meridian')

# Plot geodesic 2: Equator
theta_2 = solution_2[:, 0]
phi_2 = solution_2[:, 1]
x2 = R * np.sin(theta_2) * np.cos(phi_2)
y2 = R * np.sin(theta_2) * np.sin(phi_2)
z2 = R * np.cos(theta_2)
ax1.plot(x2, y2, z2, color=COLORS['green'], linewidth=3, label='Equator')

# Plot geodesic 3: General great circle
theta_3 = solution_3[:, 0]
phi_3 = solution_3[:, 1]
x3 = R * np.sin(theta_3) * np.cos(phi_3)
y3 = R * np.sin(theta_3) * np.sin(phi_3)
z3 = R * np.cos(theta_3)
ax1.plot(x3, y3, z3, color=COLORS['purple'], linewidth=3, label='Great circle')

# Non-geodesic: latitude circle at theta = pi/3
theta_lat = np.pi / 3
phi_lat = np.linspace(0, 2*np.pi, 100)
x_lat = R * np.sin(theta_lat) * np.cos(phi_lat)
y_lat = R * np.sin(theta_lat) * np.sin(phi_lat)
z_lat = R * np.cos(theta_lat) * np.ones_like(phi_lat)
ax1.plot(x_lat, y_lat, z_lat, '--', color=COLORS['orange'], linewidth=2, alpha=0.7, label='Latitude (not geodesic)')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Geodesics on a Sphere\n(Great Circles = Shortest Paths)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)

# Plot 2: Geodesic equation explanation
ax2 = plt.subplot(2, 2, 2)

explanation_text = """
Geodesic Equation

General form:
d²x^μ/dλ² + Γ^μ_νρ (dx^ν/dλ)(dx^ρ/dλ) = 0

For sphere (θ, φ):
d²θ/dλ² + Γ^θ_φφ (dφ/dλ)² = 0
d²φ/dλ² + 2Γ^φ_θφ (dθ/dλ)(dφ/dλ) = 0

With Christoffel symbols:
Γ^θ_φφ = -sin(θ)cos(θ)
Γ^φ_θφ = cot(θ)

Physical interpretation:
• Acceleration = 0 in curved space
• "Straight as possible" given curvature
• Locally minimizes distance

Great circles:
• Only circles on sphere that are geodesics
• All meridians are geodesics
• Equator is a geodesic
• Other latitude lines: NOT geodesics
"""

ax2.text(0.05, 0.95, explanation_text, transform=ax2.transAxes,
        fontsize=9.5, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.axis('off')

# Plot 3: Coordinate evolution
ax3 = plt.subplot(2, 2, 3)

ax3.plot(lambda_vals_3, solution_3[:, 0], color=COLORS['blue'], linewidth=2, label='θ(λ)')
ax3.plot(lambda_vals_3, solution_3[:, 1], color=COLORS['orange'], linewidth=2, label='φ(λ)')

ax3.set_xlabel('Affine parameter λ', fontsize=12)
ax3.set_ylabel('Coordinate value', fontsize=12)
ax3.set_title('Geodesic Coordinates vs Parameter', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Comparison table
ax4 = plt.subplot(2, 2, 4)

comparison_text = """
Geodesics vs Non-Geodesics on Sphere

GEODESICS (Great Circles):
✓ Meridians (longitude lines)
✓ Equator
✓ Any great circle (R = sphere radius)
✓ Shortest distance between two points
✓ Zero geodesic curvature

NON-GEODESICS:
✗ Latitude circles (except equator)
✗ Small circles (r < R)
✗ NOT shortest paths
✗ Non-zero geodesic curvature

Example Distances:
• North pole to South pole:
  - Geodesic (meridian): πR
  - Via equator: 2πR (longer!)

• Two points on equator:
  - Geodesic (equator arc): R×Δφ
  - Via north pole: 2R×(π/2) (longer if Δφ<π)

Flight paths on Earth:
• Planes follow great circles
• Look curved on flat maps (e.g., over Arctic)
• Shortest distance despite appearance!
"""

ax4.text(0.05, 0.95, comparison_text, transform=ax4.transAxes,
        fontsize=9, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.2))
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.axis('off')

plt.tight_layout()
plt.show()

# Additional visualization: geodesic on unwrapped sphere
fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Sphere view
ax5 = plt.subplot(1, 2, 1, projection='3d')
ax5.plot_surface(X_sphere, Y_sphere, Z_sphere, alpha=0.2, color=COLORS['blue'], edgecolor='none')

# Show specific geodesic with start and end points
theta_specific = np.linspace(0.1, np.pi - 0.1, 100)
phi_specific = np.zeros_like(theta_specific)
x_spec = R * np.sin(theta_specific) * np.cos(phi_specific)
y_spec = R * np.sin(theta_specific) * np.sin(phi_specific)
z_spec = R * np.cos(theta_specific)

ax5.plot(x_spec, y_spec, z_spec, color=COLORS['red'], linewidth=4, label='Geodesic')
ax5.plot([x_spec[0]], [y_spec[0]], [z_spec[0]], 'go', markersize=12, label='Start')
ax5.plot([x_spec[-1]], [y_spec[-1]], [z_spec[-1]], 'ro', markersize=12, label='End')

ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_zlabel('z')
ax5.set_title('Geodesic on Sphere (3D View)', fontsize=13, fontweight='bold')
ax5.legend()

# Right: 2D coordinate view
ax6.plot(solution_1[:, 1], solution_1[:, 0], color=COLORS['red'], linewidth=3, label='Meridian')
ax6.plot(solution_2[:, 1], solution_2[:, 0], color=COLORS['green'], linewidth=3, label='Equator')
ax6.plot(solution_3[:, 1], solution_3[:, 0], color=COLORS['purple'], linewidth=3, label='Great circle')

# Non-geodesic latitude
ax6.axhline(y=np.pi/3, color=COLORS['orange'], linestyle='--', linewidth=2, alpha=0.7, label='Latitude (not geodesic)')

ax6.set_xlabel('φ (longitude)', fontsize=12)
ax6.set_ylabel('θ (colatitude)', fontsize=12)
ax6.set_title('Geodesics in (θ, φ) Coordinates', fontsize=13, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_xlim([0, 2*np.pi])
ax6.set_ylim([0, np.pi])

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. Geodesics = 'straightest possible' paths in curved space")
print("2. On sphere: geodesics are great circles (maximum radius)")
print("3. Geodesic equation: d²x^μ/dλ² + Γ^μ_νρ v^ν v^ρ = 0")
print("4. Christoffel symbols determine geodesic curvature")
print("5. Latitude circles (except equator) are NOT geodesics")
print("6. Geodesics minimize distance (locally)")
print("\n✓ In GR: Free-falling particles follow geodesics in spacetime!")
