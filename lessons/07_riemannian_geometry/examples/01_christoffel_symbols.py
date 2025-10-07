#!/usr/bin/env python3
"""
Example: Christoffel Symbols - Connection Coefficients
Demonstrates how to compute Christoffel symbols from the metric tensor
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, Matrix, diff, simplify, sin, cos, sqrt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("CHRISTOFFEL SYMBOLS: CONNECTION COEFFICIENTS")
print("="*60)

print("\nChristoffel symbols Γ^ρ_μν describe how basis vectors change:")
print("∂e_μ/∂x^ν = Γ^ρ_μν e_ρ")
print("\nFormula: Γ^ρ_μν = ½ g^ρσ (∂g_σμ/∂x^ν + ∂g_σν/∂x^μ - ∂g_μν/∂x^σ)")

# Example 1: POLAR COORDINATES
print("\n" + "="*60)
print("1. POLAR COORDINATES (r, θ)")
print("="*60)

r, theta = symbols('r theta', real=True, positive=True)

# Metric: ds² = dr² + r² dθ²
g_polar = Matrix([
    [1, 0],
    [0, r**2]
])

print("Metric g_ij:")
print(g_polar)
print("\nLine element: ds² = dr² + r² dθ²")

# Inverse metric
g_polar_inv = g_polar.inv()
print("\nInverse metric g^ij:")
print(g_polar_inv)

# Compute Christoffel symbols
dim = 2
coords = [r, theta]

def compute_christoffel(g, g_inv, coords):
    """Compute Christoffel symbols from metric"""
    dim = len(coords)
    Gamma = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]

    for rho in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                term = 0
                for sigma in range(dim):
                    dg_sigma_mu = diff(g[sigma, mu], coords[nu])
                    dg_sigma_nu = diff(g[sigma, nu], coords[mu])
                    dg_mu_nu = diff(g[mu, nu], coords[sigma])

                    term += g_inv[rho, sigma] * (dg_sigma_mu + dg_sigma_nu - dg_mu_nu)

                Gamma[rho][mu][nu] = simplify(term / 2)

    return Gamma

Gamma_polar = compute_christoffel(g_polar, g_polar_inv, coords)

print("\nNon-zero Christoffel symbols:")
print(f"Γ^r_θθ = {Gamma_polar[0][1][1]}")  # rho=r, mu=theta, nu=theta
print(f"Γ^θ_rθ = Γ^θ_θr = {Gamma_polar[1][0][1]}")  # rho=theta, mu=r, nu=theta

print("\nPhysical interpretation:")
print("• Γ^r_θθ = -r: Centrifugal force (moving in θ pushes you radially)")
print("• Γ^θ_rθ = 1/r: Angular velocity correction (basis vectors rotate)")

# Example 2: SPHERICAL COORDINATES ON A SPHERE
print("\n" + "="*60)
print("2. SPHERE SURFACE (θ, φ) with radius R=1")
print("="*60)

theta_sphere, phi = symbols('theta phi', real=True, positive=True)

# Metric on unit sphere: ds² = dθ² + sin²θ dφ²
g_sphere = Matrix([
    [1, 0],
    [0, sin(theta_sphere)**2]
])

print("Metric g_ij:")
print(g_sphere)
print("\nLine element: ds² = dθ² + sin²θ dφ²")

g_sphere_inv = g_sphere.inv()
print("\nInverse metric g^ij:")
print(g_sphere_inv)

coords_sphere = [theta_sphere, phi]
Gamma_sphere = compute_christoffel(g_sphere, g_sphere_inv, coords_sphere)

print("\nNon-zero Christoffel symbols:")
print(f"Γ^θ_φφ = {Gamma_sphere[0][1][1]}")  # rho=theta, mu=phi, nu=phi
print(f"Γ^φ_θφ = Γ^φ_φθ = {Gamma_sphere[1][0][1]}")  # rho=phi, mu=theta, nu=phi

print("\nPhysical interpretation:")
print("• Γ^θ_φφ = -sin(θ)cos(θ): Moving along φ pulls you toward equator")
print("• Γ^φ_θφ = cot(θ): Longitude lines converge at poles")

# Numerical evaluation at specific point
r_val = 2.0
theta_val = np.pi / 4  # 45 degrees

print("\n" + "="*60)
print("3. NUMERICAL EVALUATION")
print("="*60)

# Polar coordinates at r=2, θ=π/4
Gamma_r_theta_theta_val = -r_val
Gamma_theta_r_theta_val = 1 / r_val

print(f"\nPolar coordinates at (r={r_val}, θ=π/4):")
print(f"Γ^r_θθ = {Gamma_r_theta_theta_val}")
print(f"Γ^θ_rθ = {Gamma_theta_r_theta_val:.3f}")

# Sphere at θ=π/4
theta_sphere_val = np.pi / 4
Gamma_theta_phi_phi_val = -np.sin(theta_sphere_val) * np.cos(theta_sphere_val)
Gamma_phi_theta_phi_val = np.cos(theta_sphere_val) / np.sin(theta_sphere_val)

print(f"\nSphere at θ=π/4:")
print(f"Γ^θ_φφ = {Gamma_theta_phi_phi_val:.4f}")
print(f"Γ^φ_θφ = {Gamma_phi_theta_phi_val:.4f}")

# Visualization
fig = plt.figure(figsize=(14, 10))

# Plot 1: Polar basis vectors at different points
ax1 = plt.subplot(2, 2, 1)

# Draw circles at different radii
theta_circle = np.linspace(0, 2*np.pi, 100)
for r_circle in [1, 2, 3]:
    x_circle = r_circle * np.cos(theta_circle)
    y_circle = r_circle * np.sin(theta_circle)
    ax1.plot(x_circle, y_circle, 'gray', alpha=0.3, linewidth=0.5)

# Show basis vectors at a point
r_point = 2
theta_point = np.pi / 4
x_point = r_point * np.cos(theta_point)
y_point = r_point * np.sin(theta_point)

ax1.plot(x_point, y_point, 'ro', markersize=10, label=f'Point (r={r_point}, θ=π/4)')

# e_r (radial basis)
e_r = np.array([np.cos(theta_point), np.sin(theta_point)])
ax1.arrow(x_point, y_point, 0.8*e_r[0], 0.8*e_r[1],
         head_width=0.2, head_length=0.15, fc=COLORS['blue'], ec=COLORS['blue'],
         linewidth=2, label='e_r')

# e_θ (tangent basis, perpendicular to radial)
e_theta = np.array([-np.sin(theta_point), np.cos(theta_point)])
ax1.arrow(x_point, y_point, 0.8*e_theta[0], 0.8*e_theta[1],
         head_width=0.2, head_length=0.15, fc=COLORS['orange'], ec=COLORS['orange'],
         linewidth=2, label='e_θ')

ax1.text(x_point + 1.2*e_r[0], y_point + 1.2*e_r[1], 'e_r', fontsize=11, color=COLORS['blue'])
ax1.text(x_point + 1.2*e_theta[0], y_point + 1.2*e_theta[1], 'e_θ', fontsize=11, color=COLORS['orange'])

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Polar Coordinate Basis Vectors\n(basis changes with position!)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.set_xlim([-1, 4])
ax1.set_ylim([-1, 4])

# Plot 2: Christoffel symbol values
ax2 = plt.subplot(2, 2, 2)

r_values = np.linspace(0.5, 3, 50)
Gamma_r_theta_theta_values = -r_values
Gamma_theta_r_theta_values = 1 / r_values

ax2.plot(r_values, Gamma_r_theta_theta_values, color=COLORS['blue'], linewidth=3,
        label='Γ^r_θθ = -r')
ax2.plot(r_values, Gamma_theta_r_theta_values, color=COLORS['orange'], linewidth=3,
        label='Γ^θ_rθ = 1/r')

ax2.set_xlabel('r', fontsize=12)
ax2.set_ylabel('Christoffel symbol value', fontsize=12)
ax2.set_title('Polar Christoffel Symbols vs Radius', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='k', linewidth=0.8)

# Plot 3: Sphere with geodesics
ax3 = plt.subplot(2, 2, 3, projection='3d')

u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2*np.pi, 30)
U, V = np.meshgrid(u, v)

X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

ax3.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis', edgecolor='none')

# Draw a geodesic (great circle)
phi_geodesic = np.linspace(0, 2*np.pi, 100)
theta_geodesic = np.pi / 4 * np.ones_like(phi_geodesic)
x_geo = np.sin(theta_geodesic) * np.cos(phi_geodesic)
y_geo = np.sin(theta_geodesic) * np.sin(phi_geodesic)
z_geo = np.cos(theta_geodesic)
ax3.plot(x_geo, y_geo, z_geo, color=COLORS['red'], linewidth=3, label='Great circle')

# Draw meridian
phi_meridian = 0
theta_meridian = np.linspace(0, np.pi, 100)
x_mer = np.sin(theta_meridian) * np.cos(phi_meridian)
y_mer = np.sin(theta_meridian) * np.sin(phi_meridian)
z_mer = np.cos(theta_meridian)
ax3.plot(x_mer, y_mer, z_mer, color=COLORS['blue'], linewidth=3, label='Meridian')

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_title('Sphere: Christoffel Symbols\nDetermine Geodesics', fontsize=12, fontweight='bold')
ax3.legend()

# Plot 4: Summary
ax4 = plt.subplot(2, 2, 4)

summary_text = """
Christoffel Symbols Γ^ρ_μν

Physical Meaning:
• Describe how basis vectors change
• Not tensors! (depend on coordinates)
• Zero in flat space with Cartesian coords
• Non-zero in curved coords or curved space

Formula:
Γ^ρ_μν = ½g^ρσ(∂g_σμ/∂x^ν + ∂g_σν/∂x^μ - ∂g_μν/∂x^σ)

Symmetry:
• Γ^ρ_μν = Γ^ρ_νμ (symmetric in lower indices)

Uses:
• Covariant derivative: ∇_μ v^ν = ∂_μ v^ν + Γ^ν_μρ v^ρ
• Geodesic equation: d²x^μ/dλ² + Γ^μ_νρ dx^ν/dλ dx^ρ/dλ = 0
• Curvature: Built from Christoffel symbols

Examples:
Polar: Γ^r_θθ = -r, Γ^θ_rθ = 1/r
Sphere: Γ^θ_φφ = -sin(θ)cos(θ), Γ^φ_θφ = cot(θ)
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=9.5, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))

ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. Christoffel symbols = connection coefficients")
print("2. Tell us how basis vectors change from point to point")
print("3. NOT tensors (transform differently)")
print("4. Can be non-zero even in flat space (curved coordinates)")
print("5. Essential for covariant derivatives and geodesics")
print("\n✓ In GR: Christoffel symbols give 'gravitational force'!")
