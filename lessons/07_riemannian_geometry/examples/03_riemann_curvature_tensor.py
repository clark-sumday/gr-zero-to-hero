#!/usr/bin/env python3
"""
Example: Riemann Curvature Tensor - Measuring Spacetime Curvature
Demonstrates how the Riemann tensor quantifies curvature
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy import symbols, Matrix, diff, simplify, sin, cos
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("RIEMANN CURVATURE TENSOR: QUANTIFYING CURVATURE")
print("="*60)

print("\nThe Riemann tensor R^ρ_σμν measures curvature:")
print("• Tests if parallel transport around closed loops returns to original vector")
print("• Built from Christoffel symbols and their derivatives")
print("• In 4D: 20 independent components (due to symmetries)")

print("\nFormula (simplified):")
print("R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ")

print("\nKey property:")
print("[∇_μ, ∇_ν] v^ρ = R^ρ_σμν v^σ")
print("(Covariant derivatives don't commute in curved space!)")

# Example 1: SPHERE (intrinsically curved)
print("\n" + "="*60)
print("1. UNIT SPHERE (R=1)")
print("="*60)

theta, phi = symbols('theta phi', real=True, positive=True)

# Metric: ds² = dθ² + sin²θ dφ²
g_sphere = Matrix([
    [1, 0],
    [0, sin(theta)**2]
])

print("Metric g_ij:")
print(g_sphere)

g_inv = g_sphere.inv()
print("\nInverse metric:")
print(g_inv)

# Compute Christoffel symbols
coords = [theta, phi]
dim = 2

def compute_christoffel(g, g_inv, coords):
    """Compute Christoffel symbols"""
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

Gamma = compute_christoffel(g_sphere, g_inv, coords)

print("\nNon-zero Christoffel symbols:")
print(f"Γ^θ_φφ = {Gamma[0][1][1]}")
print(f"Γ^φ_θφ = Γ^φ_φθ = {Gamma[1][0][1]}")

# Compute Riemann tensor (simplified for 2D)
def compute_riemann_2d(Gamma, coords):
    """Compute Riemann tensor in 2D"""
    dim = 2
    R = [[[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]

    for rho in range(dim):
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    # R^rho_sigma,mu,nu = ∂_mu Γ^rho_nu,sigma - ∂_nu Γ^rho_mu,sigma + ...
                    term1 = diff(Gamma[rho][nu][sigma], coords[mu])
                    term2 = diff(Gamma[rho][mu][sigma], coords[nu])

                    term3 = 0
                    term4 = 0
                    for lam in range(dim):
                        term3 += Gamma[rho][mu][lam] * Gamma[lam][nu][sigma]
                        term4 += Gamma[rho][nu][lam] * Gamma[lam][mu][sigma]

                    R[rho][sigma][mu][nu] = simplify(term1 - term2 + term3 - term4)

    return R

R_sphere = compute_riemann_2d(Gamma, coords)

print("\nRiemann tensor components:")
# In 2D, only one independent component
print(f"R^θ_φθφ = {R_sphere[0][1][0][1]}")

# Gaussian curvature K = R^θ_φθφ / g_θθ g_φφ
K_sphere = R_sphere[0][1][0][1] / (g_sphere[0,0] * g_sphere[1,1])
K_simplified = simplify(K_sphere)

print(f"\nGaussian curvature K = R^θ_φθφ / (g_θθ g_φφ)")
print(f"K = {K_simplified}")
print("For unit sphere: K = 1 (constant positive curvature)")

# Example 2: FLAT SPACE IN POLAR COORDINATES
print("\n" + "="*60)
print("2. FLAT PLANE IN POLAR COORDINATES")
print("="*60)

r, theta_polar = symbols('r theta', real=True, positive=True)

g_polar = Matrix([
    [1, 0],
    [0, r**2]
])

print("Metric g_ij:")
print(g_polar)

g_polar_inv = g_polar.inv()
coords_polar = [r, theta_polar]

Gamma_polar = compute_christoffel(g_polar, g_polar_inv, coords_polar)

print("\nChristoffel symbols:")
print(f"Γ^r_θθ = {Gamma_polar[0][1][1]}")
print(f"Γ^θ_rθ = {Gamma_polar[1][0][1]}")

R_polar = compute_riemann_2d(Gamma_polar, coords_polar)

print("\nRiemann tensor:")
print(f"R^r_θrθ = {R_polar[0][1][0][1]}")
print("All components are ZERO → flat space!")
print("(Christoffel symbols non-zero, but curvature is zero)")

# Numerical evaluation
print("\n" + "="*60)
print("3. NUMERICAL COMPARISON")
print("="*60)

theta_val = np.pi / 4
print(f"\nSphere at θ = π/4:")
K_numerical = float(K_simplified.subs(theta, theta_val))
print(f"Gaussian curvature K = {K_numerical:.6f}")
print("Expected: K = 1 for unit sphere ✓")

# Visualization
fig = plt.figure(figsize=(14, 11))

# Plot 1: Flat space (zero curvature)
ax1 = plt.subplot(2, 3, 1)
x_flat = np.linspace(-3, 3, 20)
y_flat = np.linspace(-3, 3, 20)
X_flat, Y_flat = np.meshgrid(x_flat, y_flat)
Z_flat = np.zeros_like(X_flat)

ax1.contour(X_flat, Y_flat, Z_flat, levels=10, colors='gray', alpha=0.3)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.set_title('Flat Space: R = 0\nK = 0', fontsize=12, fontweight='bold')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Draw parallel transport loop (no rotation)
square = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]])
ax1.plot(square[:, 0], square[:, 1], 'b-', linewidth=3, label='Path')
# Vectors at corners (no change)
for i in range(4):
    ax1.arrow(square[i, 0], square[i, 1], 0.3, 0, head_width=0.2, head_length=0.15,
             fc=COLORS['red'], ec=COLORS['red'], linewidth=2)
ax1.legend()

# Plot 2: Positive curvature (sphere)
ax2 = plt.subplot(2, 3, 2, projection='3d')

u_sphere = np.linspace(0, np.pi, 30)
v_sphere = np.linspace(0, 2*np.pi, 30)
U_sphere, V_sphere = np.meshgrid(u_sphere, v_sphere)
X_sphere = np.sin(U_sphere) * np.cos(V_sphere)
Y_sphere = np.sin(U_sphere) * np.sin(V_sphere)
Z_sphere = np.cos(U_sphere)

ax2.plot_surface(X_sphere, Y_sphere, Z_sphere, alpha=0.6, cmap='viridis', edgecolor='none')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Sphere: R ≠ 0\nK = 1/R² > 0', fontsize=12, fontweight='bold')

# Plot 3: Negative curvature (saddle)
ax3 = plt.subplot(2, 3, 3, projection='3d')

x_saddle = np.linspace(-2, 2, 30)
y_saddle = np.linspace(-2, 2, 30)
X_saddle, Y_saddle = np.meshgrid(x_saddle, y_saddle)
Z_saddle = X_saddle**2 - Y_saddle**2  # Hyperbolic paraboloid

ax3.plot_surface(X_saddle, Y_saddle, Z_saddle, alpha=0.6, cmap='plasma', edgecolor='none')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_title('Saddle: R ≠ 0\nK < 0', fontsize=12, fontweight='bold')

# Plot 4: Riemann tensor structure
ax4 = plt.subplot(2, 3, 4)

riemann_text = """
Riemann Tensor R^ρ_σμν

Symmetries (reduce components):
• R^ρ_σμν = -R^ρ_σνμ  (antisym in μν)
• R^ρ_σμν = -R^ρ_μσν  (antisym in first pair)
• R^ρ_σμν = R^μν_ρσ   (pair exchange)
• R^ρ_σμν + R^ρ_μνσ + R^ρ_νσμ = 0  (Bianchi)

In 4D spacetime:
• 4⁴ = 256 components naively
• After symmetries: 20 independent

Contractions:
• Ricci: R_μν = R^ρ_μρν
• Scalar: R = R^μ_μ = g^μν R_μν
"""

ax4.text(0.05, 0.95, riemann_text, transform=ax4.transAxes,
        fontsize=9.5, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.axis('off')

# Plot 5: Parallel transport comparison
ax5 = plt.subplot(2, 3, 5)

theta_transport = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta_transport)
circle_y = np.sin(theta_transport)

ax5.plot(circle_x, circle_y, 'b-', linewidth=3, label='Path (circle)')

# Show vector rotating during parallel transport
n_vectors = 8
for i in range(n_vectors):
    angle = 2*np.pi * i / n_vectors
    x = np.cos(angle)
    y = np.sin(angle)
    # Vector tangent to circle (parallel transported)
    vx = -np.sin(angle)
    vy = np.cos(angle)
    ax5.arrow(x, y, 0.3*vx, 0.3*vy, head_width=0.1, head_length=0.08,
             fc=COLORS['red'], ec=COLORS['red'], linewidth=1.5)

ax5.set_xlabel('x', fontsize=11)
ax5.set_ylabel('y', fontsize=11)
ax5.set_title('Parallel Transport in Flat Space\n(Vector rotates due to path curvature)', fontsize=11, fontweight='bold')
ax5.set_aspect('equal')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Curvature detection
ax6 = plt.subplot(2, 3, 6)

detection_text = """
Detecting Curvature

Method 1: Parallel Transport
• Transport vector around closed loop
• If returns different → CURVED
• Angle change ∝ enclosed area × curvature

Method 2: Commutator Test
• [∇_μ, ∇_ν]v^ρ = R^ρ_σμν v^σ
• If derivatives commute (R=0) → FLAT
• If don't commute (R≠0) → CURVED

Method 3: Geodesic Deviation
• Initially parallel geodesics converge/diverge
• Rate ∝ Riemann tensor

Physical Meaning:
• Flat: R^ρ_σμν = 0 everywhere
• Sphere: R^ρ_σμν ≠ 0 (positive K)
• Saddle: R^ρ_σμν ≠ 0 (negative K)

In GR: R^ρ_σμν encodes tidal forces!
"""

ax6.text(0.05, 0.95, detection_text, transform=ax6.transAxes,
        fontsize=9, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.2))
ax6.set_xlim([0, 1])
ax6.set_ylim([0, 1])
ax6.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. Riemann tensor R^ρ_σμν is THE measure of curvature")
print("2. Built from Christoffel symbols (connection)")
print("3. R = 0 everywhere ↔ flat space (intrinsically)")
print("4. R ≠ 0 even if Γ = 0 at a point (can't make Γ=0 everywhere if curved)")
print("5. Parallel transport around loop detects curvature")
print("6. Contractions: Ricci tensor R_μν, scalar curvature R")
print("\n✓ In GR: Einstein's equations determine R^ρ_σμν from matter!")
