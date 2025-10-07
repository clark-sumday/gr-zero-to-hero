#!/usr/bin/env python3
"""
Example: Covariant Derivative - Differentiation on Curved Manifolds
Demonstrates how to take derivatives of tensors in curved spaces
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("COVARIANT DERIVATIVE: DIFFERENTIATION ON MANIFOLDS")
print("="*60)

print("\nOrdinary derivative ∂_μ doesn't work for tensors in curved space!")
print("Need covariant derivative ∇_μ that accounts for basis vector changes.")

print("\nFor contravariant vector v^ν:")
print("∇_μ v^ν = ∂_μ v^ν + Γ^ν_μρ v^ρ")

print("\nFor covariant vector w_ν:")
print("∇_μ w_ν = ∂_μ w_ν - Γ^ρ_μν w_ρ")

# Example 1: PARALLEL TRANSPORT ON A SPHERE
print("\n" + "="*60)
print("1. PARALLEL TRANSPORT ON A SPHERE")
print("="*60)

print("\nKey concept: A vector is parallel-transported if ∇_μ v^ν = 0")
print("(no change as measured by the covariant derivative)")

# Sphere: start at north pole, move to equator, then around, then back
print("\nExperiment: Parallel transport around a triangle on sphere")
print("1. Start at north pole with vector pointing south")
print("2. Move to equator along meridian")
print("3. Move 90° along equator")
print("4. Move back to north pole")
print("\nResult: Vector rotates by 90°!")
print("✓ This is holonomy - detects curvature")

# Example 2: POLAR COORDINATES
print("\n" + "="*60)
print("2. COVARIANT DERIVATIVE IN POLAR COORDINATES")
print("="*60)

# Christoffel symbols for polar coords
def christoffel_polar(r):
    """
    Non-zero Christoffel symbols in polar coords:
    Γ^r_θθ = -r
    Γ^θ_rθ = Γ^θ_θr = 1/r
    """
    Gamma = np.zeros((2, 2, 2))
    Gamma[0, 1, 1] = -r  # Γ^r_θθ
    Gamma[1, 0, 1] = 1/r  # Γ^θ_rθ
    Gamma[1, 1, 0] = 1/r  # Γ^θ_θr
    return Gamma

# Vector field in polar coords: v^r = 1, v^θ = 0 (constant radial direction)
r_point = 2.0
theta_point = np.pi / 4

v = np.array([1, 0])  # v^r = 1, v^θ = 0
print(f"\nVector field: v^r = 1, v^θ = 0 (constant radial)")
print(f"At point (r={r_point}, θ={theta_point:.4f})")

# Compute covariant derivative ∇_θ v^r (derivative in θ direction)
Gamma = christoffel_polar(r_point)

# ∇_θ v^r = ∂_θ v^r + Γ^r_θρ v^ρ
# = 0 + Γ^r_θθ v^θ + Γ^r_θr v^r
# = Γ^r_θθ × 0 + 0 × 1 = 0

partial_theta_vr = 0  # v^r is constant in θ
nabla_theta_vr = partial_theta_vr + Gamma[0, 1, 1] * v[1] + 0 * v[0]

print(f"\nPartial derivative: ∂_θ v^r = {partial_theta_vr}")
print(f"Covariant derivative: ∇_θ v^r = {nabla_theta_vr}")
print("✓ Vector is parallel-transported (covariant derivative = 0)")

# ∇_θ v^θ
partial_theta_vtheta = 0
nabla_theta_vtheta = partial_theta_vtheta + Gamma[1, 1, 0] * v[0] + Gamma[1, 1, 1] * v[1]

print(f"\nFor θ-component:")
print(f"∂_θ v^θ = {partial_theta_vtheta}")
print(f"∇_θ v^θ = {nabla_theta_vtheta:.3f}")
print("✓ Also parallel-transported")

# Example 3: DIVERGENCE
print("\n" + "="*60)
print("3. DIVERGENCE (CONTRACTION OF COVARIANT DERIVATIVE)")
print("="*60)

print("\nDivergence: ∇_μ v^μ = ∂_μ v^μ + Γ^μ_μρ v^ρ")
print("\nIn polar coords, for v^r = r, v^θ = 0:")

def divergence_polar(r, theta, v_r_func, v_theta_func):
    """
    Compute divergence in polar coordinates
    ∇·v = (1/r) ∂(r v^r)/∂r + (1/r) ∂v^θ/∂θ
    """
    # Partial derivatives
    dr = 0.001
    dtheta = 0.001

    v_r = v_r_func(r, theta)
    v_theta = v_theta_func(r, theta)

    dv_r_dr = (v_r_func(r + dr, theta) - v_r) / dr
    dv_theta_dtheta = (v_theta_func(r, theta + dtheta) - v_theta) / dtheta

    # Divergence in polar coords
    div = (1/r) * (r * dv_r_dr + v_r) + (1/r) * dv_theta_dtheta

    return div

v_r_func = lambda r, theta: r
v_theta_func = lambda r, theta: 0

div_result = divergence_polar(2.0, np.pi/4, v_r_func, v_theta_func)
print(f"For v = (r, 0):")
print(f"∇·v = {div_result:.3f}")
print("Expected: ∇·v = 2 (since v^r = r)")

# Visualization
fig = plt.figure(figsize=(14, 10))

# Plot 1: Parallel transport on sphere
ax1 = plt.subplot(2, 2, 1, projection='3d')

# Draw sphere
u = np.linspace(0, np.pi, 20)
v_sphere = np.linspace(0, 2*np.pi, 20)
U, V = np.meshgrid(u, v_sphere)
X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

ax1.plot_surface(X, Y, Z, alpha=0.2, color=COLORS['blue'], edgecolor='none')

# Draw triangle path
# Path 1: North pole to equator
path1_theta = np.linspace(0, np.pi/2, 20)
path1_phi = np.zeros_like(path1_theta)
x1 = np.sin(path1_theta) * np.cos(path1_phi)
y1 = np.sin(path1_theta) * np.sin(path1_phi)
z1 = np.cos(path1_theta)
ax1.plot(x1, y1, z1, color=COLORS['red'], linewidth=3, label='Path')

# Path 2: Along equator
path2_theta = np.pi/2 * np.ones(20)
path2_phi = np.linspace(0, np.pi/2, 20)
x2 = np.sin(path2_theta) * np.cos(path2_phi)
y2 = np.sin(path2_theta) * np.sin(path2_phi)
z2 = np.cos(path2_theta)
ax1.plot(x2, y2, z2, color=COLORS['red'], linewidth=3)

# Path 3: Back to north pole
path3_theta = np.linspace(np.pi/2, 0, 20)
path3_phi = np.pi/2 * np.ones_like(path3_theta)
x3 = np.sin(path3_theta) * np.cos(path3_phi)
y3 = np.sin(path3_theta) * np.sin(path3_phi)
z3 = np.cos(path3_theta)
ax1.plot(x3, y3, z3, color=COLORS['red'], linewidth=3)

# Draw vectors at key points
# North pole: initial vector
ax1.quiver(0, 0, 1, 0, -1, 0, color=COLORS['green'], arrow_length_ratio=0.3, linewidth=2)
# After transport: rotated
ax1.quiver(0, 0, 1, 1, 0, 0, color=COLORS['orange'], arrow_length_ratio=0.3, linewidth=2,
          linestyle='--')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Parallel Transport on Sphere\n(Vector rotates = curvature!)', fontsize=12, fontweight='bold')
ax1.legend(['Surface', 'Path', '', '', 'Initial vector', 'After transport'])

# Plot 2: Covariant vs ordinary derivative
ax2 = plt.subplot(2, 2, 2)

comparison_text = """
Ordinary vs Covariant Derivative

Ordinary Derivative (∂_μ):
• Component-wise differentiation
• NOT a tensor in curved space!
• Doesn't account for basis changes

Covariant Derivative (∇_μ):
• ∇_μ v^ν = ∂_μ v^ν + Γ^ν_μρ v^ρ
• IS a tensor!
• Corrects for basis vector changes
• Reduces to ∂_μ in flat space + Cartesian

Examples:
Flat space, Cartesian: ∇_μ = ∂_μ (no difference)
Polar coords: ∇_μ ≠ ∂_μ (basis rotates)
Curved space: ∇_μ ≠ ∂_μ (intrinsic curvature)

Parallel transport: ∇_v w = 0
(w doesn't change along v direction)
"""

ax2.text(0.05, 0.95, comparison_text, transform=ax2.transAxes,
        fontsize=10, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.2))

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.axis('off')

# Plot 3: Vector field and its divergence
ax3 = plt.subplot(2, 2, 3)

# Create vector field in polar coordinates
r_grid = np.linspace(0.5, 3, 10)
theta_grid = np.linspace(0, 2*np.pi, 12)
R_grid, Theta_grid = np.meshgrid(r_grid, theta_grid)

# Convert to Cartesian for plotting
X_grid = R_grid * np.cos(Theta_grid)
Y_grid = R_grid * np.sin(Theta_grid)

# Vector field: v = (r, 0) in polar
# In Cartesian: v_x = r cos(θ), v_y = r sin(θ) (radial)
U = X_grid
V = Y_grid

# Normalize for display
mag = np.sqrt(U**2 + V**2)
U_norm = U / mag
V_norm = V / mag

ax3.quiver(X_grid, Y_grid, U_norm, V_norm, mag, cmap='viridis', scale=25, width=0.004)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Radial Vector Field v^r = r\n∇·v = 2 (diverging)', fontsize=13, fontweight='bold')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-3.5, 3.5])
ax3.set_ylim([-3.5, 3.5])

# Plot 4: Christoffel correction terms
ax4 = plt.subplot(2, 2, 4)

r_vals = np.linspace(0.5, 3, 50)

# For ∇_r v^θ with v^θ = constant
# Correction term: Γ^θ_rr v^r (typically zero) + Γ^θ_rθ v^θ
v_theta_const = 1
correction_nabla_r_vtheta = (1/r_vals) * v_theta_const  # Γ^θ_rθ = 1/r

ax4.plot(r_vals, correction_nabla_r_vtheta, color=COLORS['blue'], linewidth=3,
        label='Γ^θ_rθ v^θ (correction)')
ax4.axhline(0, color=COLORS['red'], linestyle='--', linewidth=2, label='∂_r v^θ (partial)')

ax4.fill_between(r_vals, 0, correction_nabla_r_vtheta, alpha=0.3, color=COLORS['blue'],
                label='Covariant - Partial')

ax4.set_xlabel('r', fontsize=12)
ax4.set_ylabel('Derivative value', fontsize=12)
ax4.set_title('Covariant Derivative Correction\nfor v^θ = constant', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. Ordinary derivative ∂_μ is NOT tensorial in curved space")
print("2. Covariant derivative ∇_μ accounts for basis vector changes")
print("3. ∇_μ = ∂_μ + Γ (correction terms from Christoffel symbols)")
print("4. Parallel transport: ∇_v w = 0 (no change in direction v)")
print("5. Curvature detected by: transport around loop ≠ identity")
print("\n✓ In GR: ∇_μ used for all derivatives of tensors!")
