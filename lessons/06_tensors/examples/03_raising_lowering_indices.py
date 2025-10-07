#!/usr/bin/env python3
"""
Example: Raising and Lowering Indices with the Metric
Demonstrates how the metric converts between contravariant and covariant tensors
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("RAISING AND LOWERING INDICES WITH THE METRIC")
print("="*60)

# 1. EUCLIDEAN SPACE (3D)
print("\n1. EUCLIDEAN SPACE (g_ij = δ_ij)")
print("-" * 60)

g_euclidean = np.eye(3)
g_inv_euclidean = np.linalg.inv(g_euclidean)

print("Metric g_ij:")
print(g_euclidean)
print("\nInverse metric g^ij:")
print(g_inv_euclidean)
print("(Same for Euclidean!)")

# Contravariant vector
v_upper = np.array([2, 3, 4])
print(f"\nContravariant vector: v^i = {v_upper}")

# Lower index
v_lower = g_euclidean @ v_upper
print(f"Lowered: v_i = g_ij v^j = {v_lower}")

# Raise back
v_upper_again = g_inv_euclidean @ v_lower
print(f"Raised back: v^i = g^ij v_j = {v_upper_again}")
print(f"✓ Match: {np.allclose(v_upper, v_upper_again)}")

# 2. MINKOWSKI SPACETIME
print("\n2. MINKOWSKI SPACETIME (η_μν)")
print("-" * 60)

eta = np.diag([-1, 1, 1, 1])
eta_inv = np.diag([-1, 1, 1, 1])  # Self-inverse!

print("Minkowski metric η_μν:")
print(eta)
print("\nInverse metric η^μν:")
print(eta_inv)
print("(Self-inverse! η_μρ η^ρν = δ_μ^ν)")

# 4-velocity of particle at rest
u_upper = np.array([1, 0, 0, 0])  # (dt/dτ, dx/dτ, dy/dτ, dz/dτ)
print(f"\n4-velocity (contravariant): u^μ = {u_upper}")
print("(Particle at rest: only time component)")

# Lower index
u_lower = eta @ u_upper
print(f"\nLowered: u_μ = η_μν u^ν = {u_lower}")
print("Notice: u_0 = -1 (sign flip due to metric signature!)")

# Normalization check
norm = u_upper @ u_lower
print(f"\nNormalization: u^μ u_μ = {norm}")
print("Expected: -1 for massive particle (timelike)")
print("✓ Correct!" if np.isclose(norm, -1) else "✗ Error!")

# 3. MOVING PARTICLE
print("\n3. MOVING PARTICLE (v = 0.6c)")
print("-" * 60)

v = 0.6  # velocity as fraction of c
gamma = 1 / np.sqrt(1 - v**2)  # Lorentz factor

u_moving_upper = np.array([gamma, gamma * v, 0, 0])
print(f"Lorentz factor γ = 1/√(1-v²) = {gamma:.4f}")
print(f"4-velocity: u^μ = γ(1, v, 0, 0) = {u_moving_upper}")

u_moving_lower = eta @ u_moving_upper
print(f"Lowered: u_μ = {u_moving_lower}")

norm_moving = u_moving_upper @ u_moving_lower
print(f"\nNormalization: u^μ u_μ = {norm_moving:.6f}")
print("✓ Still -1 (invariant!)" if np.isclose(norm_moving, -1) else "✗ Error!")

# 4. TENSOR WITH MULTIPLE INDICES
print("\n4. RAISING/LOWERING MULTIPLE INDICES")
print("-" * 60)

# (0,2)-tensor
T_lower_lower = np.array([
    [1, 2],
    [3, 4]
])

g_2d = np.eye(2)
g_2d_inv = np.eye(2)

print("(0,2)-tensor T_μν:")
print(T_lower_lower)

# Raise first index: T^μ_ν = g^μρ T_ρν
T_mixed = g_2d_inv @ T_lower_lower
print("\n(1,1)-tensor T^μ_ν = g^μρ T_ρν:")
print(T_mixed)

# Raise second index: T^μν = g^νσ T^μ_σ
T_upper_upper = T_mixed @ g_2d_inv
print("\n(2,0)-tensor T^μν = g^νσ T^μ_σ:")
print(T_upper_upper)

# Direct: both at once
T_upper_upper_direct = g_2d_inv @ T_lower_lower @ g_2d_inv
print("\nDirect: T^μν = g^μρ g^νσ T_ρσ:")
print(T_upper_upper_direct)
print(f"✓ Match: {np.allclose(T_upper_upper, T_upper_upper_direct)}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Musical isomorphism diagram
ax1 = axes[0, 0]
ax1.text(0.5, 0.8, 'The "Musical Isomorphism"', transform=ax1.transAxes,
        fontsize=16, ha='center', fontweight='bold')

diagram_text = """
Contravariant ←→ Covariant
    v^μ        ↔      v_μ

    ↓ g_μν           ↑ g^μν
  (lower)          (raise)

Vector Space ←→ Dual Space
(tangent)      (cotangent)
"""

ax1.text(0.5, 0.4, diagram_text, transform=ax1.transAxes,
        fontsize=13, ha='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))

ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.axis('off')

# Plot 2: Minkowski metric effect
ax2 = axes[0, 1]

indices = ['0 (t)', '1 (x)', '2 (y)', '3 (z)']
u_upper_plot = u_upper
u_lower_plot = u_lower

x_pos = np.arange(4)
width = 0.35

bars1 = ax2.bar(x_pos - width/2, u_upper_plot, width, label='u^μ (up)', color=COLORS['blue'])
bars2 = ax2.bar(x_pos + width/2, u_lower_plot, width, label='u_μ (down)', color=COLORS['orange'])

ax2.set_xlabel('Index μ', fontsize=12)
ax2.set_ylabel('Component value', fontsize=12)
ax2.set_title('4-Velocity: Raising/Lowering in Minkowski Space', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(indices)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(0, color='k', linewidth=0.8)

# Annotate the sign flip
ax2.annotate('Sign flip!\nη_00 = -1', xy=(0, -0.5), xytext=(0.5, -0.8),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Plot 3: Tensor rank ladder
ax3 = axes[1, 0]

ladder_text = """
Raising/Lowering Tensor Indices

(0,2) → (1,1) → (2,0)
T_μν → T^μ_ν → T^μν

Operations:
• T^μ_ν = g^μρ T_ρν  (raise first)
• T^μν = g^νσ T^μ_σ  (raise second)
• T_μν = g_μρ g_νσ T^ρσ  (lower both)

Inverse operations:
• T_μ^ν = g_μρ T^ρν  (lower first)
• T_μν = g_νσ T_μ^σ  (lower second)

Key property:
g_μρ g^ρν = δ_μ^ν (Kronecker delta)
"""

ax3.text(0.05, 0.95, ladder_text, transform=ax3.transAxes,
        fontsize=11, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.15))

ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.axis('off')

# Plot 4: Example calculation
ax4 = axes[1, 1]

# Show step-by-step lowering for moving particle
calc_text = f"""
Example: Lowering 4-velocity (v=0.6c)

Given: u^μ = [{u_moving_upper[0]:.3f}, {u_moving_upper[1]:.3f}, 0, 0]

Step 1: u_0 = η_0ν u^ν
      = η_00 u^0 + η_01 u^1 + ...
      = (-1)({u_moving_upper[0]:.3f}) + 0 + ...
      = {u_moving_lower[0]:.3f}

Step 2: u_1 = η_1ν u^ν
      = η_10 u^0 + η_11 u^1 + ...
      = 0 + (1)({u_moving_upper[1]:.3f}) + ...
      = {u_moving_lower[1]:.3f}

Result: u_μ = [{u_moving_lower[0]:.3f}, {u_moving_lower[1]:.3f}, 0, 0]

Check: u^μ u_μ = {norm_moving:.6f} ✓
"""

ax4.text(0.05, 0.95, calc_text, transform=ax4.transAxes,
        fontsize=10, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. Metric g_μν is the 'index lowering operator'")
print("2. Inverse metric g^μν is the 'index raising operator'")
print("3. These are called the 'musical isomorphisms' (♯ and ♭)")
print("4. In Euclidean space: raising/lowering doesn't change components")
print("5. In Minkowski space: TIME component flips sign!")
print("6. The operation preserves the geometric object (just changes representation)")
print("\n✓ This allows us to work with whichever form is most convenient!")
