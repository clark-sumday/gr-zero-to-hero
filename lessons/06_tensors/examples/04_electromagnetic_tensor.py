#!/usr/bin/env python3
"""
Example: Electromagnetic Field Tensor
Demonstrates how E and B fields combine into a single antisymmetric tensor F_μν
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

def em_field_tensor(E, B, c=1):
    """
    Construct electromagnetic field tensor F_μν from E and B fields

    Convention:
    - F_0i = E_i/c (electric field)
    - F_ij = -ε_ijk B_k (magnetic field via Levi-Civita)

    Args:
        E: Electric field [Ex, Ey, Ez]
        B: Magnetic field [Bx, By, Bz]
        c: Speed of light (default 1 in natural units)

    Returns:
        4x4 antisymmetric tensor F_μν
    """
    F = np.zeros((4, 4))

    # Time-space components: F_0i = E_i/c
    F[0, 1] = E[0] / c
    F[0, 2] = E[1] / c
    F[0, 3] = E[2] / c

    # Antisymmetry: F_i0 = -F_0i
    F[1, 0] = -F[0, 1]
    F[2, 0] = -F[0, 2]
    F[3, 0] = -F[0, 3]

    # Space-space components: F_ij = -ε_ijk B_k
    F[1, 2] = -B[2]  # F_12 = -B_z
    F[2, 1] = -F[1, 2]

    F[1, 3] = B[1]   # F_13 = B_y
    F[3, 1] = -F[1, 3]

    F[2, 3] = -B[0]  # F_23 = -B_x
    F[3, 2] = -F[2, 3]

    return F

def lorentz_boost_matrix(v, c=1):
    """
    Lorentz boost matrix in x-direction

    Args:
        v: velocity (as fraction of c if c=1)
        c: speed of light

    Returns:
        4x4 Lorentz transformation matrix
    """
    gamma = 1 / np.sqrt(1 - (v/c)**2)
    beta = v / c

    Lambda = np.array([
        [gamma, -gamma*beta, 0, 0],
        [-gamma*beta, gamma, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return Lambda

print("="*60)
print("ELECTROMAGNETIC FIELD TENSOR F_μν")
print("="*60)

# Example 1: Pure electric field
print("\n1. PURE ELECTRIC FIELD (E pointing in x-direction)")
print("-" * 60)

E1 = np.array([3, 0, 0])  # Electric field along x
B1 = np.array([0, 0, 0])  # No magnetic field

F1 = em_field_tensor(E1, B1, c=1)

print(f"E field: {E1}")
print(f"B field: {B1}")
print("\nElectromagnetic tensor F_μν:")
print(F1)
print("\nStructure:")
print("F_μν = [  0   Ex/c  Ey/c  Ez/c ]")
print("       [-Ex/c  0   -Bz    By   ]")
print("       [-Ey/c  Bz   0    -Bx   ]")
print("       [-Ez/c -By   Bx    0    ]")

# Verify antisymmetry
print(f"\nVerify antisymmetry F_μν = -F_νμ: {np.allclose(F1, -F1.T)}")

# Example 2: Pure magnetic field
print("\n2. PURE MAGNETIC FIELD (B pointing in z-direction)")
print("-" * 60)

E2 = np.array([0, 0, 0])
B2 = np.array([0, 0, 2])  # Magnetic field along z

F2 = em_field_tensor(E2, B2, c=1)

print(f"E field: {E2}")
print(f"B field: {B2}")
print("\nElectromagnetic tensor F_μν:")
print(F2)

# Example 3: Both E and B fields
print("\n3. BOTH E AND B FIELDS")
print("-" * 60)

E3 = np.array([1, 2, 0])
B3 = np.array([0, 0, 3])

F3 = em_field_tensor(E3, B3, c=1)

print(f"E field: {E3}")
print(f"B field: {B3}")
print("\nElectromagnetic tensor F_μν:")
print(F3)

# Invariants
print("\n4. ELECTROMAGNETIC INVARIANTS")
print("-" * 60)

# First invariant: F_μν F^μν = 2(B² - E²)
# For simplicity with c=1: just use F_μν F_μν
I1 = 0.5 * np.einsum('ij,ij->', F3, F3)

E_squared = np.dot(E3, E3)
B_squared = np.dot(B3, B3)

print(f"First invariant: ½ F_μν F^μν = {I1:.2f}")
print(f"Expected: B² - E² = {B_squared} - {E_squared} = {B_squared - E_squared}")
print(f"Match: {np.isclose(I1, B_squared - E_squared)}")

print("\nPhysical meaning:")
if I1 < 0:
    print("  I1 < 0 → Electric field dominates (E² > B²)")
elif I1 > 0:
    print("  I1 > 0 → Magnetic field dominates (B² > E²)")
else:
    print("  I1 = 0 → E and B have equal magnitude")

# Example 4: Lorentz transformation
print("\n5. LORENTZ TRANSFORMATION OF FIELDS")
print("-" * 60)
print("Key insight: E and B mix under boosts!")

# Start with pure E field
E_rest = np.array([5, 0, 0])
B_rest = np.array([0, 0, 0])
F_rest = em_field_tensor(E_rest, B_rest, c=1)

print(f"\nRest frame:")
print(f"  E = {E_rest}")
print(f"  B = {B_rest}")

# Boost in y-direction (perpendicular to E)
v_boost = 0.6  # 60% of c
Lambda = lorentz_boost_matrix(v_boost, c=1)

# Transform: F'_μν = Λ_μ^ρ Λ_ν^σ F_ρσ
F_boosted = Lambda @ F_rest @ Lambda.T

print(f"\nBoosted frame (v = {v_boost}c in x-direction):")
print(f"F'_μν =")
print(F_boosted)

# Extract E' and B' from boosted tensor
E_boosted_x = F_boosted[0, 1]
E_boosted_y = F_boosted[0, 2]
E_boosted_z = F_boosted[0, 3]
B_boosted_x = -F_boosted[2, 3]
B_boosted_y = F_boosted[1, 3]
B_boosted_z = -F_boosted[1, 2]

print(f"\nExtracted fields in boosted frame:")
print(f"  E' = [{E_boosted_x:.3f}, {E_boosted_y:.3f}, {E_boosted_z:.3f}]")
print(f"  B' = [{B_boosted_x:.3f}, {B_boosted_y:.3f}, {B_boosted_z:.3f}]")
print("\n✓ Notice: Pure E field in rest frame → E + B in moving frame!")

# Visualization
fig = plt.figure(figsize=(14, 10))

# Plot 1: Tensor structure
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(F3, cmap='RdBu_r', vmin=-3, vmax=3)
ax1.set_title('EM Tensor F_μν Structure', fontsize=14, fontweight='bold')
ax1.set_xticks([0, 1, 2, 3])
ax1.set_yticks([0, 1, 2, 3])
ax1.set_xticklabels(['t', 'x', 'y', 'z'])
ax1.set_yticklabels(['t', 'x', 'y', 'z'])

for i in range(4):
    for j in range(4):
        text = ax1.text(j, i, f'{F3[i,j]:.1f}', ha="center", va="center",
                       color='white' if abs(F3[i,j]) > 1.5 else 'black',
                       fontsize=11, fontweight='bold')

plt.colorbar(im1, ax=ax1)

# Add labels
ax1.text(1.5, -0.8, 'E-field components', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor=COLORS['orange'], alpha=0.5))
ax1.text(4.8, 2, 'B-field\ncomponents', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor=COLORS['blue'], alpha=0.5))

# Plot 2: Antisymmetric structure
ax2 = plt.subplot(2, 2, 2)

antisym_text = """
Antisymmetric Tensor

Properties:
• F_μν = -F_νμ
• F_μμ = 0 (diagonal is zero)
• 6 independent components:
  - 3 from E field (F_0i)
  - 3 from B field (F_ij)

Maxwell's Equations (tensor form):
• ∂_μ F^μν = J^ν (source)
• ∂_μ F*^μν = 0 (no monopoles)

where F*_μν is the dual tensor
"""

ax2.text(0.05, 0.95, antisym_text, transform=ax2.transAxes,
        fontsize=11, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.axis('off')

# Plot 3: E and B field vectors
ax3 = plt.subplot(2, 2, 3, projection='3d')

# Draw coordinate axes
ax3.plot([0, 2], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.3)
ax3.plot([0, 0], [0, 2], [0, 0], 'k-', linewidth=1, alpha=0.3)
ax3.plot([0, 0], [0, 0], [0, 2], 'k-', linewidth=1, alpha=0.3)

# Draw E field
if np.linalg.norm(E3) > 0:
    ax3.quiver(0, 0, 0, E3[0], E3[1], E3[2],
              color=COLORS['red'], arrow_length_ratio=0.2, linewidth=3,
              label=f'E = {E3}')

# Draw B field
if np.linalg.norm(B3) > 0:
    ax3.quiver(0, 0, 0, B3[0], B3[1], B3[2],
              color=COLORS['blue'], arrow_length_ratio=0.2, linewidth=3,
              label=f'B = {B3}')

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_title('E and B Fields in 3D', fontsize=13, fontweight='bold')
ax3.legend()
ax3.set_xlim([0, 3])
ax3.set_ylim([0, 3])
ax3.set_zlim([0, 3])

# Plot 4: Invariant comparison
ax4 = plt.subplot(2, 2, 4)

# Create several field configurations
configs = [
    ("Pure E", np.array([3, 0, 0]), np.array([0, 0, 0])),
    ("Pure B", np.array([0, 0, 0]), np.array([0, 0, 3])),
    ("E > B", np.array([4, 0, 0]), np.array([0, 0, 2])),
    ("E < B", np.array([2, 0, 0]), np.array([0, 0, 4])),
    ("E = B", np.array([3, 0, 0]), np.array([0, 0, 3])),
]

labels = []
E_mags = []
B_mags = []
invariants = []

for name, E, B in configs:
    labels.append(name)
    E_mags.append(np.linalg.norm(E))
    B_mags.append(np.linalg.norm(B))
    F = em_field_tensor(E, B)
    I = 0.5 * np.einsum('ij,ij->', F, F)
    invariants.append(I)

x_pos = np.arange(len(labels))
width = 0.25

ax4.bar(x_pos - width, E_mags, width, label='|E|', color=COLORS['red'], alpha=0.7)
ax4.bar(x_pos, B_mags, width, label='|B|', color=COLORS['blue'], alpha=0.7)
ax4.bar(x_pos + width, invariants, width, label='B²-E²', color=COLORS['green'], alpha=0.7)

ax4.set_xlabel('Configuration', fontsize=12)
ax4.set_ylabel('Magnitude', fontsize=12)
ax4.set_title('EM Invariant: B² - E²', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(labels, rotation=15, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(0, color='k', linewidth=0.8)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. E and B fields are components of single tensor F_μν")
print("2. F_μν is antisymmetric: 6 independent components (3 E + 3 B)")
print("3. Under Lorentz boosts: E and B fields MIX")
print("4. Invariants: B² - E² and E·B are frame-independent")
print("5. Maxwell's equations: ∂_μ F^μν = J^ν (elegant tensor form!)")
print("\n✓ This unification hints at spacetime as fundamental!")
