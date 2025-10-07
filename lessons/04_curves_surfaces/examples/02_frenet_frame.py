#!/usr/bin/env python3
"""
Example: Frenet-Serret Frame
Demonstrates the moving coordinate system along a curve (T, N, B vectors)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Use a helix for demonstration
def curve(t, a=1.0, b=0.3):
    """Helix: r(t) = (a·cos(t), a·sin(t), b·t)"""
    x = a * np.cos(t)
    y = a * np.sin(t)
    z = b * t
    return np.array([x, y, z])

def curve_derivative(t, a=1.0, b=0.3):
    """r'(t)"""
    x = -a * np.sin(t)
    y = a * np.cos(t)
    z = b * np.ones_like(t)
    return np.array([x, y, z])

def curve_second_derivative(t, a=1.0, b=0.3):
    """r''(t)"""
    x = -a * np.cos(t)
    y = -a * np.sin(t)
    z = np.zeros_like(t)
    return np.array([x, y, z])

def frenet_frame(r_prime, r_double_prime):
    """
    Compute Frenet-Serret frame:
    T = tangent vector (unit velocity)
    N = normal vector (points toward center of curvature)
    B = binormal vector (perpendicular to osculating plane)

    Returns: T, N, B (all unit vectors)
    """
    # Tangent: T = r' / |r'|
    speed = np.linalg.norm(r_prime, axis=0)
    T = r_prime / speed

    # Derivative of T with respect to arc length
    dT = r_double_prime / speed[:, np.newaxis].T

    # Remove component parallel to T (make it perpendicular)
    dT_perp = dT - np.sum(dT * T, axis=0) * T

    # Normal: N = dT/ds / |dT/ds|
    dT_perp_norm = np.linalg.norm(dT_perp, axis=0)
    N = dT_perp / dT_perp_norm

    # Binormal: B = T × N
    B = np.cross(T.T, N.T).T

    return T, N, B

# Parameters
a, b = 1.0, 0.3

print("Frenet-Serret Frame")
print("=" * 60)
print("Moving orthonormal frame along a curve:")
print()
print("T (Tangent): Points in direction of motion")
print("  T = r' / |r'|")
print()
print("N (Normal): Points toward center of curvature")
print("  N = (dT/ds) / |dT/ds|")
print()
print("B (Binormal): Perpendicular to osculating plane")
print("  B = T × N")
print()
print("These form a right-handed orthonormal basis at each point!")

# Compute curve
t = np.linspace(0, 4*np.pi, 500)
r = curve(t, a, b)
r_prime = curve_derivative(t, a, b)
r_double_prime = curve_second_derivative(t, a, b)

# Compute Frenet frame
T, N, B = frenet_frame(r_prime, r_double_prime)

# Verify orthonormality at a sample point
idx = 100
T_sample = T[:, idx]
N_sample = N[:, idx]
B_sample = B[:, idx]

print(f"\nVerification at t = {t[idx]:.3f}:")
print(f"  |T| = {np.linalg.norm(T_sample):.6f} (should be 1)")
print(f"  |N| = {np.linalg.norm(N_sample):.6f} (should be 1)")
print(f"  |B| = {np.linalg.norm(B_sample):.6f} (should be 1)")
print(f"  T·N = {np.dot(T_sample, N_sample):.6f} (should be 0)")
print(f"  T·B = {np.dot(T_sample, B_sample):.6f} (should be 0)")
print(f"  N·B = {np.dot(N_sample, B_sample):.6f} (should be 0)")

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: 3D curve with Frenet frame at selected points
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(r[0], r[1], r[2], color=COLORS['gray'], linewidth=2, alpha=0.5)

# Show frames at several points
t_selected = np.linspace(np.pi, 3*np.pi, 6)
scale = 0.5  # Scale for visualization

for ts in t_selected:
    idx = np.argmin(np.abs(t - ts))
    pos = r[:, idx]
    T_vec = T[:, idx] * scale
    N_vec = N[:, idx] * scale
    B_vec = B[:, idx] * scale

    # Draw the three vectors
    ax1.quiver(pos[0], pos[1], pos[2], T_vec[0], T_vec[1], T_vec[2],
              color=COLORS['red'], arrow_length_ratio=0.3, linewidth=2)
    ax1.quiver(pos[0], pos[1], pos[2], N_vec[0], N_vec[1], N_vec[2],
              color=COLORS['green'], arrow_length_ratio=0.3, linewidth=2)
    ax1.quiver(pos[0], pos[1], pos[2], B_vec[0], B_vec[1], B_vec[2],
              color=COLORS['blue'], arrow_length_ratio=0.3, linewidth=2)

    ax1.scatter([pos[0]], [pos[1]], [pos[2]], color='black', s=50, zorder=5)

# Add legend (dummy plots)
ax1.plot([], [], color=COLORS['red'], linewidth=2, label='T (tangent)')
ax1.plot([], [], color=COLORS['green'], linewidth=2, label='N (normal)')
ax1.plot([], [], color=COLORS['blue'], linewidth=2, label='B (binormal)')

ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.set_zlabel('z', fontsize=11)
ax1.set_title('Frenet-Serret Frame Along Helix', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)

# Plot 2: Single frame in detail
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
idx_detail = 200
pos = r[:, idx_detail]
T_vec = T[:, idx_detail] * 1.0
N_vec = N[:, idx_detail] * 1.0
B_vec = B[:, idx_detail] * 1.0

# Draw curve segment
segment_idx = slice(idx_detail-50, idx_detail+50)
ax2.plot(r[0, segment_idx], r[1, segment_idx], r[2, segment_idx],
        color=COLORS['gray'], linewidth=3)

# Draw frame
ax2.quiver(pos[0], pos[1], pos[2], T_vec[0], T_vec[1], T_vec[2],
          color=COLORS['red'], arrow_length_ratio=0.2, linewidth=3, label='T')
ax2.quiver(pos[0], pos[1], pos[2], N_vec[0], N_vec[1], N_vec[2],
          color=COLORS['green'], arrow_length_ratio=0.2, linewidth=3, label='N')
ax2.quiver(pos[0], pos[1], pos[2], B_vec[0], B_vec[1], B_vec[2],
          color=COLORS['blue'], arrow_length_ratio=0.2, linewidth=3, label='B')

# Draw osculating plane (spanned by T and N)
plane_scale = 0.8
plane_T = np.linspace(-plane_scale, plane_scale, 10)
plane_N = np.linspace(-plane_scale, plane_scale, 10)
PT, PN = np.meshgrid(plane_T, plane_N)
plane_x = pos[0] + PT * T_vec[0] + PN * N_vec[0]
plane_y = pos[1] + PT * T_vec[1] + PN * N_vec[1]
plane_z = pos[2] + PT * T_vec[2] + PN * N_vec[2]
ax2.plot_surface(plane_x, plane_y, plane_z, alpha=0.2, color=COLORS['yellow'])

ax2.scatter([pos[0]], [pos[1]], [pos[2]], color='black', s=100, zorder=5)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y', fontsize=11)
ax2.set_zlabel('z', fontsize=11)
ax2.set_title('Osculating Plane (spanned by T and N)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)

# Plot 3: T vector components over time
ax3 = plt.subplot(2, 3, 3)
ax3.plot(t, T[0], color=COLORS['red'], linewidth=2, label='T_x')
ax3.plot(t, T[1], color=COLORS['green'], linewidth=2, label='T_y')
ax3.plot(t, T[2], color=COLORS['blue'], linewidth=2, label='T_z')
ax3.set_xlabel('Parameter t', fontsize=12)
ax3.set_ylabel('Component', fontsize=12)
ax3.set_title('Tangent Vector Components', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Plot 4: N vector components over time
ax4 = plt.subplot(2, 3, 4)
ax4.plot(t, N[0], color=COLORS['red'], linewidth=2, label='N_x')
ax4.plot(t, N[1], color=COLORS['green'], linewidth=2, label='N_y')
ax4.plot(t, N[2], color=COLORS['blue'], linewidth=2, label='N_z')
ax4.set_xlabel('Parameter t', fontsize=12)
ax4.set_ylabel('Component', fontsize=12)
ax4.set_title('Normal Vector Components', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# Plot 5: B vector components over time
ax5 = plt.subplot(2, 3, 5)
ax5.plot(t, B[0], color=COLORS['red'], linewidth=2, label='B_x')
ax5.plot(t, B[1], color=COLORS['green'], linewidth=2, label='B_y')
ax5.plot(t, B[2], color=COLORS['blue'], linewidth=2, label='B_z')
ax5.set_xlabel('Parameter t', fontsize=12)
ax5.set_ylabel('Component', fontsize=12)
ax5.set_title('Binormal Vector Components', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

# Plot 6: Frenet-Serret formulas verification
# dT/ds = κN, dN/ds = -κT + τB, dB/ds = -τN
ax6 = plt.subplot(2, 3, 6)

# Compute approximate derivatives
dt = t[1] - t[0]
dT_dt = np.gradient(T, dt, axis=1)
dN_dt = np.gradient(N, dt, axis=1)
dB_dt = np.gradient(B, dt, axis=1)

# Speed for arc length parameterization
speed = np.linalg.norm(r_prime, axis=0)

# Check: dT/ds should be proportional to N
dT_ds = dT_dt / speed
dT_ds_norm = np.linalg.norm(dT_ds, axis=0)

# Check parallelness with N
dot_product = np.sum(dT_ds * N, axis=0)

ax6.plot(t, dT_ds_norm, color=COLORS['red'], linewidth=2,
        label='|dT/ds| (should equal κ)')
ax6.axhline(y=a/(a**2 + b**2), color=COLORS['blue'], linestyle='--',
           linewidth=2, label=f'κ = {a/(a**2+b**2):.4f}')
ax6.set_xlabel('Parameter t', fontsize=12)
ax6.set_ylabel('Magnitude', fontsize=12)
ax6.set_title('Frenet-Serret Formula: dT/ds = κN', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Frenet-Serret Formulas:")
print("="*60)
print("  dT/ds = κN")
print("  dN/ds = -κT + τB")
print("  dB/ds = -τN")
print()
print("where s = arc length, κ = curvature, τ = torsion")
print()
print("These describe how the frame rotates as you move along the curve!")
print()
print("✓ In GR: Parallel transport generalizes this to curved spacetime!")
print("✓ Try: Use different curves (circle, figure-8, etc.)")
