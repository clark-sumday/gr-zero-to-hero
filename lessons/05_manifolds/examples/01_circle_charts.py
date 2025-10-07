#!/usr/bin/env python3
"""
Example: Circle Charts and Atlas
Demonstrates how to cover a manifold with overlapping coordinate charts
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# The circle S¹ as a manifold
# Cannot be covered by a single coordinate chart!
# Need at least two overlapping charts

def circle_chart_U1(theta):
    """
    Chart U₁: Covers circle except point (-1, 0)
    Maps circle to interval (-π, π)
    Using stereographic projection from point (-1, 0)
    """
    x = np.cos(theta)
    y = np.sin(theta)
    # Stereographic projection coordinate
    phi = np.arctan2(y, x + 1) * 2
    return phi

def circle_chart_U2(theta):
    """
    Chart U₂: Covers circle except point (1, 0)
    Maps circle to interval (0, 2π)
    Using stereographic projection from point (1, 0)
    """
    x = np.cos(theta)
    y = np.sin(theta)
    # Alternative coordinate
    psi = np.arctan2(y, x - 1) * 2 + np.pi
    return psi

def inverse_chart_U1(phi):
    """Map from coordinate back to circle"""
    # phi/2 is angle from point (-1, 0)
    theta = phi / 2 + np.pi
    return theta

def inverse_chart_U2(psi):
    """Map from coordinate back to circle"""
    theta = psi / 2
    return theta

print("Circle as a Manifold: Charts and Atlas")
print("=" * 60)
print("The circle S¹ is a 1-dimensional manifold")
print()
print("Key insight: Cannot use a single coordinate to cover entire circle!")
print("  (Any single coordinate must have a 'gap' or discontinuity)")
print()
print("Solution: Use an ATLAS of overlapping charts")
print()
print("Chart U₁: Covers S¹ \\ {(-1, 0)}")
print("  Uses stereographic projection from (-1, 0)")
print("  Coordinate φ ∈ (-π, π)")
print()
print("Chart U₂: Covers S¹ \\ {(1, 0)}")
print("  Uses stereographic projection from (1, 0)")
print("  Coordinate ψ ∈ (0, 2π)")
print()
print("Overlap region: Both charts cover most of the circle")
print("Transition function: Relates φ and ψ in overlap region")

# Create circle
theta = np.linspace(0, 2*np.pi, 1000)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

# Define regions
# U1 covers all except point near (-1, 0)
theta_U1 = theta[np.abs(theta - np.pi) > 0.1]
x_U1 = np.cos(theta_U1)
y_U1 = np.sin(theta_U1)

# U2 covers all except point near (1, 0)
theta_U2 = theta[(theta < 2*np.pi - 0.1) & (theta > 0.1)]
x_U2 = np.cos(theta_U2)
y_U2 = np.sin(theta_U2)

# Overlap region
theta_overlap = theta[(theta > 0.1) & (theta < np.pi - 0.1) |
                     (theta > np.pi + 0.1) & (theta < 2*np.pi - 0.1)]
x_overlap = np.cos(theta_overlap)
y_overlap = np.sin(theta_overlap)

# Compute coordinates in each chart
phi_U1 = circle_chart_U1(theta_U1)
psi_U2 = circle_chart_U2(theta_U2)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: The circle with chart regions
ax1 = plt.subplot(2, 3, 1)
ax1.plot(x_circle, y_circle, color=COLORS['gray'], linewidth=1, alpha=0.3)

# Show U1 region
ax1.plot(x_U1, y_U1, color=COLORS['blue'], linewidth=4, label='Chart U₁', alpha=0.7)

# Mark excluded point for U1
ax1.scatter([-1], [0], color=COLORS['blue'], s=200, marker='x',
           linewidth=3, zorder=5, label='Excluded from U₁')

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Chart U₁ Domain\n(stereographic from (-1,0))', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_aspect('equal')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Plot 2: Chart U2
ax2 = plt.subplot(2, 3, 2)
ax2.plot(x_circle, y_circle, color=COLORS['gray'], linewidth=1, alpha=0.3)
ax2.plot(x_U2, y_U2, color=COLORS['orange'], linewidth=4, label='Chart U₂', alpha=0.7)
ax2.scatter([1], [0], color=COLORS['orange'], s=200, marker='x',
           linewidth=3, zorder=5, label='Excluded from U₂')

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Chart U₂ Domain\n(stereographic from (1,0))', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_aspect('equal')
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# Plot 3: Both charts showing overlap
ax3 = plt.subplot(2, 3, 3)
ax3.plot(x_circle, y_circle, color=COLORS['gray'], linewidth=1, alpha=0.3)
ax3.plot(x_U1, y_U1, color=COLORS['blue'], linewidth=3, label='Chart U₁', alpha=0.5)
ax3.plot(x_U2, y_U2, color=COLORS['orange'], linewidth=3, label='Chart U₂', alpha=0.5)
ax3.plot(x_overlap, y_overlap, color=COLORS['green'], linewidth=5,
        label='Overlap region', alpha=0.7)

ax3.scatter([-1], [0], color=COLORS['blue'], s=200, marker='x', linewidth=3, zorder=5)
ax3.scatter([1], [0], color=COLORS['orange'], s=200, marker='x', linewidth=3, zorder=5)

ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Atlas: Both Charts Cover S¹', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_aspect('equal')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)

# Plot 4: Coordinate on chart U1
ax4 = plt.subplot(2, 3, 4)
ax4.plot(theta_U1, phi_U1, color=COLORS['blue'], linewidth=2)
ax4.set_xlabel('θ (angle on circle)', fontsize=12)
ax4.set_ylabel('φ (coordinate in U₁)', fontsize=12)
ax4.set_title('Chart U₁: Circle → ℝ\nφ(θ)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linewidth=0.5)

# Plot 5: Coordinate on chart U2
ax5 = plt.subplot(2, 3, 5)
ax5.plot(theta_U2, psi_U2, color=COLORS['orange'], linewidth=2)
ax5.set_xlabel('θ (angle on circle)', fontsize=12)
ax5.set_ylabel('ψ (coordinate in U₂)', fontsize=12)
ax5.set_title('Chart U₂: Circle → ℝ\nψ(θ)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='k', linewidth=0.5)

# Plot 6: Transition function
ax6 = plt.subplot(2, 3, 6)

# In overlap region, compute both coordinates
theta_trans = theta_overlap
phi_trans = circle_chart_U1(theta_trans)
psi_trans = circle_chart_U2(theta_trans)

ax6.plot(phi_trans, psi_trans, color=COLORS['green'], linewidth=2)
ax6.set_xlabel('φ (coordinate in U₁)', fontsize=12)
ax6.set_ylabel('ψ (coordinate in U₂)', fontsize=12)
ax6.set_title('Transition Function ψ(φ)\n(in overlap region)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Key Concepts:")
print("="*60)
print("1. Manifold: Space that locally looks like ℝⁿ")
print("   (Circle locally looks like a line)")
print()
print("2. Chart: Local coordinate system")
print("   (φ, ψ) - Maps piece of manifold to ℝ")
print()
print("3. Atlas: Collection of charts covering entire manifold")
print("   (U₁, U₂) together cover all of S¹")
print()
print("4. Transition function: Relates coordinates in overlap")
print("   ψ = ψ(φ) in U₁ ∩ U₂")
print()
print("5. Compatibility: Transition functions must be smooth")
print("   (Differentiable, ensures manifold structure)")
print()
print("Why this matters:")
print("  • Many spaces cannot be described by single coordinates")
print("  • Need overlapping 'patches' with consistent transition maps")
print("  • This is how we do calculus on curved spaces!")
print()
print("✓ In GR: Spacetime is a 4D manifold!")
print("✓ Need multiple coordinate systems (charts)")
print("✓ Physics must be independent of chart choice!")
print()
print("Try: Modify to show 3+ charts covering the circle")
