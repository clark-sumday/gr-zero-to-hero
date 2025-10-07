#!/usr/bin/env python3
"""
Example: Lorentz Transformations - Coordinate Changes in Special Relativity
Demonstrates how spacetime coordinates transform between inertial frames
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("LORENTZ TRANSFORMATIONS: SPACETIME COORDINATE CHANGES")
print("="*60)

c = 1.0  # Speed of light (natural units)

def lorentz_boost(v, c=1.0):
    """
    Lorentz boost matrix in x-direction
    Λ^μ_ν for boost velocity v
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

def lorentz_factor(v, c=1.0):
    """Lorentz γ factor"""
    return 1 / np.sqrt(1 - (v/c)**2)

# Example transformations
v_values = [0.3*c, 0.6*c, 0.9*c]

print("\nLorentz boost matrix Λ for velocity v in x-direction:")
print("x'⁰ = γ(x⁰ - βx¹)  (t' = γ(t - vx/c²))")
print("x'¹ = γ(x¹ - βx⁰)  (x' = γ(x - vt))")

for v in v_values:
    gamma = lorentz_factor(v, c)
    print(f"\nv = {v/c:.1f}c:")
    print(f"  γ = {gamma:.4f}")
    print(f"  β = {v/c:.1f}")

# Event in rest frame
event_rest = np.array([2.0, 3.0, 0.0, 0.0])  # (t, x, y, z)
print(f"\nEvent in rest frame: (t,x,y,z) = {event_rest}")

# Transform to moving frames
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Event in different frames
ax1 = axes[0, 0]
frames = ['Rest'] + [f'v={v/c:.1f}c' for v in v_values]
t_vals = [event_rest[0]]
x_vals = [event_rest[1]]

for v in v_values:
    Lambda = lorentz_boost(v, c)
    event_moving = Lambda @ event_rest
    t_vals.append(event_moving[0])
    x_vals.append(event_moving[1])
    print(f"\nIn frame moving at v={v/c:.1f}c:")
    print(f"  Event: (t',x',y',z') = {event_moving}")

x_pos = np.arange(len(frames))
width = 0.35
ax1.bar(x_pos - width/2, t_vals, width, label='Time t', color=COLORS['blue'])
ax1.bar(x_pos + width/2, x_vals, width, label='Position x', color=COLORS['orange'])
ax1.set_xticks(x_pos)
ax1.set_xticklabels(frames)
ax1.set_ylabel('Coordinate value', fontsize=12)
ax1.set_title('Event Coordinates in Different Frames', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Lorentz factor vs velocity
ax2 = axes[0, 1]
v_range = np.linspace(0, 0.99*c, 100)
gamma_range = lorentz_factor(v_range, c)

ax2.plot(v_range/c, gamma_range, color=COLORS['purple'], linewidth=3)
ax2.axhline(1, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Velocity v/c', fontsize=12)
ax2.set_ylabel('Lorentz factor γ', fontsize=12)
ax2.set_title('γ = 1/√(1-v²/c²)\n(Diverges as v→c)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 10])

# Mark specific velocities
for v in v_values:
    gamma = lorentz_factor(v, c)
    ax2.plot(v/c, gamma, 'ro', markersize=8)
    ax2.text(v/c, gamma + 0.5, f'γ={gamma:.2f}', fontsize=9, ha='center')

# Plot 3: Spacetime diagram
ax3 = axes[1, 0]

# Draw light cone
t_range = np.linspace(-5, 5, 100)
ax3.plot(t_range, t_range, color=COLORS['yellow'], linewidth=3, label='Light rays', alpha=0.7)
ax3.plot(-t_range, t_range, color=COLORS['yellow'], linewidth=3, alpha=0.7)

# Draw worldlines in different frames
t_line = np.linspace(0, 4, 100)

# Rest frame (vertical worldline)
ax3.plot(np.zeros_like(t_line), t_line, color=COLORS['blue'], linewidth=3, label='Rest frame')

# Moving frames (tilted worldlines)
for i, v in enumerate([0.3*c, 0.6*c]):
    x_line = v * t_line
    ax3.plot(x_line, t_line, linewidth=2, label=f'v={v/c:.1f}c',
            color=[COLORS['green'], COLORS['red']][i])

ax3.set_xlabel('x (space)', fontsize=12)
ax3.set_ylabel('t (time)', fontsize=12)
ax3.set_title('Spacetime Diagram\nWorldlines for Different Observers', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-3, 3])
ax3.set_ylim([0, 4])
ax3.axhline(0, color='k', linewidth=0.5)
ax3.axvline(0, color='k', linewidth=0.5)

# Plot 4: Properties summary
ax4 = axes[1, 1]
summary = """
Lorentz Transformation Properties

Boost in x-direction (velocity v):
┌                          ┐
│  γ    -γβ   0    0      │
│ -γβ    γ    0    0      │
│  0     0    1    0      │
│  0     0    0    1      │
└                          ┘

where: β = v/c, γ = 1/√(1-β²)

Key Features:
• Linear transformation
• Preserves spacetime interval:
  s² = -t² + x² + y² + z²
• Reduces to Galilean at v<<c:
  t' ≈ t, x' ≈ x - vt
• Light speed invariant: c' = c
• Boosts don't commute (Thomas precession)

Composition:
Boost(v₁) ∘ Boost(v₂) = Boost(v₃) ∘ Rotation
where v₃ = (v₁+v₂)/(1+v₁v₂/c²)

Inverse:
Λ⁻¹(v) = Λ(-v)
"""

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
        fontsize=9, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))
ax4.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. Lorentz transformations mix space and time")
print("2. γ factor diverges as v → c (impossible to reach c)")
print("3. Preserves spacetime interval s²")
print("4. Speed of light c is same in all frames")
print("5. Simultaneity is relative!")
print("\n✓ Foundation of special relativity!")
