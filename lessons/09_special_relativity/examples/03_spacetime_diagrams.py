#!/usr/bin/env python3
"""
Example: Spacetime Diagrams and Causality
Demonstrates Minkowski diagrams and light cones
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("SPACETIME DIAGRAMS AND CAUSALITY")
print("="*60)

c = 1.0  # Natural units

# Create spacetime diagram
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Basic spacetime diagram
ax1 = axes[0, 0]

# Light cone
t = np.linspace(-5, 5, 100)
ax1.fill_between(t, 0, t, alpha=0.2, color=COLORS['yellow'], label='Future light cone')
ax1.fill_between(t, 0, -t, alpha=0.2, color=COLORS['orange'], label='Past light cone')
ax1.plot(t, t, color=COLORS['orange'], linewidth=3, label='Light ray (45°)')
ax1.plot(t, -t, color=COLORS['orange'], linewidth=3)

# Worldlines
t_particle = np.linspace(0, 5, 100)
x_slow = 0.3 * t_particle
x_fast = 0.8 * t_particle

ax1.plot(x_slow, t_particle, color=COLORS['blue'], linewidth=2, label='v=0.3c (timelike)')
ax1.plot(x_fast, t_particle, color=COLORS['green'], linewidth=2, label='v=0.8c (timelike)')

# Spacelike separation
ax1.plot([2, 4], [3, 3], 'r--', linewidth=2, label='Spacelike (impossible)')
ax1.plot(2, 3, 'ro', markersize=8)
ax1.plot(4, 3, 'ro', markersize=8)

ax1.set_xlabel('x (space)', fontsize=12)
ax1.set_ylabel('t (time)', fontsize=12)
ax1.set_title('Spacetime Diagram\nLight Cones & Causality', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-5, 5])
ax1.set_ylim([-1, 5])
ax1.axhline(0, color='k', linewidth=0.8)
ax1.axvline(0, color='k', linewidth=0.8)

# Plot 2: Different frames
ax2 = axes[0, 1]

# Rest frame axes
ax2.plot([0, 5], [0, 0], 'k-', linewidth=2, label="x axis (rest frame)")
ax2.plot([0, 0], [0, 5], 'k-', linewidth=2, label="t axis (rest frame)")

# Moving frame axes (v=0.6c)
v = 0.6 * c
gamma = 1/np.sqrt(1 - (v/c)**2)
tan_theta = v/c

x_axis_moving = np.array([0, 5])
t_axis_moving_x = tan_theta * np.array([0, 5])
t_axis_moving_t = np.array([0, 5])
x_axis_moving_t = tan_theta * np.array([0, 5])

ax2.plot(x_axis_moving, x_axis_moving_t, 'b--', linewidth=2, label="x' axis (v=0.6c)")
ax2.plot(t_axis_moving_x, t_axis_moving_t, 'r--', linewidth=2, label="t' axis (v=0.6c)")

# Light cone
t_lc = np.linspace(0, 5, 100)
ax2.plot(t_lc, t_lc, color=COLORS['yellow'], linewidth=3, alpha=0.7, label='Light ray')
ax2.plot(-t_lc, t_lc, color=COLORS['yellow'], linewidth=3, alpha=0.7)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('t', fontsize=12)
ax2.set_title('Different Reference Frames\n(Axes rotate but light cone stays 45°)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-2, 5])
ax2.set_ylim([0, 5])

# Plot 3: Causality regions
ax3 = axes[1, 0]

# Origin event
ax3.plot(0, 0, 'ko', markersize=12, label='Event (here, now)')

# Light cones
t_future = np.linspace(0, 5, 100)
ax3.fill_between(t_future, 0, t_future, alpha=0.3, color=COLORS['green'], label='Future (causally connected)')
ax3.fill_between(-t_future, 0, -t_future, alpha=0.3, color=COLORS['blue'], label='Past (causally connected)')
ax3.fill_betweenx([0, 5], 5, 10, alpha=0.3, color=COLORS['red'], label='Elsewhere (spacelike)')
ax3.fill_betweenx([0, 5], -10, -5, alpha=0.3, color=COLORS['red'])

ax3.plot(t_future, t_future, 'k-', linewidth=2)
ax3.plot(-t_future, t_future, 'k-', linewidth=2)
ax3.plot(t_future, -t_future, 'k-', linewidth=2)
ax3.plot(-t_future, -t_future, 'k-', linewidth=2)

# Sample events
ax3.plot(2, 4, 'go', markersize=8)
ax3.text(2.3, 4, 'Can influence', fontsize=10, color='green')

ax3.plot(1, -2, 'bo', markersize=8)
ax3.text(1.3, -2, 'Could have caused', fontsize=10, color='blue')

ax3.plot(6, 2, 'ro', markersize=8)
ax3.text(6.3, 2, 'No causal connection', fontsize=10, color='red')

ax3.set_xlabel('x (space)', fontsize=12)
ax3.set_ylabel('t (time)', fontsize=12)
ax3.set_title('Causal Structure of Spacetime', fontsize=13, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-10, 10])
ax3.set_ylim([-5, 5])
ax3.axhline(0, color='k', linewidth=0.8)
ax3.axvline(0, color='k', linewidth=0.8)

# Plot 4: Summary
ax4 = axes[1, 1]
summary = """
Spacetime Diagrams (Minkowski Diagrams)

Convention:
• Vertical axis: time t
• Horizontal axis: space x
• Units: c=1 (light at 45°)

Light Cone Structure:
• Future light cone: events that can be influenced
• Past light cone: events that could have caused
• Elsewhere: spacelike separated (no causal connection)

Intervals:
• Timelike: Δs² = -Δt² + Δx² < 0
  (can be connected by v < c particle)
• Lightlike (null): Δs² = 0
  (connected by light ray)
• Spacelike: Δs² > 0
  (cannot be causally connected)

Worldlines:
• Massive particle: timelike (within light cone)
• Photon: lightlike (on light cone)
• Tachyon: spacelike (IMPOSSIBLE - would violate causality)

Causality:
• No signal faster than light
• Effect cannot precede cause
• Light cone separates possible from impossible
• Protected by Lorentz invariance
"""

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
        fontsize=9, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))
ax4.axis('off')

plt.tight_layout()
plt.show()

print("\n✓ Causality preserved: No information faster than light!")
