#!/usr/bin/env python3
"""
Example: Equivalence Principle Demonstration
Shows that free fall cancels gravity, and acceleration mimics gravity
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Physical constants
g = 9.8  # m/s^2 (Earth's gravity)

# Time array
time = np.linspace(0, 2, 100)

# Scenario 1: Ball dropped in FREE-FALLING elevator
# Observer inside the elevator sees the ball floating (zero relative acceleration)
free_fall_ball_height = np.zeros_like(time)  # Ball stays at same height relative to floor

# Scenario 2: Ball dropped in ACCELERATING rocket in deep space
# Observer inside rocket sees ball fall with acceleration g
rocket_ball_height = -0.5 * g * time**2

# Scenario 3: Ball dropped in STATIONARY elevator on Earth
# Observer sees ball fall with acceleration g (same as rocket!)
stationary_ball_height = -0.5 * g * time**2

print("=" * 60)
print("EQUIVALENCE PRINCIPLE DEMONSTRATION")
print("=" * 60)
print("\nScenario 1: Free-falling elevator on Earth")
print(f"  Ball height at t=1.0s: {free_fall_ball_height[50]:.2f} m")
print("  → Ball appears WEIGHTLESS (floats)")
print("\nScenario 2: Accelerating rocket (a=9.8 m/s²) in deep space")
print(f"  Ball height at t=1.0s: {rocket_ball_height[50]:.2f} m")
print("  → Ball appears to FALL")
print("\nScenario 3: Stationary elevator on Earth")
print(f"  Ball height at t=1.0s: {stationary_ball_height[50]:.2f} m")
print("  → Ball FALLS (same as rocket!)")
print("\n" + "=" * 60)
print("KEY INSIGHT: Scenarios 2 and 3 are INDISTINGUISHABLE")
print("            (gravity ≡ acceleration)")
print("=" * 60)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Scenario 1: Free fall
ax1 = axes[0]
ax1.plot(time, free_fall_ball_height, color=COLORS['blue'], linewidth=2)
ax1.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Floor')
ax1.fill_between(time, -5, 0, alpha=0.1, color=COLORS['gray'])
ax1.set_xlabel('Time (s)', fontsize=11)
ax1.set_ylabel('Ball Height (m)', fontsize=11)
ax1.set_title('Free-Falling Elevator\n(Ball floats)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-5, 1)
ax1.text(1, 0.5, '⬆ Ball floats!', fontsize=10, color=COLORS['blue'])
ax1.legend()

# Scenario 2: Accelerating rocket
ax2 = axes[1]
ax2.plot(time, rocket_ball_height, color=COLORS['orange'], linewidth=2)
ax2.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Floor')
ax2.fill_between(time, rocket_ball_height.min()-1, 0, alpha=0.1, color=COLORS['gray'])
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Ball Height (m)', fontsize=11)
ax2.set_title('Accelerating Rocket (a=g)\n(Ball falls)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-20, 1)
ax2.text(1, -15, '⬇ Ball falls!', fontsize=10, color=COLORS['orange'])
ax2.legend()

# Scenario 3: Stationary on Earth
ax3 = axes[2]
ax3.plot(time, stationary_ball_height, color=COLORS['red'], linewidth=2)
ax3.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Floor')
ax3.fill_between(time, stationary_ball_height.min()-1, 0, alpha=0.1, color=COLORS['gray'])
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.set_ylabel('Ball Height (m)', fontsize=11)
ax3.set_title('Stationary on Earth\n(Ball falls)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-20, 1)
ax3.text(1, -15, '⬇ Ball falls!', fontsize=10, color=COLORS['red'])
ax3.legend()

plt.tight_layout()
plt.suptitle('Einstein\'s Equivalence Principle: Gravity ≡ Acceleration',
             fontsize=14, fontweight='bold', y=1.02)
plt.show()

print("\nConclusion:")
print("  • Free fall ELIMINATES gravity locally")
print("  • Acceleration CREATES artificial gravity")
print("  • No local experiment can distinguish gravity from acceleration!")
print("  • This leads to: GRAVITY = CURVED SPACETIME")
