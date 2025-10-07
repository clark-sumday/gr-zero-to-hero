#!/usr/bin/env python3
"""
Example: Limits Visualization
Demonstrates how functions approach limit values and indeterminate forms
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Example 1: Simple continuous function
def f1(x):
    """f(x) = x² - 2x + 1"""
    return x**2 - 2*x + 1

# Example 2: Function with removable discontinuity
def f2(x):
    """f(x) = (x² - 4)/(x - 2)"""
    # Avoid division by zero
    result = np.zeros_like(x, dtype=float)
    mask = np.abs(x - 2) > 1e-10
    result[mask] = (x[mask]**2 - 4) / (x[mask] - 2)
    result[~mask] = np.nan  # Mark the discontinuity
    return result

# Approach a point from both sides
def approach_point(f, a, epsilon_values):
    """Show function values as we approach point a"""
    print(f"\nApproaching x = {a}:")
    print(f"{'x (from left)':<15} {'f(x)':<15} {'x (from right)':<15} {'f(x)':<15}")
    print("-" * 60)

    for eps in epsilon_values:
        left_x = a - eps
        right_x = a + eps
        left_y = f(np.array([left_x]))[0]
        right_y = f(np.array([right_x]))[0]
        print(f"{left_x:<15.8f} {left_y:<15.8f} {right_x:<15.8f} {right_y:<15.8f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Continuous function
x1 = np.linspace(-1, 3, 200)
y1 = f1(x1)
limit_point = 1.0

ax1.plot(x1, y1, color=COLORS['blue'], linewidth=2, label='f(x) = x² - 2x + 1')
ax1.axvline(x=limit_point, color=COLORS['red'], linestyle='--', alpha=0.5, label=f'x = {limit_point}')
ax1.plot(limit_point, f1(np.array([limit_point]))[0], 'o', color=COLORS['orange'],
         markersize=10, label=f'f({limit_point}) = {f1(np.array([limit_point]))[0]:.1f}')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('Continuous Function\nLimit = Function Value', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.axhline(y=0, color='k', linewidth=0.5)

# Right plot: Removable discontinuity
x2 = np.linspace(-1, 5, 400)
y2 = f2(x2)
disc_point = 2.0
limit_value = 4.0  # The actual limit

ax2.plot(x2, y2, color=COLORS['blue'], linewidth=2, label='f(x) = (x² - 4)/(x - 2)')
ax2.axvline(x=disc_point, color=COLORS['red'], linestyle='--', alpha=0.5, label=f'x = {disc_point}')
# Show the limit value (not the function value, which doesn't exist)
ax2.plot(disc_point, limit_value, 'o', color=COLORS['orange'], markersize=10,
         markerfacecolor='none', markeredgewidth=2, label=f'lim = {limit_value}')
ax2.plot(disc_point, limit_value, 'x', color=COLORS['red'], markersize=12,
         markeredgewidth=2, label='f(2) undefined')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('f(x)', fontsize=12)
ax2.set_title('Removable Discontinuity\nLimit Exists, But f(2) Undefined', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.set_ylim(-2, 8)

plt.tight_layout()
plt.show()

# Numerical demonstration
print("\n" + "="*60)
print("NUMERICAL LIMIT DEMONSTRATION")
print("="*60)

print("\nExample 1: f(x) = x² - 2x + 1, approaching x = 1")
epsilons = [0.1, 0.01, 0.001, 0.0001]
approach_point(f1, 1.0, epsilons)
print(f"\nConclusion: lim[x→1] f(x) = {f1(np.array([1.0]))[0]:.6f} = f(1)")

print("\n" + "-"*60)

print("\nExample 2: f(x) = (x² - 4)/(x - 2), approaching x = 2")
approach_point(f2, 2.0, epsilons)
print(f"\nConclusion: lim[x→2] f(x) = 4, but f(2) is undefined!")
print("This is a removable discontinuity - we can 'fill in' the hole.")

print("\n" + "="*60)
print("\nKey Insights:")
print("  • For continuous functions: lim[x→a] f(x) = f(a)")
print("  • Limits describe behavior NEAR a point, not AT the point")
print("  • Indeterminate forms (0/0) require algebraic manipulation")
print("  • Factor and cancel to find hidden limits!")
print("="*60)
