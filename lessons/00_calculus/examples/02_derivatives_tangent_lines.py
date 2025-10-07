#!/usr/bin/env python3
"""
Example: Derivatives and Tangent Lines
Demonstrates the derivative as the slope of the tangent line
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

def f(x):
    """Function: f(x) = 0.5x² - 2x + 3"""
    return 0.5 * x**2 - 2*x + 3

def f_prime(x):
    """Derivative: f'(x) = x - 2"""
    return x - 2

def secant_line(x, x0, h):
    """
    Secant line through points (x0, f(x0)) and (x0+h, f(x0+h))
    Slope: [f(x0+h) - f(x0)] / h
    """
    slope = (f(x0 + h) - f(x0)) / h
    return f(x0) + slope * (x - x0)

def tangent_line(x, x0):
    """
    Tangent line at x0
    Slope: f'(x0)
    """
    slope = f_prime(x0)
    return f(x0) + slope * (x - x0)

# Setup
x = np.linspace(-1, 5, 200)
x0 = 2.0  # Point where we compute derivative

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))

# Top row: Show secant lines approaching tangent
h_values = [2.0, 1.0, 0.5, 0.1]
for i, h in enumerate(h_values, 1):
    ax = plt.subplot(2, 4, i)

    # Plot function
    ax.plot(x, f(x), color=COLORS['blue'], linewidth=2, label='f(x)')

    # Plot secant line
    x_secant = np.linspace(x0 - 0.5, x0 + h + 0.5, 100)
    y_secant = secant_line(x_secant, x0, h)
    ax.plot(x_secant, y_secant, color=COLORS['red'], linewidth=2,
            linestyle='--', label=f'Secant (h={h})')

    # Mark the two points
    ax.plot(x0, f(x0), 'o', color=COLORS['orange'], markersize=8)
    ax.plot(x0 + h, f(x0 + h), 'o', color=COLORS['orange'], markersize=8)

    # Compute slope
    slope = (f(x0 + h) - f(x0)) / h

    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('f(x)', fontsize=10)
    ax.set_title(f'h = {h}\nSlope ≈ {slope:.3f}', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.legend(fontsize=8)

# Bottom left: Tangent line (limit as h→0)
ax5 = plt.subplot(2, 4, 5)
ax5.plot(x, f(x), color=COLORS['blue'], linewidth=2.5, label='f(x) = 0.5x² - 2x + 3')
x_tangent = np.linspace(x0 - 2, x0 + 2, 100)
y_tangent = tangent_line(x_tangent, x0)
ax5.plot(x_tangent, y_tangent, color=COLORS['red'], linewidth=2.5,
         label=f"Tangent at x={x0}")
ax5.plot(x0, f(x0), 'o', color=COLORS['orange'], markersize=10)
slope_exact = f_prime(x0)
ax5.set_xlim(-1, 5)
ax5.set_ylim(-1, 8)
ax5.grid(True, alpha=0.3)
ax5.set_xlabel('x', fontsize=11)
ax5.set_ylabel('f(x)', fontsize=11)
ax5.set_title(f'Tangent Line (h→0)\nSlope = f\'({x0}) = {slope_exact:.3f}',
              fontsize=12, fontweight='bold')
ax5.axhline(y=0, color='k', linewidth=0.5)
ax5.legend(fontsize=9)

# Bottom middle: Show multiple tangent lines
ax6 = plt.subplot(2, 4, 6)
ax6.plot(x, f(x), color=COLORS['blue'], linewidth=2.5, label='f(x)')
x_points = [0, 1, 2, 3, 4]
colors = [COLORS['red'], COLORS['orange'], COLORS['green'], COLORS['purple'], COLORS['cyan']]
for x_pt, color in zip(x_points, colors):
    x_tang = np.linspace(x_pt - 1, x_pt + 1, 50)
    y_tang = tangent_line(x_tang, x_pt)
    ax6.plot(x_tang, y_tang, color=color, linewidth=1.5, alpha=0.7)
    ax6.plot(x_pt, f(x_pt), 'o', color=color, markersize=6)
ax6.set_xlim(-1, 5)
ax6.set_ylim(-1, 8)
ax6.grid(True, alpha=0.3)
ax6.set_xlabel('x', fontsize=11)
ax6.set_ylabel('f(x)', fontsize=11)
ax6.set_title('Multiple Tangent Lines\nSlope Changes with Position',
              fontsize=12, fontweight='bold')
ax6.axhline(y=0, color='k', linewidth=0.5)

# Bottom right: Derivative function
ax7 = plt.subplot(2, 4, (7, 8))
x_deriv = np.linspace(-1, 5, 200)
ax7_twin = ax7.twinx()

# Plot original function (blue, left axis)
ax7.plot(x_deriv, f(x_deriv), color=COLORS['blue'], linewidth=2.5, label='f(x)')
ax7.set_xlabel('x', fontsize=11)
ax7.set_ylabel('f(x)', fontsize=11, color=COLORS['blue'])
ax7.tick_params(axis='y', labelcolor=COLORS['blue'])

# Plot derivative (red, right axis)
ax7_twin.plot(x_deriv, f_prime(x_deriv), color=COLORS['red'], linewidth=2.5, label="f'(x)")
ax7_twin.set_ylabel("f'(x) = slope", fontsize=11, color=COLORS['red'])
ax7_twin.tick_params(axis='y', labelcolor=COLORS['red'])

# Mark where derivative is zero (minimum of f)
x_min = 2.0
ax7.plot(x_min, f(x_min), 'o', color=COLORS['orange'], markersize=10,
         label=f'Minimum at x={x_min}')
ax7_twin.axhline(y=0, color=COLORS['red'], linestyle='--', linewidth=1.5, alpha=0.7)

ax7.grid(True, alpha=0.3)
ax7.set_title("Function and Its Derivative\nf'(x) = 0 at minimum",
              fontsize=12, fontweight='bold')
ax7.legend(loc='upper left', fontsize=9)
ax7_twin.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

# Print numerical analysis
print("\n" + "="*60)
print("DERIVATIVE AS INSTANTANEOUS RATE OF CHANGE")
print("="*60)

print(f"\nFor f(x) = 0.5x² - 2x + 3")
print(f"Derivative: f'(x) = x - 2")
print("\nAnalyzing at x = 2:")
print(f"  f(2) = {f(2):.3f}")
print(f"  f'(2) = {f_prime(2):.3f}")
print("\nInterpretation:")
print(f"  • At x=2, the function has value {f(2):.3f}")
print(f"  • The tangent line at x=2 is horizontal (slope = 0)")
print(f"  • This is a critical point (minimum of the parabola)")

print("\n" + "-"*60)
print("Secant slopes approaching the derivative:")
print(f"{'h':<10} {'Secant Slope':<15} {'Error from f\'(2)':<15}")
print("-"*60)

for h in [1.0, 0.5, 0.1, 0.01, 0.001]:
    secant_slope = (f(x0 + h) - f(x0)) / h
    exact_slope = f_prime(x0)
    error = abs(secant_slope - exact_slope)
    print(f"{h:<10.4f} {secant_slope:<15.8f} {error:<15.8e}")

print("\nAs h → 0, secant slope → derivative!")

print("\n" + "="*60)
print("Key Insights:")
print("  • Derivative = slope of tangent line")
print("  • Tangent = limit of secant lines as h → 0")
print("  • f'(x) tells us how fast f(x) is changing at x")
print("  • f'(x) = 0 at local maxima and minima")
print("="*60)
