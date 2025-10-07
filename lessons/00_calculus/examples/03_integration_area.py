#!/usr/bin/env python3
"""
Example: Integration as Area Under Curve
Demonstrates Riemann sums and the Fundamental Theorem of Calculus
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

def f(x):
    """Function to integrate: f(x) = x² - 2x + 3"""
    return x**2 - 2*x + 3

def F(x):
    """Antiderivative: F(x) = x³/3 - x² + 3x"""
    return x**3 / 3 - x**2 + 3*x

def riemann_sum(func, a, b, n, method='midpoint'):
    """
    Compute Riemann sum approximation
    method: 'left', 'right', 'midpoint'
    """
    dx = (b - a) / n
    total = 0

    if method == 'left':
        x_vals = np.linspace(a, b - dx, n)
    elif method == 'right':
        x_vals = np.linspace(a + dx, b, n)
    elif method == 'midpoint':
        x_vals = np.linspace(a + dx/2, b - dx/2, n)
    else:
        raise ValueError("method must be 'left', 'right', or 'midpoint'")

    total = np.sum(func(x_vals)) * dx
    return total, x_vals, dx

# Integration bounds
a, b = 0, 3

# Create figure
fig = plt.figure(figsize=(16, 10))

# Show Riemann sums with increasing rectangles
n_values = [4, 8, 16, 50]
for i, n in enumerate(n_values, 1):
    ax = plt.subplot(2, 4, i)

    # Plot function
    x_smooth = np.linspace(a - 0.5, b + 0.5, 200)
    ax.plot(x_smooth, f(x_smooth), color=COLORS['blue'], linewidth=2, label='f(x)')

    # Compute and plot Riemann sum (midpoint)
    area, x_vals, dx = riemann_sum(f, a, b, n, method='midpoint')

    # Draw rectangles
    for x_mid in x_vals:
        x_left = x_mid - dx/2
        height = f(x_mid)
        rect = plt.Rectangle((x_left, 0), dx, height,
                             facecolor=COLORS['orange'], edgecolor=COLORS['red'],
                             alpha=0.4, linewidth=1)
        ax.add_patch(rect)

    # Exact value
    exact = F(b) - F(a)
    error = abs(area - exact)

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 7)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('f(x)', fontsize=10)
    ax.set_title(f'n = {n} rectangles\nArea ≈ {area:.4f}\nError = {error:.4f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

# Bottom left: Compare left, right, midpoint
ax5 = plt.subplot(2, 4, 5)
n = 8
x_smooth = np.linspace(a - 0.5, b + 0.5, 200)
ax5.plot(x_smooth, f(x_smooth), color=COLORS['blue'], linewidth=2.5, label='f(x) = x² - 2x + 3')

# Only show one set of rectangles (midpoint) for clarity
area, x_vals, dx = riemann_sum(f, a, b, n, method='midpoint')
for x_mid in x_vals:
    x_left = x_mid - dx/2
    height = f(x_mid)
    rect = plt.Rectangle((x_left, 0), dx, height,
                         facecolor=COLORS['orange'], edgecolor=COLORS['red'],
                         alpha=0.4, linewidth=1.5)
    ax5.add_patch(rect)

ax5.axvline(x=a, color=COLORS['red'], linestyle='--', linewidth=2, alpha=0.7, label=f'x = {a}')
ax5.axvline(x=b, color=COLORS['red'], linestyle='--', linewidth=2, alpha=0.7, label=f'x = {b}')
ax5.set_xlim(-0.5, 3.5)
ax5.set_ylim(0, 7)
ax5.set_xlabel('x', fontsize=11)
ax5.set_ylabel('f(x)', fontsize=11)
ax5.set_title(f'Riemann Sum (Midpoint, n={n})\nApproximates Area Under Curve',
              fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)
ax5.axhline(y=0, color='k', linewidth=0.5)

# Bottom middle: Convergence plot
ax6 = plt.subplot(2, 4, 6)
n_range = np.arange(1, 101, 1)
errors = []
for n in n_range:
    area, _, _ = riemann_sum(f, a, b, n, method='midpoint')
    exact = F(b) - F(a)
    errors.append(abs(area - exact))

ax6.plot(n_range, errors, color=COLORS['red'], linewidth=2, label='Absolute Error')
ax6.set_xlabel('Number of rectangles (n)', fontsize=11)
ax6.set_ylabel('|Error|', fontsize=11)
ax6.set_title('Convergence to Exact Value\nError → 0 as n → ∞',
              fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=9)
ax6.set_yscale('log')

# Bottom right: Fundamental Theorem visualization
ax7 = plt.subplot(2, 4, (7, 8))
x_range = np.linspace(0, 3, 200)

# Plot accumulation function A(x) = ∫[0,x] f(t)dt
A_vals = [F(x) - F(0) for x in x_range]

ax7_twin = ax7.twinx()

# Original function (blue, left axis)
ax7.plot(x_range, f(x_range), color=COLORS['blue'], linewidth=2.5, label='f(x) = x² - 2x + 3')
ax7.set_xlabel('x', fontsize=11)
ax7.set_ylabel('f(x)', fontsize=11, color=COLORS['blue'])
ax7.tick_params(axis='y', labelcolor=COLORS['blue'])
ax7.grid(True, alpha=0.3)
ax7.axhline(y=0, color='k', linewidth=0.5)

# Accumulation function (green, right axis)
ax7_twin.plot(x_range, A_vals, color=COLORS['green'], linewidth=2.5,
              label='A(x) = ∫[0,x] f(t)dt')
ax7_twin.set_ylabel('A(x) = accumulated area', fontsize=11, color=COLORS['green'])
ax7_twin.tick_params(axis='y', labelcolor=COLORS['green'])

ax7.set_title('Fundamental Theorem of Calculus\ndA/dx = f(x)',
              fontsize=12, fontweight='bold')
ax7.legend(loc='upper left', fontsize=9)
ax7_twin.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

# Print numerical results
print("\n" + "="*60)
print("INTEGRATION AS AREA UNDER CURVE")
print("="*60)

print(f"\nFunction: f(x) = x² - 2x + 3")
print(f"Antiderivative: F(x) = x³/3 - x² + 3x")
print(f"Bounds: [{a}, {b}]")

print("\n" + "-"*60)
print("Riemann Sum Approximations:")
print(f"{'n':<10} {'Midpoint':<15} {'Error':<15}")
print("-"*60)

exact = F(b) - F(a)
for n in [5, 10, 20, 50, 100, 1000]:
    area, _, _ = riemann_sum(f, a, b, n, method='midpoint')
    error = abs(area - exact)
    print(f"{n:<10} {area:<15.8f} {error:<15.2e}")

print("\n" + "-"*60)
print(f"Exact value (using FTC):")
print(f"  ∫[{a},{b}] f(x)dx = F({b}) - F({a})")
print(f"                   = {F(b):.6f} - {F(a):.6f}")
print(f"                   = {exact:.6f}")

print("\n" + "="*60)
print("Key Insights:")
print("  • Riemann sum = sum of rectangle areas")
print("  • As n → ∞, Riemann sum → exact integral")
print("  • Fundamental Theorem: use antiderivative F to compute area!")
print("  • ∫[a,b] f(x)dx = F(b) - F(a)")
print("  • Integration and differentiation are inverse operations")
print("="*60)

print("\n✓ Try changing the function f(x) and bounds [a,b]!")
