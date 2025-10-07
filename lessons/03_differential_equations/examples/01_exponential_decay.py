#!/usr/bin/env python3
"""
Example: Exponential Decay and Growth
Demonstrates solutions to dy/dt = ky (radioactive decay, population growth)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Differential equation: dy/dt = k*y
# Analytical solution: y(t) = y₀ * e^(kt)

def exponential_ode(y, t, k):
    """dy/dt = k*y"""
    return k * y

def analytical_solution(t, y0, k):
    """y(t) = y₀ * e^(kt)"""
    return y0 * np.exp(k * t)

# Initial condition
y0 = 100.0

# Different growth/decay rates
k_values = [0.5, 0.2, -0.2, -0.5]
k_labels = ['k = 0.5 (fast growth)', 'k = 0.2 (slow growth)',
            'k = -0.2 (slow decay)', 'k = -0.5 (fast decay)']
colors = [COLORS['red'], COLORS['orange'], COLORS['blue'], COLORS['purple']]

# Time array
t = np.linspace(0, 10, 200)

print("Exponential Growth/Decay: dy/dt = ky")
print("=" * 60)
print(f"Initial condition: y(0) = {y0}")
print(f"Analytical solution: y(t) = y₀ * e^(kt)")
print()
print("Different values of k:")
for i, (k, label) in enumerate(zip(k_values, k_labels)):
    half_life = -np.log(2) / k if k < 0 else np.log(2) / k
    print(f"  {label:25} => ", end="")
    if k > 0:
        print(f"Doubling time = {half_life:.2f}")
    else:
        print(f"Half-life = {half_life:.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: All solutions together
ax1 = axes[0, 0]
for i, (k, label, color) in enumerate(zip(k_values, k_labels, colors)):
    # Numerical solution
    y_numerical = odeint(exponential_ode, y0, t, args=(k,))

    # Analytical solution
    y_analytical = analytical_solution(t, y0, k)

    # Plot both (they should overlap perfectly)
    ax1.plot(t, y_numerical, '-', color=color, linewidth=2, label=label)
    ax1.plot(t, y_analytical, '--', color=color, linewidth=1, alpha=0.5)

ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel('y(t)', fontsize=12)
ax1.set_title('Solutions for Different k Values', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.axhline(y=y0, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Initial value')

# Plot 2: Log scale (exponentials become lines)
ax2 = axes[0, 1]
for i, (k, label, color) in enumerate(zip(k_values, k_labels, colors)):
    y_analytical = analytical_solution(t, y0, k)
    ax2.semilogy(t, y_analytical, '-', color=color, linewidth=2, label=label)

ax2.set_xlabel('Time t', fontsize=12)
ax2.set_ylabel('log(y(t))', fontsize=12)
ax2.set_title('Log Scale: Exponentials → Lines', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Plot 3: Slope field (phase portrait)
ax3 = axes[1, 0]
k = 0.3  # Choose one value
y_range = np.linspace(0, 200, 20)
t_range = np.linspace(0, 10, 20)
T, Y = np.meshgrid(t_range, y_range)

# Compute slopes: dy/dt = k*y
dY = k * Y
dT = np.ones_like(Y)

# Normalize for better visualization
magnitude = np.sqrt(dT**2 + dY**2)
dT_norm = dT / magnitude
dY_norm = dY / magnitude

ax3.quiver(T, Y, dT_norm, dY_norm, color=COLORS['gray'], alpha=0.5, width=0.003)

# Plot solution curve
y_solution = odeint(exponential_ode, y0, t, args=(k,))
ax3.plot(t, y_solution, color=COLORS['red'], linewidth=3, label=f'Solution (k={k})')
ax3.scatter([0], [y0], color=COLORS['blue'], s=100, zorder=5, label='Initial condition')

ax3.set_xlabel('Time t', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title(f'Slope Field for dy/dt = {k}y', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Plot 4: Rate of change vs y
ax4 = axes[1, 1]
y_vals = np.linspace(0, 200, 100)
for k, label, color in zip(k_values, k_labels, colors):
    dy_dt = k * y_vals
    ax4.plot(y_vals, dy_dt, color=color, linewidth=2, label=label)

ax4.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
ax4.axvline(x=0, color='k', linewidth=0.5)
ax4.set_xlabel('y', fontsize=12)
ax4.set_ylabel('dy/dt', fontsize=12)
ax4.set_title('Rate of Change vs Current Value', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Physical Examples:")
print("="*60)
print("k > 0: Population growth, compound interest")
print("k < 0: Radioactive decay, cooling (Newton's law)")
print()
print("Properties:")
print("  • Solution never reaches zero (approaches asymptotically)")
print("  • Rate of change proportional to current value")
print("  • Doubling time (k>0) or half-life (k<0) is constant")
print()
print("✓ In GR: Exponential expansion of the universe!")
print("✓ Try: Modify k to see different growth/decay rates")
