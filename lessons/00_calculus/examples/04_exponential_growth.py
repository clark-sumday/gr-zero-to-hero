#!/usr/bin/env python3
"""
Example: Exponential Functions and Growth
Demonstrates e^x, natural logarithm, and exponential growth/decay
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Exponential growth model
def exponential_model(t, N0=1.0, k=1.0):
    """
    N(t) = N₀·e^(kt)
    N0: initial amount
    k: growth rate (k>0: growth, k<0: decay)
    """
    return N0 * np.exp(k * t)

# Create comprehensive figure
fig = plt.figure(figsize=(16, 12))

# Top left: e^x and its derivative
ax1 = plt.subplot(2, 3, 1)
x = np.linspace(-2, 3, 200)
y_exp = np.exp(x)

ax1.plot(x, y_exp, color=COLORS['blue'], linewidth=2.5, label='f(x) = e^x')
ax1.plot(x, y_exp, '--', color=COLORS['red'], linewidth=2, alpha=0.7, label="f'(x) = e^x")
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.set_title('The Special Property of e^x\nDerivative Equals Itself!',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Mark e^0 = 1
ax1.plot(0, 1, 'o', color=COLORS['orange'], markersize=10)
ax1.text(0.1, 1.2, 'e^0 = 1', fontsize=10)

# Top middle: ln(x) and its derivative
ax2 = plt.subplot(2, 3, 2)
x_pos = np.linspace(0.1, 5, 200)
y_ln = np.log(x_pos)
y_ln_deriv = 1 / x_pos

ax2.plot(x_pos, y_ln, color=COLORS['blue'], linewidth=2.5, label='f(x) = ln(x)')
ax2.plot(x_pos, y_ln_deriv, '--', color=COLORS['red'], linewidth=2, alpha=0.7, label="f'(x) = 1/x")
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y', fontsize=11)
ax2.set_title('Natural Logarithm\n(ln x)\' = 1/x',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# Mark ln(1) = 0 and ln(e) = 1
ax2.plot(1, 0, 'o', color=COLORS['orange'], markersize=10)
ax2.text(1.1, 0.2, 'ln(1) = 0', fontsize=10)
ax2.plot(np.e, 1, 'o', color=COLORS['green'], markersize=10)
ax2.text(np.e + 0.1, 1.2, f'ln(e) = 1', fontsize=10)

# Top right: e^x and ln(x) are inverses
ax3 = plt.subplot(2, 3, 3)
x_range = np.linspace(-2, 3, 200)
x_pos_range = np.linspace(0.1, 5, 200)

ax3.plot(x_range, np.exp(x_range), color=COLORS['blue'], linewidth=2.5, label='y = e^x')
ax3.plot(x_pos_range, np.log(x_pos_range), color=COLORS['red'], linewidth=2.5, label='y = ln(x)')
ax3.plot(x_range, x_range, '--', color=COLORS['gray'], linewidth=2, alpha=0.7, label='y = x')

ax3.set_xlabel('x', fontsize=11)
ax3.set_ylabel('y', fontsize=11)
ax3.set_title('Inverse Functions\ne^(ln(x)) = x, ln(e^x) = x',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_xlim(-2, 5)
ax3.set_ylim(-2, 5)

# Bottom left: Exponential growth (different rates)
ax4 = plt.subplot(2, 3, 4)
t = np.linspace(0, 3, 200)
growth_rates = [0.5, 1.0, 1.5, 2.0]

for k in growth_rates:
    N = exponential_model(t, N0=1.0, k=k)
    ax4.plot(t, N, linewidth=2.5, label=f'k = {k}')

ax4.set_xlabel('Time t', fontsize=11)
ax4.set_ylabel('N(t)', fontsize=11)
ax4.set_title('Exponential Growth\nN(t) = N₀·e^(kt), k > 0',
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.axhline(y=1, color='k', linewidth=0.5, linestyle='--', alpha=0.5)

# Bottom middle: Exponential decay
ax5 = plt.subplot(2, 3, 5)
t = np.linspace(0, 5, 200)
decay_rates = [-0.3, -0.5, -0.8, -1.2]

for k in decay_rates:
    N = exponential_model(t, N0=1.0, k=k)
    ax5.plot(t, N, linewidth=2.5, label=f'k = {k}')

ax5.set_xlabel('Time t', fontsize=11)
ax5.set_ylabel('N(t)', fontsize=11)
ax5.set_title('Exponential Decay\nN(t) = N₀·e^(kt), k < 0',
              fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)
ax5.axhline(y=0, color='k', linewidth=0.5)

# Bottom right: Half-life example (radioactive decay)
ax6 = plt.subplot(2, 3, 6)
# Carbon-14 half-life ≈ 5730 years (scaled for visualization)
half_life = 5730
k = -np.log(2) / half_life  # Decay constant
t_years = np.linspace(0, 20000, 200)
N_t = exponential_model(t_years, N0=100.0, k=k)

ax6.plot(t_years, N_t, color=COLORS['blue'], linewidth=2.5, label='C-14 remaining (%)')

# Mark half-lives
for i in range(1, 4):
    t_half = i * half_life
    N_half = 100 * (0.5)**i
    ax6.plot(t_half, N_half, 'o', color=COLORS['orange'], markersize=8)
    ax6.axvline(x=t_half, color=COLORS['red'], linestyle='--', alpha=0.3)
    ax6.text(t_half, N_half + 5, f't = {i}·t₁/₂', fontsize=9, ha='center')

ax6.set_xlabel('Time (years)', fontsize=11)
ax6.set_ylabel('Amount (%)', fontsize=11)
ax6.set_title('Radioactive Decay: Carbon-14\nHalf-life = 5730 years',
              fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=10)
ax6.axhline(y=0, color='k', linewidth=0.5)
ax6.axhline(y=50, color='k', linewidth=0.5, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Print educational information
print("\n" + "="*60)
print("EXPONENTIAL FUNCTIONS AND NATURAL LOGARITHM")
print("="*60)

print("\nEuler's number: e ≈", f"{np.e:.10f}")
print("\nWhy is e special?")
print("  • e^x is the ONLY function equal to its own derivative")
print("  • d/dx(e^x) = e^x")
print("  • This makes it fundamental in differential equations")

print("\n" + "-"*60)
print("Exponential Growth Model: N(t) = N₀·e^(kt)")
print("-"*60)

N0 = 100  # Initial population
k = 0.05  # Growth rate (5% per time unit)
t_values = [0, 1, 5, 10, 20]

print(f"\nInitial amount: N₀ = {N0}")
print(f"Growth rate: k = {k}")
print(f"\n{'Time t':<10} {'N(t)':<15} {'Growth from t=0':<20}")
print("-"*60)

for t in t_values:
    N_t = exponential_model(t, N0=N0, k=k)
    growth_factor = N_t / N0
    print(f"{t:<10} {N_t:<15.2f} {growth_factor:<20.3f}x")

print("\n" + "-"*60)
print("Exponential Decay: Half-Life")
print("-"*60)

print("\nFor radioactive decay, half-life t₁/₂ is when N(t) = N₀/2")
print("Formula: t₁/₂ = ln(2) / |k|")
print(f"\nExample: Carbon-14")
print(f"  Half-life: 5730 years")
print(f"  Decay constant: k = -ln(2)/5730 ≈ {-np.log(2)/5730:.6e}")

print("\nAfter n half-lives:")
print(f"  N(n·t₁/₂) = N₀·(1/2)^n")
print(f"  1 half-life:  N = 50% of N₀")
print(f"  2 half-lives: N = 25% of N₀")
print(f"  3 half-lives: N = 12.5% of N₀")

print("\n" + "="*60)
print("Applications in Physics:")
print("  • Radioactive decay (nuclear physics)")
print("  • Population growth")
print("  • Compound interest (finance)")
print("  • RC circuits (electronics)")
print("  • Cosmological expansion (de Sitter space in GR!)")
print("  • Schwarzschild metric: e^(2Φ/c²) term")
print("="*60)

print("\n" + "-"*60)
print("Derivative Rules:")
print("  d/dx(e^x) = e^x")
print("  d/dx(ln(x)) = 1/x")
print("  d/dx(e^(kx)) = k·e^(kx)  [chain rule]")
print("  d/dx(ln(f(x))) = f'(x)/f(x)  [chain rule]")
print("-"*60)

print("\n✓ Try modeling your own exponential process!")
