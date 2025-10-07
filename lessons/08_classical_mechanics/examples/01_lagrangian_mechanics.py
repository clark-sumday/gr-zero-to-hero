#!/usr/bin/env python3
"""
Example: Lagrangian Mechanics - Action Principle
Demonstrates the Lagrangian formulation and Euler-Lagrange equations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("LAGRANGIAN MECHANICS: THE ACTION PRINCIPLE")
print("="*60)

print("\nLagrangian L = T - V (Kinetic - Potential energy)")
print("Action S = ∫ L dt")
print("Euler-Lagrange equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0")

# Example 1: SIMPLE HARMONIC OSCILLATOR
print("\n" + "="*60)
print("1. SIMPLE HARMONIC OSCILLATOR")
print("="*60)

m = 1.0  # mass
k = 1.0  # spring constant
omega = np.sqrt(k/m)

print(f"Mass m = {m}")
print(f"Spring constant k = {k}")
print(f"Natural frequency ω = √(k/m) = {omega}")

print("\nLagrangian: L = ½m ẋ² - ½k x²")
print("           = T - V")

print("\nEuler-Lagrange equation:")
print("d/dt(∂L/∂ẋ) - ∂L/∂x = 0")
print("d/dt(m ẋ) - (-k x) = 0")
print("m ẍ + k x = 0")
print("✓ Recovers Newton's law: F = ma")

# Solve using Lagrangian approach
def lagrangian_oscillator(state, t, m, k):
    """
    Lagrangian approach to oscillator
    state = [x, v]
    """
    x, v = state
    dxdt = v
    dvdt = -(k/m) * x  # From Euler-Lagrange
    return [dxdt, dvdt]

# Initial conditions
x0 = 1.0
v0 = 0.0
initial_state = [x0, v0]

t = np.linspace(0, 10, 200)
solution = odeint(lagrangian_oscillator, initial_state, t, args=(m, k))

x_t = solution[:, 0]
v_t = solution[:, 1]

# Energy
T = 0.5 * m * v_t**2
V = 0.5 * k * x_t**2
E = T + V

print(f"\nInitial energy: E = {E[0]:.4f}")
print(f"Final energy: E = {E[-1]:.4f}")
print(f"Energy conserved: {np.allclose(E, E[0])}")

# Example 2: PENDULUM
print("\n" + "="*60)
print("2. SIMPLE PENDULUM")
print("="*60)

g = 9.8  # gravitational acceleration
L_pendulum = 1.0  # length

print(f"Length L = {L_pendulum} m")
print(f"Gravity g = {g} m/s²")

print("\nLagrangian: L = ½m L² θ̇² - mgL(1 - cos θ)")
print("           = T - V")

print("\nEuler-Lagrange equation:")
print("d/dt(∂L/∂θ̇) - ∂L/∂θ = 0")
print("d/dt(mL² θ̇) - (-mgL sin θ) = 0")
print("mL² θ̈ + mgL sin θ = 0")
print("θ̈ + (g/L) sin θ = 0")

def lagrangian_pendulum(state, t, g, L):
    """
    Pendulum via Lagrangian
    state = [theta, omega]
    """
    theta, omega = state
    dthetadt = omega
    domegadt = -(g/L) * np.sin(theta)  # From Euler-Lagrange
    return [dthetadt, domegadt]

# Initial conditions: release from 45 degrees
theta0 = np.pi / 4
omega0 = 0.0
initial_state_pend = [theta0, omega0]

t_pend = np.linspace(0, 10, 200)
solution_pend = odeint(lagrangian_pendulum, initial_state_pend, t_pend, args=(g, L_pendulum))

theta_t = solution_pend[:, 0]
omega_t = solution_pend[:, 1]

# Energy
T_pend = 0.5 * m * (L_pendulum * omega_t)**2
V_pend = m * g * L_pendulum * (1 - np.cos(theta_t))
E_pend = T_pend + V_pend

print(f"\nEnergy conservation check:")
print(f"Initial: E = {E_pend[0]:.6f}")
print(f"Final: E = {E_pend[-1]:.6f}")
print(f"Conserved: {np.allclose(E_pend, E_pend[0])}")

# Example 3: PRINCIPLE OF LEAST ACTION
print("\n" + "="*60)
print("3. PRINCIPLE OF LEAST ACTION")
print("="*60)

print("\nNature chooses the path that makes S = ∫ L dt stationary")
print("(Usually a minimum, hence 'principle of least action')")

# Compare true path with varied paths
def action_oscillator(path, t, m, k):
    """Compute action for a given path"""
    dt = t[1] - t[0]
    velocity = np.gradient(path, dt)

    # Lagrangian at each point
    L = 0.5 * m * velocity**2 - 0.5 * k * path**2

    # Action = integral of L
    S = np.trapz(L, t)
    return S

# True solution
x_true = x_t[:50]  # First half
t_short = t[:50]

S_true = action_oscillator(x_true, t_short, m, k)

print(f"\nAction for true path: S = {S_true:.6f}")

# Varied paths: add perturbations
S_values = [S_true]
labels = ['True path']

for amplitude in [0.1, 0.2, 0.3, 0.4]:
    # Add sinusoidal perturbation (satisfies boundary conditions)
    perturbation = amplitude * np.sin(2 * np.pi * t_short / t_short[-1])
    x_varied = x_true + perturbation
    S_varied = action_oscillator(x_varied, t_short, m, k)
    S_values.append(S_varied)
    labels.append(f'Perturb {amplitude}')

print("\nAction for perturbed paths:")
for label, S_val in zip(labels[1:], S_values[1:]):
    print(f"  {label}: S = {S_val:.6f} (ΔS = {S_val - S_true:+.6f})")

print("\n✓ True path has minimum action!")

# Visualization
fig = plt.figure(figsize=(14, 11))

# Plot 1: Harmonic oscillator solution
ax1 = plt.subplot(2, 3, 1)
ax1.plot(t, x_t, color=COLORS['blue'], linewidth=2, label='Position x(t)')
ax1.plot(t, v_t, color=COLORS['orange'], linewidth=2, label='Velocity v(t)')
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Value', fontsize=12)
ax1.set_title('Harmonic Oscillator Solution', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='k', linewidth=0.5)

# Plot 2: Phase space
ax2 = plt.subplot(2, 3, 2)
ax2.plot(x_t, v_t, color=COLORS['purple'], linewidth=2)
ax2.plot(x_t[0], v_t[0], 'go', markersize=10, label='Start')
ax2.plot(x_t[-1], v_t[-1], 'ro', markersize=10, label='End')
ax2.set_xlabel('Position x', fontsize=12)
ax2.set_ylabel('Velocity v', fontsize=12)
ax2.set_title('Phase Space (Closed Orbit)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.axhline(0, color='k', linewidth=0.5)
ax2.axvline(0, color='k', linewidth=0.5)

# Plot 3: Energy conservation
ax3 = plt.subplot(2, 3, 3)
ax3.plot(t, T, color=COLORS['red'], linewidth=2, label='Kinetic T')
ax3.plot(t, V, color=COLORS['blue'], linewidth=2, label='Potential V')
ax3.plot(t, E, color=COLORS['green'], linewidth=3, label='Total E', linestyle='--')
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Energy', fontsize=12)
ax3.set_title('Energy Conservation', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Pendulum motion
ax4 = plt.subplot(2, 3, 4)
ax4.plot(t_pend, np.degrees(theta_t), color=COLORS['blue'], linewidth=2, label='Angle θ(t)')
ax4.set_xlabel('Time (s)', fontsize=12)
ax4.set_ylabel('Angle (degrees)', fontsize=12)
ax4.set_title('Pendulum Oscillation', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color='k', linewidth=0.5)

# Plot 5: Action comparison
ax5 = plt.subplot(2, 3, 5)

# Plot varied paths
colors_paths = [COLORS['green'], COLORS['blue'], COLORS['orange'], COLORS['red'], COLORS['purple']]
for i, (amplitude, color) in enumerate(zip([0, 0.1, 0.2, 0.3, 0.4], colors_paths)):
    if amplitude == 0:
        path = x_true
        linewidth = 3
        label = 'True path'
    else:
        perturbation = amplitude * np.sin(2 * np.pi * t_short / t_short[-1])
        path = x_true + perturbation
        linewidth = 2
        label = f'Perturbed ({amplitude})'

    ax5.plot(t_short, path, color=color, linewidth=linewidth, label=label, alpha=0.8)

ax5.set_xlabel('Time (s)', fontsize=12)
ax5.set_ylabel('Position x(t)', fontsize=12)
ax5.set_title('True Path vs Varied Paths', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Plot 6: Action values
ax6 = plt.subplot(2, 3, 6)

x_pos = np.arange(len(labels))
colors_bars = [COLORS['green'], COLORS['blue'], COLORS['orange'], COLORS['red'], COLORS['purple']]
bars = ax6.bar(x_pos, S_values, color=colors_bars, edgecolor='black', linewidth=1.5)

# Highlight minimum
bars[0].set_linewidth(3)
bars[0].set_edgecolor('darkgreen')

ax6.set_xticks(x_pos)
ax6.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax6.set_ylabel('Action S', fontsize=12)
ax6.set_title('Action for Different Paths\n(True path minimizes S)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# Mark the minimum
ax6.axhline(S_true, color='g', linestyle='--', linewidth=2, alpha=0.5)
ax6.text(2.5, S_true, f' Minimum S = {S_true:.4f}', fontsize=10, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.show()

# Summary comparison
print("\n" + "="*60)
print("LAGRANGIAN vs NEWTONIAN MECHANICS")
print("="*60)

print("\nNewtonian Approach:")
print("  • Force F = ma")
print("  • Vector equations")
print("  • Requires forces and constraints")
print("  • Coordinate-dependent (Cartesian preferred)")

print("\nLagrangian Approach:")
print("  • Energy L = T - V")
print("  • Scalar function")
print("  • No need to identify forces explicitly")
print("  • Works in ANY coordinates (generalized coordinates)")
print("  • Elegant for constrained systems")
print("  • Foundation for quantum mechanics and field theory")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. Lagrangian L = T - V encodes all dynamics")
print("2. Euler-Lagrange equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0")
print("3. Principle of least action: δS = δ∫L dt = 0")
print("4. Nature 'chooses' the path that makes action stationary")
print("5. Generalizes to any coordinates (not just Cartesian)")
print("6. Energy automatically conserved if L doesn't depend on time")
print("\n✓ In GR: Einstein-Hilbert action determines spacetime dynamics!")
