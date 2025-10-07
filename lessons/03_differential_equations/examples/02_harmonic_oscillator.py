#!/usr/bin/env python3
"""
Example: Harmonic Oscillator
Demonstrates solutions to d²x/dt² = -ω²x (mass on spring, pendulum)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Second-order ODE: d²x/dt² + ω²x = 0
# Rewrite as system:
#   dx/dt = v
#   dv/dt = -ω²x

def harmonic_oscillator(state, t, omega, damping=0):
    """
    State = [x, v]
    dx/dt = v
    dv/dt = -ω²x - 2γv (with damping γ)
    """
    x, v = state
    dxdt = v
    dvdt = -omega**2 * x - 2*damping*v
    return [dxdt, dvdt]

def analytical_solution(t, x0, v0, omega):
    """
    Undamped solution:
    x(t) = x₀cos(ωt) + (v₀/ω)sin(ωt)
    """
    return x0 * np.cos(omega*t) + (v0/omega) * np.sin(omega*t)

# Parameters
omega = 2.0  # Angular frequency
x0 = 1.0     # Initial position
v0 = 0.0     # Initial velocity

# Time array
t = np.linspace(0, 20, 1000)

print("Harmonic Oscillator: d²x/dt² + ω²x = 0")
print("=" * 60)
print(f"Angular frequency: ω = {omega}")
print(f"Period: T = 2π/ω = {2*np.pi/omega:.3f}")
print(f"Frequency: f = ω/2π = {omega/(2*np.pi):.3f} Hz")
print(f"\nInitial conditions: x(0) = {x0}, v(0) = {v0}")
print(f"Analytical solution: x(t) = {x0}cos({omega}t) + {v0/omega:.1f}sin({omega}t)")

# Solve with different damping
damping_values = [0, 0.1, 0.3, 1.0]
damping_labels = ['Undamped', 'Underdamped (γ=0.1)', 'Underdamped (γ=0.3)', 'Critically damped (γ=1.0)']
colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red']]

# Visualization
fig = plt.figure(figsize=(15, 10))

# Plot 1: Position vs time for different damping
ax1 = plt.subplot(2, 3, 1)
for damping, label, color in zip(damping_values, damping_labels, colors):
    state0 = [x0, v0]
    solution = odeint(harmonic_oscillator, state0, t, args=(omega, damping))
    x = solution[:, 0]
    ax1.plot(t, x, color=color, linewidth=2, label=label)

ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel('Position x(t)', fontsize=12)
ax1.set_title('Harmonic Oscillator with Damping', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.axhline(y=0, color='k', linewidth=0.5)

# Plot 2: Phase portrait (x vs v)
ax2 = plt.subplot(2, 3, 2)
for damping, label, color in zip(damping_values, damping_labels, colors):
    state0 = [x0, v0]
    solution = odeint(harmonic_oscillator, state0, t, args=(omega, damping))
    x = solution[:, 0]
    v = solution[:, 1]
    ax2.plot(x, v, color=color, linewidth=2, label=label)
    ax2.scatter([x0], [v0], color=color, s=100, zorder=5)

ax2.set_xlabel('Position x', fontsize=12)
ax2.set_ylabel('Velocity v', fontsize=12)
ax2.set_title('Phase Portrait', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_aspect('equal')

# Plot 3: Energy for undamped case
ax3 = plt.subplot(2, 3, 3)
state0 = [x0, v0]
solution = odeint(harmonic_oscillator, state0, t, args=(omega, 0))
x = solution[:, 0]
v = solution[:, 1]

# Energy components
kinetic = 0.5 * v**2
potential = 0.5 * omega**2 * x**2
total = kinetic + potential

ax3.plot(t, kinetic, color=COLORS['red'], linewidth=2, label='Kinetic (½v²)')
ax3.plot(t, potential, color=COLORS['blue'], linewidth=2, label='Potential (½ω²x²)')
ax3.plot(t, total, color=COLORS['black'], linewidth=2, linestyle='--', label='Total Energy')

ax3.set_xlabel('Time t', fontsize=12)
ax3.set_ylabel('Energy', fontsize=12)
ax3.set_title('Energy Conservation (Undamped)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Plot 4: Different initial conditions (undamped)
ax4 = plt.subplot(2, 3, 4)
initial_conditions = [(1, 0), (0.5, 0), (0, 2), (1, 1)]
ic_labels = ['x₀=1, v₀=0', 'x₀=0.5, v₀=0', 'x₀=0, v₀=2', 'x₀=1, v₀=1']
ic_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]

for (x0_ic, v0_ic), label, color in zip(initial_conditions, ic_labels, ic_colors):
    state0 = [x0_ic, v0_ic]
    solution = odeint(harmonic_oscillator, state0, t[:200], args=(omega, 0))
    x = solution[:, 0]
    ax4.plot(t[:200], x, color=color, linewidth=2, label=label)

ax4.set_xlabel('Time t', fontsize=12)
ax4.set_ylabel('Position x(t)', fontsize=12)
ax4.set_title('Different Initial Conditions', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9)
ax4.axhline(y=0, color='k', linewidth=0.5)

# Plot 5: Phase portrait with multiple trajectories
ax5 = plt.subplot(2, 3, 5)
for (x0_ic, v0_ic), color in zip(initial_conditions, ic_colors):
    state0 = [x0_ic, v0_ic]
    solution = odeint(harmonic_oscillator, state0, t, args=(omega, 0))
    x = solution[:, 0]
    v = solution[:, 1]
    ax5.plot(x, v, color=color, linewidth=2, alpha=0.7)
    ax5.scatter([x0_ic], [v0_ic], color=color, s=100, zorder=5)

# Add vector field
x_range = np.linspace(-2, 2, 15)
v_range = np.linspace(-4, 4, 15)
X, V = np.meshgrid(x_range, v_range)
dX = V
dV = -omega**2 * X
magnitude = np.sqrt(dX**2 + dV**2)
ax5.quiver(X, V, dX/magnitude, dV/magnitude, color=COLORS['gray'],
          alpha=0.3, width=0.003)

ax5.set_xlabel('Position x', fontsize=12)
ax5.set_ylabel('Velocity v', fontsize=12)
ax5.set_title('Phase Space (Undamped)', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='k', linewidth=0.5)
ax5.axvline(x=0, color='k', linewidth=0.5)
ax5.set_aspect('equal')

# Plot 6: Comparison with analytical solution
ax6 = plt.subplot(2, 3, 6)
state0 = [x0, v0]
solution = odeint(harmonic_oscillator, state0, t[:200], args=(omega, 0))
x_numerical = solution[:, 0]
x_analytical = analytical_solution(t[:200], x0, v0, omega)

ax6.plot(t[:200], x_numerical, color=COLORS['blue'], linewidth=2,
        label='Numerical (odeint)')
ax6.plot(t[:200], x_analytical, '--', color=COLORS['red'], linewidth=2,
        alpha=0.7, label='Analytical')
ax6.plot(t[:200], x_numerical - x_analytical, color=COLORS['green'],
        linewidth=1, label='Error (× 1000)')

ax6.set_xlabel('Time t', fontsize=12)
ax6.set_ylabel('x(t)', fontsize=12)
ax6.set_title('Numerical vs Analytical Solution', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Physical Interpretation:")
print("="*60)
print("Undamped: Perfect oscillation (total energy conserved)")
print("Underdamped: Oscillation with decay (friction)")
print("Critically damped: Fastest return to equilibrium (no overshoot)")
print("Overdamped: Slow return to equilibrium")
print()
print("Phase portrait: Closed loops = periodic motion")
print()
print("✓ In GR: Geodesic deviation equation is like coupled oscillators!")
print("✓ Try: Change ω to see different frequencies")
