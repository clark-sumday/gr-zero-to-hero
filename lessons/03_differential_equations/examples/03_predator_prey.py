#!/usr/bin/env python3
"""
Example: Predator-Prey System (Lotka-Volterra)
Demonstrates coupled nonlinear ODEs with cyclic behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Lotka-Volterra equations:
# dx/dt = αx - βxy  (prey: growth - predation)
# dy/dt = δxy - γy  (predator: predation - death)

def lotka_volterra(state, t, alpha, beta, gamma, delta):
    """
    x = prey population
    y = predator population
    """
    x, y = state
    dxdt = alpha*x - beta*x*y       # Prey equation
    dydt = delta*x*y - gamma*y      # Predator equation
    return [dxdt, dydt]

# Parameters (classic values)
alpha = 1.0   # Prey birth rate
beta = 0.1    # Predation rate
gamma = 1.5   # Predator death rate
delta = 0.075 # Predator reproduction efficiency

# Initial conditions
x0 = 10.0  # Initial prey population
y0 = 5.0   # Initial predator population

# Time array
t = np.linspace(0, 50, 2000)

print("Lotka-Volterra Predator-Prey Model")
print("=" * 60)
print("Equations:")
print(f"  dx/dt = αx - βxy   (prey)")
print(f"  dy/dt = δxy - γy   (predator)")
print()
print("Parameters:")
print(f"  α = {alpha}  (prey birth rate)")
print(f"  β = {beta}  (predation rate)")
print(f"  γ = {gamma}  (predator death rate)")
print(f"  δ = {delta}  (predator efficiency)")
print()
print(f"Initial populations: prey = {x0}, predators = {y0}")
print()
print("Equilibrium point:")
x_eq = gamma / delta
y_eq = alpha / beta
print(f"  x* = γ/δ = {x_eq:.2f}")
print(f"  y* = α/β = {y_eq:.2f}")

# Solve the system
state0 = [x0, y0]
solution = odeint(lotka_volterra, state0, t, args=(alpha, beta, gamma, delta))
prey = solution[:, 0]
predators = solution[:, 1]

# Visualization
fig = plt.figure(figsize=(15, 10))

# Plot 1: Population vs time
ax1 = plt.subplot(2, 3, 1)
ax1.plot(t, prey, color=COLORS['blue'], linewidth=2, label='Prey (x)')
ax1.plot(t, predators, color=COLORS['red'], linewidth=2, label='Predators (y)')
ax1.axhline(y=x_eq, color=COLORS['blue'], linestyle='--', alpha=0.5, label=f'Prey equilibrium ({x_eq:.1f})')
ax1.axhline(y=y_eq, color=COLORS['red'], linestyle='--', alpha=0.5, label=f'Predator equilibrium ({y_eq:.1f})')
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Population', fontsize=12)
ax1.set_title('Population Dynamics', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Plot 2: Phase portrait
ax2 = plt.subplot(2, 3, 2)
ax2.plot(prey, predators, color=COLORS['purple'], linewidth=2, label='Trajectory')
ax2.scatter([x0], [y0], color=COLORS['green'], s=150, marker='o',
           edgecolor='black', linewidth=2, zorder=5, label='Start')
ax2.scatter([x_eq], [y_eq], color=COLORS['orange'], s=150, marker='*',
           edgecolor='black', linewidth=2, zorder=5, label='Equilibrium')

# Add direction arrows
n_arrows = 8
arrow_indices = np.linspace(0, len(prey)-1, n_arrows, dtype=int)
for i in arrow_indices[:-1]:
    ax2.annotate('', xy=(prey[i+50], predators[i+50]), xytext=(prey[i], predators[i]),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.5))

ax2.set_xlabel('Prey Population (x)', fontsize=12)
ax2.set_ylabel('Predator Population (y)', fontsize=12)
ax2.set_title('Phase Portrait (State Space)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Plot 3: Vector field in phase space
ax3 = plt.subplot(2, 3, 3)

# Create grid for vector field
x_range = np.linspace(0, 40, 20)
y_range = np.linspace(0, 20, 20)
X, Y = np.meshgrid(x_range, y_range)

# Compute derivatives at each point
dX = alpha*X - beta*X*Y
dY = delta*X*Y - gamma*Y

# Normalize for better visualization
magnitude = np.sqrt(dX**2 + dY**2)
dX_norm = dX / (magnitude + 1e-10)
dY_norm = dY / (magnitude + 1e-10)

ax3.quiver(X, Y, dX_norm, dY_norm, color=COLORS['gray'], alpha=0.5, width=0.003)
ax3.plot(prey, predators, color=COLORS['purple'], linewidth=2, label='Solution')
ax3.scatter([x_eq], [y_eq], color=COLORS['orange'], s=150, marker='*',
           edgecolor='black', linewidth=2, zorder=5, label='Equilibrium')

# Plot nullclines (where dx/dt = 0 or dy/dt = 0)
ax3.axvline(x=x_eq, color=COLORS['red'], linestyle='--', alpha=0.5, label='dy/dt = 0')
ax3.axhline(y=y_eq, color=COLORS['blue'], linestyle='--', alpha=0.5, label='dx/dt = 0')

ax3.set_xlabel('Prey Population (x)', fontsize=12)
ax3.set_ylabel('Predator Population (y)', fontsize=12)
ax3.set_title('Vector Field and Nullclines', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)
ax3.set_xlim(0, 40)
ax3.set_ylim(0, 20)

# Plot 4: Different initial conditions
ax4 = plt.subplot(2, 3, 4)

initial_conditions = [(10, 5), (15, 8), (20, 3), (8, 10)]
ic_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]

for (x0_ic, y0_ic), color in zip(initial_conditions, ic_colors):
    state0 = [x0_ic, y0_ic]
    sol = odeint(lotka_volterra, state0, t, args=(alpha, beta, gamma, delta))
    ax4.plot(sol[:, 0], sol[:, 1], color=color, linewidth=2, alpha=0.7)
    ax4.scatter([x0_ic], [y0_ic], color=color, s=100, zorder=5)

ax4.scatter([x_eq], [y_eq], color=COLORS['red'], s=200, marker='*',
           edgecolor='black', linewidth=2, zorder=6, label='Equilibrium')
ax4.set_xlabel('Prey Population (x)', fontsize=12)
ax4.set_ylabel('Predator Population (y)', fontsize=12)
ax4.set_title('Multiple Trajectories (Different ICs)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# Plot 5: Zoomed timeline showing phase relationship
ax5 = plt.subplot(2, 3, 5)
t_zoom = t[:500]
prey_zoom = prey[:500]
pred_zoom = predators[:500]

ax5.plot(t_zoom, prey_zoom, color=COLORS['blue'], linewidth=2, label='Prey')
ax5.plot(t_zoom, pred_zoom, color=COLORS['red'], linewidth=2, label='Predators')

# Mark peaks
from scipy.signal import find_peaks
prey_peaks, _ = find_peaks(prey_zoom, distance=50)
pred_peaks, _ = find_peaks(pred_zoom, distance=50)

ax5.scatter(t_zoom[prey_peaks], prey_zoom[prey_peaks],
           color=COLORS['blue'], s=100, marker='v', zorder=5)
ax5.scatter(t_zoom[pred_peaks], pred_zoom[pred_peaks],
           color=COLORS['red'], s=100, marker='v', zorder=5)

ax5.set_xlabel('Time', fontsize=12)
ax5.set_ylabel('Population', fontsize=12)
ax5.set_title('Phase Lag: Predator Peaks Follow Prey Peaks', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

# Plot 6: Rates of change
ax6 = plt.subplot(2, 3, 6)

# Compute rates
dx_dt = alpha*prey - beta*prey*predators
dy_dt = delta*prey*predators - gamma*predators

ax6.plot(t, dx_dt, color=COLORS['blue'], linewidth=2, label='dx/dt (prey rate)')
ax6.plot(t, dy_dt, color=COLORS['red'], linewidth=2, label='dy/dt (predator rate)')
ax6.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
ax6.set_xlabel('Time', fontsize=12)
ax6.set_ylabel('Rate of Change', fontsize=12)
ax6.set_title('Population Growth Rates', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Key Observations:")
print("="*60)
print("1. Oscillations: Populations cycle periodically")
print("2. Phase lag: Predator peaks follow prey peaks")
print("3. Equilibrium: Unstable center point (cycles around it)")
print("4. Conservation: Orbits are closed loops in phase space")
print()
print("Biological interpretation:")
print("  • More prey → predators thrive → prey decline")
print("  • Fewer prey → predators starve → prey recover")
print("  • The cycle repeats indefinitely!")
print()
print("✓ In GR: Coupled equations also describe gravitational waves!")
print("✓ Try: Change parameters to see different cycle periods")
