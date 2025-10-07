#!/usr/bin/env python3
"""
Example: Hamiltonian Mechanics - Phase Space Dynamics
Demonstrates Hamiltonian formulation and canonical equations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("HAMILTONIAN MECHANICS: PHASE SPACE FORMULATION")
print("="*60)

print("\nHamiltonian: H = T + V (Total energy)")
print("Phase space: (q, p) where p = ∂L/∂q̇ (momentum)")
print("\nHamilton's equations:")
print("  dq/dt = ∂H/∂p")
print("  dp/dt = -∂H/∂q")

# Example: Harmonic Oscillator
m = 1.0
k = 1.0
omega = np.sqrt(k/m)

print(f"\nHarmonic oscillator: m={m}, k={k}")
print(f"Lagrangian: L = ½m q̇² - ½k q²")
print(f"Momentum: p = ∂L/∂q̇ = m q̇")
print(f"Hamiltonian: H = p²/(2m) + ½k q²")

def hamiltonian_oscillator(state, t, m, k):
    """Hamilton's equations for oscillator"""
    q, p = state
    dqdt = p / m  # ∂H/∂p
    dpdt = -k * q  # -∂H/∂q
    return [dqdt, dpdt]

# Initial conditions
q0 = 1.0
p0 = 0.0
initial_state = [q0, p0]

t = np.linspace(0, 20, 400)
solution = odeint(hamiltonian_oscillator, initial_state, t, args=(m, k))

q_t = solution[:, 0]
p_t = solution[:, 1]

# Energy (Hamiltonian)
H_t = p_t**2 / (2*m) + 0.5 * k * q_t**2

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Time evolution
ax1 = axes[0, 0]
ax1.plot(t, q_t, color=COLORS['blue'], linewidth=2, label='Position q(t)')
ax1.plot(t, p_t, color=COLORS['orange'], linewidth=2, label='Momentum p(t)')
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Value', fontsize=12)
ax1.set_title('Hamiltonian Evolution', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Phase space trajectory
ax2 = axes[0, 1]
ax2.plot(q_t, p_t, color=COLORS['purple'], linewidth=2)
ax2.plot(q0, p0, 'go', markersize=12, label='Start')
ax2.set_xlabel('Position q', fontsize=12)
ax2.set_ylabel('Momentum p', fontsize=12)
ax2.set_title('Phase Space Trajectory\n(Ellipse for SHO)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Plot 3: Energy conservation
ax3 = axes[1, 0]
ax3.plot(t, H_t, color=COLORS['green'], linewidth=2)
ax3.set_xlabel('Time', fontsize=12)
ax3.set_ylabel('Hamiltonian H', fontsize=12)
ax3.set_title(f'Energy Conservation\nH = {H_t[0]:.4f} (constant)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Comparison summary
ax4 = axes[1, 1]
summary = """
Lagrangian vs Hamiltonian

Lagrangian L = T - V
• Configuration space (q, q̇)
• Euler-Lagrange: d/dt(∂L/∂q̇) - ∂L/∂q = 0
• Good for deriving equations

Hamiltonian H = T + V
• Phase space (q, p)
• Hamilton's equations:
  dq/dt = ∂H/∂p
  dp/dt = -∂H/∂q
• Canonical structure (symplectic)
• Good for conservation laws
• Foundation for quantum mechanics

Legendre Transform:
H(q,p) = pq̇ - L(q,q̇)
p = ∂L/∂q̇
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
        fontsize=10, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))
ax4.axis('off')

plt.tight_layout()
plt.show()

print("\n✓ In GR: ADM formalism uses Hamiltonian approach!")
