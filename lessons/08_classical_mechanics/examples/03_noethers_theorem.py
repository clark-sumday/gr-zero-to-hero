#!/usr/bin/env python3
"""
Example: Noether's Theorem - Symmetries and Conservation Laws
Demonstrates the connection between symmetries and conserved quantities
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("NOETHER'S THEOREM: SYMMETRIES → CONSERVATION LAWS")
print("="*60)

print("\nNoether's Theorem:")
print("Every continuous symmetry of the action corresponds to a conserved quantity")

print("\nExamples:")
print("  Time translation symmetry → Energy conservation")
print("  Space translation symmetry → Momentum conservation")
print("  Rotation symmetry → Angular momentum conservation")

# Example: Free particle
m = 1.0
print(f"\nFree particle (mass m={m}):")
print("L = ½m ẋ²  (no potential, no external forces)")

print("\n1. TIME TRANSLATION SYMMETRY")
print("   L doesn't depend on t explicitly")
print("   → Energy E = ½m ẋ² is conserved")

print("\n2. SPACE TRANSLATION SYMMETRY")
print("   L doesn't depend on x explicitly")
print("   → Momentum p = m ẋ is conserved")

# Simulate
def free_particle(state, t, m):
    """Free particle equations"""
    x, v = state
    return [v, 0]  # No acceleration

x0, v0 = 0.0, 1.0
t = np.linspace(0, 10, 100)
sol = odeint(free_particle, [x0, v0], t, args=(m,))

E = 0.5 * m * sol[:, 1]**2
p = m * sol[:, 1]

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(t, sol[:, 0], color=COLORS['blue'], linewidth=2)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Position x')
axes[0].set_title('Free Particle Motion')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, E, color=COLORS['green'], linewidth=2)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Energy E')
axes[1].set_title('Energy Conservation\n(Time symmetry)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, p, color=COLORS['orange'], linewidth=2)
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Momentum p')
axes[2].set_title('Momentum Conservation\n(Space symmetry)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ In GR: Energy-momentum tensor from spacetime symmetries!")
