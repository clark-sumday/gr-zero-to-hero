#!/usr/bin/env python3
"""
Example: Time Dilation and Length Contraction
Demonstrates relativistic effects on time and space measurements
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("="*60)
print("TIME DILATION & LENGTH CONTRACTION")
print("="*60)

c = 1.0  # Speed of light

def gamma(v, c=1.0):
    """Lorentz factor"""
    return 1 / np.sqrt(1 - (v/c)**2)

# Time dilation
print("\nTIME DILATION:")
print("Δt' = γ Δt₀  (moving clocks run slow)")
print("where Δt₀ is proper time (rest frame)")

# Example: Muon decay
tau_0 = 2.2e-6  # Muon lifetime in rest frame (seconds)
v_muon = 0.99 * c

gamma_muon = gamma(v_muon, c)
tau_lab = gamma_muon * tau_0

print(f"\nMuon decay:")
print(f"  Lifetime in rest frame: τ₀ = {tau_0*1e6:.1f} μs")
print(f"  Velocity: v = {v_muon/c:.2f}c")
print(f"  γ factor: {gamma_muon:.2f}")
print(f"  Lifetime in lab: τ = γτ₀ = {tau_lab*1e6:.1f} μs")
print(f"  → {gamma_muon:.1f}× longer!")

# Length contraction
print("\nLENGTH CONTRACTION:")
print("L = L₀/γ  (moving objects contract)")
print("where L₀ is proper length (rest frame)")

# Example: Spacecraft
L_0 = 100  # meters
v_ship = 0.9 * c

gamma_ship = gamma(v_ship, c)
L_contracted = L_0 / gamma_ship

print(f"\nSpacecraft:")
print(f"  Length at rest: L₀ = {L_0} m")
print(f"  Velocity: v = {v_ship/c:.1f}c")
print(f"  γ factor: {gamma_ship:.2f}")
print(f"  Length when moving: L = L₀/γ = {L_contracted:.1f} m")
print(f"  → Contracted to {L_contracted/L_0:.1%} of rest length!")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Time dilation vs velocity
ax1 = axes[0, 0]
v_range = np.linspace(0, 0.99*c, 100)
time_dilation_factor = gamma(v_range, c)

ax1.plot(v_range/c, time_dilation_factor, color=COLORS['blue'], linewidth=3)
ax1.axhline(1, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Velocity v/c', fontsize=12)
ax1.set_ylabel('Time dilation factor γ', fontsize=12)
ax1.set_title("Time Dilation: Δt' = γΔt₀\n(Moving clocks run slow)", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([1, 10])

# Mark muon example
ax1.plot(v_muon/c, gamma_muon, 'ro', markersize=10)
ax1.annotate(f'Muon\nv=0.99c\nγ={gamma_muon:.1f}',
            xy=(v_muon/c, gamma_muon), xytext=(0.7, 5),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=2))

# Plot 2: Length contraction vs velocity
ax2 = axes[0, 1]
length_contraction_factor = 1 / gamma(v_range, c)

ax2.plot(v_range/c, length_contraction_factor, color=COLORS['red'], linewidth=3)
ax2.axhline(1, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Velocity v/c', fontsize=12)
ax2.set_ylabel('Length contraction factor 1/γ', fontsize=12)
ax2.set_title('Length Contraction: L = L₀/γ\n(Moving objects shrink)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# Mark spacecraft example
ax2.plot(v_ship/c, 1/gamma_ship, 'go', markersize=10)
ax2.annotate(f'Spacecraft\nv=0.9c\nL/L₀={1/gamma_ship:.2f}',
            xy=(v_ship/c, 1/gamma_ship), xytext=(0.5, 0.7),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=2))

# Plot 3: Twin paradox visualization
ax3 = axes[1, 0]

# Traveling twin's worldline
t_travel = np.array([0, 5, 10])
x_travel = np.array([0, 4, 0])

ax3.plot([0, 10], [0, 10], color=COLORS['blue'], linewidth=3, label='Stay-at-home twin')
ax3.plot(x_travel, t_travel, color=COLORS['red'], linewidth=3, marker='o', markersize=8,
        label='Traveling twin')

# Light cones
t_range = np.linspace(0, 10, 100)
ax3.fill_between(t_range, 0, t_range, alpha=0.2, color=COLORS['yellow'], label='Light cone')
ax3.fill_between(t_range, 0, -t_range, alpha=0.2, color=COLORS['yellow'])

ax3.set_xlabel('x (light-years)', fontsize=12)
ax3.set_ylabel('t (years)', fontsize=12)
ax3.set_title('Twin Paradox Spacetime Diagram\n(Traveler ages less!)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-1, 5])
ax3.set_ylim([0, 10])

# Calculate ages
v_twin = 0.8 * c
gamma_twin = gamma(v_twin, c)
proper_time_travel = 10 / gamma_twin

ax3.text(2, 8, f'Stay-at-home: 10 years\nTraveler: {proper_time_travel:.1f} years',
        fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 4: Effects summary
ax4 = axes[1, 1]
summary = """
Relativistic Effects Summary

TIME DILATION:
Δt' = γ Δt₀
• Moving clocks tick slower
• Proper time τ is shortest (rest frame)
• GPS satellites: ~38 μs/day correction!

LENGTH CONTRACTION:
L = L₀/γ
• Only in direction of motion
• Proper length L₀ is longest (rest frame)
• Transverse dimensions unchanged

Twin Paradox:
• Twin travels at high speed, returns
• Traveling twin is younger
• Not symmetric! (traveler accelerates)
• Proper time τ = ∫√(-ds²) along path

Examples:
• Muons reach Earth's surface
  (lifetime extended by γ≈20)
• Particle accelerators
  (particles live longer)
• Time dilation at v=0.9c: γ≈2.3
  (age 1 year per 2.3 years at rest)
"""

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
        fontsize=9.5, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2))
ax4.axis('off')

plt.tight_layout()
plt.show()

print("\n✓ These are REAL effects, measured experimentally!")
