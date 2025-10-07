#!/usr/bin/env python3
"""
Example: Gravitational Wave Visualization
Shows how gravitational waves stretch and compress spacetime
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Gravitational wave parameters
frequency = 1.0  # Hz (for visualization)
amplitude = 0.3  # Dimensionless strain (exaggerated for visibility)
omega = 2 * np.pi * frequency

# Create a ring of test particles
n_particles = 16
theta = np.linspace(0, 2*np.pi, n_particles, endpoint=False)
x0 = np.cos(theta)
y0 = np.sin(theta)

print("=" * 60)
print("GRAVITATIONAL WAVE PROPAGATION")
print("=" * 60)
print(f"\nWave frequency: f = {frequency} Hz")
print(f"Wave amplitude: h = {amplitude} (exaggerated for visualization)")
print(f"\nGravitational wave polarizations:")
print("  • Plus polarization (+): stretches along x-axis, compresses along y-axis")
print("  • Cross polarization (×): stretches along 45° diagonals")
print("\nLIGO detected strain amplitude: h ~ 10⁻²¹")
print("  (Our visualization uses h ~ 0.3 for clarity!)")

# Time points
t = np.linspace(0, 2/frequency, 100)

# Static visualization at different time snapshots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

time_snapshots = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
for idx, t_frac in enumerate(time_snapshots):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    t_val = t_frac / frequency  # Time value
    phase = omega * t_val

    # Plus polarization: h_+ = h₀ cos(ωt)
    h_plus = amplitude * np.cos(phase)

    # Effect on particle positions
    # x → x(1 + h_+/2), y → y(1 - h_+/2)
    x = x0 * (1 + h_plus/2)
    y = y0 * (1 - h_plus/2)

    # Plot original circle
    ax.plot(x0, y0, 'o-', color=COLORS['gray'], alpha=0.3,
            markersize=4, linewidth=1, label='Original')

    # Plot deformed circle
    ax.plot(x, y, 'o-', color=COLORS['blue'],
            markersize=6, linewidth=2, label='Deformed')

    # Add axes
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title(f't = {t_frac:.3f}/f\nh₊ = {h_plus:.2f}', fontsize=10)

    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.suptitle('Gravitational Wave (+ Polarization) Effect on Ring of Particles',
             fontsize=14, fontweight='bold', y=0.995)
plt.show()

# Second figure: Wave strain over time
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Strain vs time
ax1 = axes2[0]
t_array = np.linspace(0, 3/frequency, 500)
h_plus_array = amplitude * np.cos(omega * t_array)
h_cross_array = amplitude * np.sin(omega * t_array)

ax1.plot(t_array * frequency, h_plus_array, color=COLORS['blue'],
         linewidth=2, label='h₊ (plus polarization)')
ax1.plot(t_array * frequency, h_cross_array, color=COLORS['orange'],
         linewidth=2, label='h× (cross polarization)', linestyle='--')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.set_xlabel('Time (cycles)', fontsize=11)
ax1.set_ylabel('Strain h', fontsize=11)
ax1.set_title('Gravitational Wave Strain', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Energy spectrum
ax2 = axes2[1]
frequencies = np.linspace(0.1, 5*frequency, 1000)
# Energy flux ∝ f² for GW
energy_flux = (frequencies / frequency)**2 * np.exp(-(frequencies - frequency)**2 / (0.1*frequency)**2)
energy_flux = energy_flux / energy_flux.max()

ax2.plot(frequencies, energy_flux, color=COLORS['green'], linewidth=2)
ax2.axvline(x=frequency, color=COLORS['red'], linestyle='--',
            linewidth=2, label=f'Peak at f = {frequency} Hz')
ax2.set_xlabel('Frequency (Hz)', fontsize=11)
ax2.set_ylabel('Relative Energy Flux', fontsize=11)
ax2.set_title('Gravitational Wave Energy Spectrum', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()

# Third figure: LIGO strain comparison
fig3, ax3 = plt.subplots(figsize=(12, 6))

# Simulated LIGO-like signal (GW150914 style)
t_ligo = np.linspace(-0.2, 0.2, 1000)
# Chirp: frequency increases, amplitude increases then drops
freq_chirp = 35 + 100 * (t_ligo + 0.2)**2 / 0.16  # Hz, increasing
amplitude_chirp = 1e-21 * (1 + 10*t_ligo) * np.exp(-30*(t_ligo-0.05)**2)
phase_chirp = np.cumsum(2*np.pi*freq_chirp) * (t_ligo[1]-t_ligo[0])
strain_chirp = amplitude_chirp * np.cos(phase_chirp)

ax3.plot(t_ligo * 1000, strain_chirp * 1e21, color=COLORS['purple'], linewidth=1.5)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.set_xlabel('Time (milliseconds)', fontsize=11)
ax3.set_ylabel('Strain h × 10²¹', fontsize=11)
ax3.set_title('LIGO-style Gravitational Wave Signal (Binary Black Hole Merger)',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.text(-150, 0.8, 'Inspiral\n(frequency ↑)', fontsize=10,
         color=COLORS['purple'], ha='center')
ax3.text(50, 0.8, 'Merger\n(peak amplitude)', fontsize=10,
         color=COLORS['purple'], ha='center')
ax3.text(150, 0.3, 'Ringdown\n(decay)', fontsize=10,
         color=COLORS['purple'], ha='center')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("• Gravitational waves are ripples in spacetime itself")
print("• They stretch space in one direction while compressing in perpendicular direction")
print("• Two independent polarizations: + and ×")
print("• LIGO sensitivity: h ~ 10⁻²¹ (ratio ΔL/L)")
print("• Binary mergers produce characteristic 'chirp' signals")
print("• Direct confirmation of Einstein's 1916 prediction!")
print("=" * 60)
