#!/usr/bin/env python3
"""
Example: Gravitational Wave Solutions
Demonstrates plane wave solutions and energy transport
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("GRAVITATIONAL WAVE SOLUTIONS")
print("=" * 60)
print("\nLinearized metric: g_μν = η_μν + h_μν")
print("where |h_μν| << 1 (weak field)")
print("\nPlane wave solution:")
print("  h_μν = A_μν exp(i k_α x^α)")
print("  where k² = 0 (null wavevector)")

# Wave parameters
frequency = 100  # Hz (LIGO range)
wavelength = 3e8 / frequency  # meters
k = 2 * np.pi / wavelength  # Wave number

print(f"\nExample: Binary black hole merger gravitational wave")
print(f"  Frequency: f = {frequency} Hz")
print(f"  Wavelength: λ = {wavelength:.0f} m = {wavelength/1000:.0f} km")
print(f"  Wave number: k = {k:.2e} m⁻¹")

# Strain amplitude (exaggerated for visualization)
h0 = 0.2  # Dimensionless (real GW: h ~ 10⁻²¹)
print(f"  Strain amplitude: h₀ = {h0} (exaggerated; real h ~ 10⁻²¹)")

# Spatial grid
x = np.linspace(-2*wavelength, 2*wavelength, 200)
t_snapshots = np.linspace(0, 2/frequency, 8)  # 2 periods

# Plus polarization: h_+ = h₀ cos(kx - ωt)
omega = 2 * np.pi * frequency

print("\n" + "=" * 60)
print("GRAVITATIONAL WAVE POLARIZATIONS")
print("=" * 60)
print("\nTwo independent polarizations:")
print("  • Plus (+): stretches along x, compresses along y")
print("  • Cross (×): stretches along 45° diagonals")
print("\nBoth are transverse and traceless (TT gauge)")

# Visualization 1: Wave propagation
fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))

for idx, t in enumerate(t_snapshots):
    row = idx // 4
    col = idx % 4
    ax = axes1[row, col]

    # Plus and cross polarization strains
    h_plus = h0 * np.cos(k*x - omega*t)
    h_cross = h0 * np.sin(k*x - omega*t)

    ax.plot(x/wavelength, h_plus, color=COLORS['blue'], linewidth=2, label='h₊')
    ax.plot(x/wavelength, h_cross, color=COLORS['orange'], linewidth=2,
            linestyle='--', label='h×')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('x / λ', fontsize=10)
    ax.set_ylabel('Strain h', fontsize=10)
    ax.set_title(f't = {t*frequency:.2f} / f', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2*h0, 1.2*h0)
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.suptitle('Gravitational Wave Propagation (Plane Wave)',
             fontsize=14, fontweight='bold', y=0.995)
plt.show()

# Visualization 2: Effect on test masses
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))

# Ring of test masses
n_masses = 12
theta_ring = np.linspace(0, 2*np.pi, n_masses, endpoint=False)
x0 = np.cos(theta_ring)
y0 = np.sin(theta_ring)

for idx, t in enumerate(t_snapshots):
    row = idx // 4
    col = idx % 4
    ax = axes2[row, col]

    # Plus polarization at this time (at x=0)
    h_plus_t = h0 * np.cos(-omega*t)

    # Deformation: x → x(1 + h₊/2), y → y(1 - h₊/2)
    x_def = x0 * (1 + h_plus_t/2)
    y_def = y0 * (1 - h_plus_t/2)

    # Plot original and deformed
    ax.plot(x0, y0, 'o-', color=COLORS['gray'], alpha=0.3,
            markersize=4, linewidth=1, label='Original')
    ax.plot(x_def, y_def, 'o-', color=COLORS['blue'],
            markersize=6, linewidth=2, label='Deformed')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f't = {t*frequency:.2f}/f\nh₊ = {h_plus_t:.2f}', fontsize=10)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.suptitle('Effect of Gravitational Wave on Ring of Test Masses',
             fontsize=14, fontweight='bold', y=0.995)
plt.show()

# Energy flux in gravitational waves
print("\n" + "=" * 60)
print("ENERGY TRANSPORT")
print("=" * 60)
print("\nEnergy flux (time-averaged):")
print("  F = (c³/16πG) <ḣ₊² + ḣ×²>")
print("\nFor plane wave h = h₀ cos(ωt):")
print("  <ḣ²> = ω² h₀² / 2")

# Energy flux calculation
c = 3e8  # m/s
G = 6.674e-11  # m³/kg/s²
h0_real = 1e-21  # Realistic LIGO strain

hdot_squared_avg = (omega**2 * h0_real**2) / 2
energy_flux = (c**3 / (16 * np.pi * G)) * 2 * hdot_squared_avg  # Factor 2 for both polarizations

print(f"\nFor LIGO detection (h₀ ~ {h0_real:.0e}, f = {frequency} Hz):")
print(f"  Energy flux: F = {energy_flux:.2e} W/m²")

# Luminosity of gravitational wave source
# For binary merger at distance d
distance_Mpc = 410  # Mpc (GW150914)
distance_m = distance_Mpc * 3.086e22  # meters

luminosity_GW = 4 * np.pi * distance_m**2 * energy_flux
luminosity_solar = 3.828e26  # W

print(f"\nAt distance d = {distance_Mpc} Mpc:")
print(f"  GW luminosity: L_GW = {luminosity_GW:.2e} W")
print(f"  In solar luminosities: L_GW = {luminosity_GW/luminosity_solar:.2e} L☉")
print(f"  (Roughly {luminosity_GW/(c**2*2e30):.1f} solar masses per second!)")

# Visualization 3: Energy and amplitude vs frequency
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

# Detector sensitivity curves
freq_range = np.logspace(0, 4, 1000)  # 1 Hz to 10 kHz

# LIGO sensitivity (approximate)
f_ligo = freq_range
h_ligo = 1e-23 * np.sqrt((f_ligo/150)**(-4) + 2*(1 + (f_ligo/150)**2))

# Energy flux vs frequency (for fixed strain)
energy_flux_freq = (c**3 / (16*np.pi*G)) * (2*np.pi*freq_range)**2 * (1e-21)**2

ax1 = axes3[0]
ax1.loglog(f_ligo, h_ligo, color=COLORS['blue'], linewidth=2, label='LIGO sensitivity')
ax1.axvline(x=100, color=COLORS['red'], linestyle='--', linewidth=1,
            alpha=0.5, label='Typical BH merger')
ax1.axhline(y=1e-21, color=COLORS['gray'], linestyle=':', alpha=0.5)
ax1.set_xlabel('Frequency (Hz)', fontsize=11)
ax1.set_ylabel('Strain sensitivity h', fontsize=11)
ax1.set_title('Gravitational Wave Detector Sensitivity', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=10)
ax1.set_xlim(10, 1e4)
ax1.set_ylim(1e-24, 1e-20)

ax2 = axes3[1]
ax2.loglog(freq_range, energy_flux_freq, color=COLORS['green'], linewidth=2)
ax2.set_xlabel('Frequency (Hz)', fontsize=11)
ax2.set_ylabel('Energy flux F (W/m²)', fontsize=11)
ax2.set_title('GW Energy Flux vs Frequency (h₀=10⁻²¹)',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.text(100, 1e-6, 'F ∝ f²\n(higher freq\n→ more energy)',
         fontsize=9, color=COLORS['green'], ha='center')

plt.tight_layout()
plt.show()

# Quadrupole formula for GW emission
print("\n" + "=" * 60)
print("QUADRUPOLE FORMULA")
print("=" * 60)
print("\nGravitational wave power from source:")
print("  P_GW = (G/5c⁵) <Q̈ⁱʲ Q̈_ij>")
print("\nwhere Q̈ⁱʲ is third time derivative of quadrupole moment")
print("\nFor circular binary (masses m₁, m₂, separation r, frequency Ω):")
print("  P_GW = (32/5) (G⁴/c⁵) (m₁m₂)² (m₁+m₂) / r⁵")

# Example: GW150914 parameters
m1_solar = 36  # Solar masses
m2_solar = 29
M_sun = 2e30  # kg

total_mass = (m1_solar + m2_solar) * M_sun
chirp_mass = ((m1_solar * m2_solar)**(3/5) / (m1_solar + m2_solar)**(1/5)) * M_sun

print(f"\nGW150914 (binary black hole merger):")
print(f"  m₁ = {m1_solar} M☉, m₂ = {m2_solar} M☉")
print(f"  Total mass: M = {m1_solar + m2_solar} M☉")
print(f"  Chirp mass: ℳ = {chirp_mass/M_sun:.1f} M☉")
print(f"\nAt merger: Power ~ 10⁵⁶ W (~ 3 M☉c²/s)")
print("  Brighter than all stars in observable universe combined!")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("• Gravitational waves are solutions to linearized Einstein equations")
print("• Two polarizations: + and ×")
print("• Transverse and travel at speed c")
print("• Quadrupole radiation (no monopole or dipole)")
print("• Energy flux ∝ frequency² × amplitude²")
print("• Detected by LIGO/Virgo: h ~ 10⁻²¹ (incredibly small!)")
print("=" * 60)
