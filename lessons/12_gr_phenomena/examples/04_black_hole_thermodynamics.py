#!/usr/bin/env python3
"""
Example: Black Hole Thermodynamics
Demonstrates Hawking radiation, black hole temperature, and entropy
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("BLACK HOLE THERMODYNAMICS")
print("=" * 60)
print("\nBlack holes are thermodynamic objects!")
print("They have temperature, entropy, and evaporate")

# Physical constants
c = 2.998e8      # m/s
G = 6.674e-11    # m^3 kg^-1 s^-2
hbar = 1.055e-34  # J·s
k_B = 1.381e-23  # J/K
M_sun = 1.989e30  # kg

# Schwarzschild radius
def schwarzschild_radius(M):
    """Schwarzschild radius in meters"""
    return 2 * G * M / c**2

# Hawking temperature
# T_H = ℏc³/(8πGMk_B)
def hawking_temperature(M):
    """Hawking temperature in Kelvin"""
    return hbar * c**3 / (8 * np.pi * G * M * k_B)

# Bekenstein-Hawking entropy
# S_BH = (k_B c³/4ℏG) × A = (k_B c³/ℏG) × πr_s²
def bh_entropy(M):
    """Black hole entropy (dimensionless, in units of k_B)"""
    r_s = schwarzschild_radius(M)
    return (c**3 / (4 * hbar * G)) * np.pi * r_s**2

# Hawking luminosity (power radiated)
# L = ℏc⁶/(15360πG²M²)
def hawking_luminosity(M):
    """Hawking luminosity in Watts"""
    return hbar * c**6 / (15360 * np.pi * G**2 * M**2)

# Evaporation time
# t_evap = (5120πG²M³)/(ℏc⁴)
def evaporation_time(M):
    """Evaporation time in seconds"""
    return (5120 * np.pi * G**2 * M**3) / (hbar * c**4)

# Example black holes
print(f"\n" + "=" * 60)
print("BLACK HOLE EXAMPLES")
print("=" * 60)

bh_masses = {
    "Solar mass": 1 * M_sun,
    "Stellar (10 M☉)": 10 * M_sun,
    "Supermassive (4×10⁶ M☉, Sgr A*)": 4e6 * M_sun,
    "Primordial (Moon mass)": 7.35e22,  # kg
    "Microscopic (1 kg)": 1.0
}

for name, M in bh_masses.items():
    r_s = schwarzschild_radius(M)
    T_H = hawking_temperature(M)
    S_BH = bh_entropy(M)
    L_H = hawking_luminosity(M)
    t_ev = evaporation_time(M)

    print(f"\n{name}:")
    print(f"  Mass: M = {M:.2e} kg")
    print(f"  Schwarzschild radius: r_s = {r_s:.2e} m")
    print(f"  Hawking temperature: T_H = {T_H:.2e} K")
    print(f"  Entropy: S = {S_BH:.2e} k_B")
    print(f"  Luminosity: L = {L_H:.2e} W")

    # Convert evaporation time to years
    if t_ev < 3.156e7:  # Less than 1 year
        print(f"  Evaporation time: {t_ev:.2e} seconds")
    elif t_ev < 3.156e13:  # Less than 1 million years
        print(f"  Evaporation time: {t_ev/3.156e7:.2e} years")
    else:
        print(f"  Evaporation time: {t_ev/3.156e7:.2e} years")

# Mass range for plotting
M_range = np.logspace(0, 40, 1000)  # 1 kg to 10^40 kg (10^10 M☉)

T_H_range = hawking_temperature(M_range)
S_BH_range = bh_entropy(M_range)
L_H_range = hawking_luminosity(M_range)
t_ev_range = evaporation_time(M_range)

# CMB temperature for comparison
T_CMB = 2.725  # K

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Hawking temperature vs mass
ax1 = axes[0, 0]
ax1.loglog(M_range/M_sun, T_H_range, color=COLORS['blue'], linewidth=2)
ax1.axhline(y=T_CMB, color=COLORS['red'], linestyle='--', linewidth=2,
            alpha=0.7, label=f'CMB temperature ({T_CMB} K)')
ax1.axvline(x=1, color=COLORS['gray'], linestyle=':', alpha=0.5,
            label='1 M☉')
ax1.set_xlabel('Mass (M☉)', fontsize=11)
ax1.set_ylabel('Hawking temperature T_H (K)', fontsize=11)
ax1.set_title('Black Hole Temperature', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=9)
ax1.text(1e-5, 1e10, 'T ∝ 1/M\n(smaller → hotter)',
         fontsize=9, color=COLORS['blue'], ha='center')

# Mark where T_H = T_CMB (stops evaporating)
M_CMB = hbar * c**3 / (8 * np.pi * G * T_CMB * k_B)
ax1.plot([M_CMB/M_sun], [T_CMB], 'o', color=COLORS['orange'],
         markersize=10, label=f'T_H = T_CMB\n({M_CMB/M_sun:.2e} M☉)')

# Plot 2: Entropy vs mass
ax2 = axes[0, 1]
ax2.loglog(M_range/M_sun, S_BH_range, color=COLORS['green'], linewidth=2)
ax2.axvline(x=1, color=COLORS['gray'], linestyle=':', alpha=0.5)
ax2.set_xlabel('Mass (M☉)', fontsize=11)
ax2.set_ylabel('Entropy S (k_B)', fontsize=11)
ax2.set_title('Black Hole Entropy', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.text(1e5, 1e80, 'S ∝ M²\n(area law)',
         fontsize=9, color=COLORS['green'], ha='center')

# Plot 3: Hawking luminosity vs mass
ax3 = axes[1, 0]
ax3.loglog(M_range/M_sun, L_H_range, color=COLORS['purple'], linewidth=2)
ax3.axvline(x=1, color=COLORS['gray'], linestyle=':', alpha=0.5)

# Solar luminosity for comparison
L_sun = 3.828e26  # W
ax3.axhline(y=L_sun, color=COLORS['orange'], linestyle='--', linewidth=2,
            alpha=0.7, label='Solar luminosity')

ax3.set_xlabel('Mass (M☉)', fontsize=11)
ax3.set_ylabel('Hawking luminosity L_H (W)', fontsize=11)
ax3.set_title('Black Hole Luminosity (Hawking Radiation)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')
ax3.legend(fontsize=9)
ax3.text(1e-10, 1e30, 'L ∝ 1/M²\n(smaller → brighter)',
         fontsize=9, color=COLORS['purple'], ha='center')

# Plot 4: Evaporation time vs mass
ax4 = axes[1, 1]
t_ev_years = t_ev_range / (3.156e7)  # Convert to years
age_universe = 13.8e9  # years

ax4.loglog(M_range/M_sun, t_ev_years, color=COLORS['cyan'], linewidth=2)
ax4.axvline(x=1, color=COLORS['gray'], linestyle=':', alpha=0.5, label='1 M☉')
ax4.axhline(y=age_universe, color=COLORS['red'], linestyle='--', linewidth=2,
            alpha=0.7, label=f'Age of universe ({age_universe:.1e} yr)')
ax4.set_xlabel('Mass (M☉)', fontsize=11)
ax4.set_ylabel('Evaporation time (years)', fontsize=11)
ax4.set_title('Black Hole Evaporation Time', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, which='both')
ax4.legend(fontsize=9)
ax4.text(1e10, 1e30, 't ∝ M³\n(larger → longer)',
         fontsize=9, color=COLORS['cyan'], ha='center')

plt.tight_layout()
plt.show()

# Four Laws of Black Hole Thermodynamics
print(f"\n" + "=" * 60)
print("FOUR LAWS OF BLACK HOLE THERMODYNAMICS")
print("=" * 60)

print("\n0th Law: Surface gravity κ is constant on event horizon")
print("  (Like temperature being uniform in thermal equilibrium)")

print("\n1st Law: dM = (κ/8πG) dA + Ω dJ + Φ dQ")
print("  Energy conservation (analogous to dE = TdS - PdV)")
print("  where A = horizon area, J = angular momentum, Q = charge")

print("\n2nd Law: Horizon area never decreases (Hawking's area theorem)")
print("  dA ≥ 0")
print("  (Analogous to entropy increase in thermodynamics)")

print("\n3rd Law: Cannot reduce surface gravity to zero")
print("  (Analogous to: cannot reach absolute zero temperature)")

# Information paradox
print(f"\n" + "=" * 60)
print("BLACK HOLE INFORMATION PARADOX")
print("=" * 60)
print("\nClassical GR: Information is lost in black holes")
print("  → Violates quantum mechanics (unitarity)")
print("\nHawking radiation: Thermal (no information)")
print("  → Information still lost!")
print("\nResolution (ongoing research):")
print("  • Information encoded in Hawking radiation (subtle correlations)")
print("  • Complementarity (observer-dependent)")
print("  • Holographic principle (boundary encoding)")
print("  • Firewall paradox")
print("\nStill an active area of research!")

# Entropy comparison
print(f"\n" + "=" * 60)
print("ENTROPY COMPARISON")
print("=" * 60)

# Solar mass black hole entropy
M_solar_bh = M_sun
S_solar_bh = bh_entropy(M_solar_bh)

# Compare to stellar entropy (rough estimate)
# Stellar entropy ~ k_B × (number of particles)
N_particles_sun = M_sun / (1.67e-27)  # Rough estimate (hydrogen mass)
S_stellar = N_particles_sun  # In units of k_B

print(f"\nSolar mass black hole:")
print(f"  S_BH = {S_solar_bh:.2e} k_B")
print(f"\nTypical star (same mass):")
print(f"  S_star ~ {S_stellar:.2e} k_B")
print(f"\nRatio: S_BH/S_star ~ {S_solar_bh/S_stellar:.2e}")
print(f"\n→ Black holes have ENORMOUS entropy!")
print(f"  (Maximum entropy for given mass)")

# Planck mass black hole
M_planck = np.sqrt(hbar * c / G)
T_planck = hawking_temperature(M_planck)
print(f"\n" + "=" * 60)
print("PLANCK SCALE")
print("=" * 60)
print(f"\nPlanck mass: M_P = {M_planck:.2e} kg")
print(f"  (Smallest possible black hole)")
print(f"Hawking temperature: T_H = {T_planck:.2e} K")
print(f"  (Quantum gravity regime)")

print(f"\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("• Black holes have temperature T_H ∝ 1/M")
print("• Black holes have entropy S_BH ∝ Area ∝ M²")
print("• Black holes evaporate via Hawking radiation")
print("• Smaller black holes are hotter and evaporate faster")
print("• Solar mass BH: T ~ 10⁻⁷ K (colder than CMB)")
print("• Evaporation time: t ∝ M³ (solar mass: 10⁶⁴ years)")
print("• Black holes are maximum entropy objects")
print("• Connects GR, quantum mechanics, and thermodynamics")
print("• Information paradox still unresolved!")
print("=" * 60)
