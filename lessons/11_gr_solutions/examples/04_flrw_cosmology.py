#!/usr/bin/env python3
"""
Example: FLRW Cosmology - Expanding Universe
Demonstrates Friedmann equations and cosmic expansion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("FLRW COSMOLOGY: THE EXPANDING UNIVERSE")
print("=" * 60)
print("\nFriedmann-Lemaître-Robertson-Walker metric:")
print("  ds² = -c²dt² + a(t)²[dr²/(1-kr²) + r²(dθ² + sin²θ dφ²)]")
print("\nwhere:")
print("  • a(t) = scale factor (describes expansion)")
print("  • k = curvature parameter (0: flat, +1: closed, -1: open)")

# Hubble constant today (km/s/Mpc)
H0_kmsMpc = 70  # Approximate value
# Convert to SI: 1 Mpc = 3.086e22 m
H0_SI = H0_kmsMpc * 1000 / (3.086e22)  # s^-1
print(f"\nHubble constant today: H₀ ≈ {H0_kmsMpc} km/s/Mpc")

# Critical density
G = 6.674e-11  # m^3 kg^-1 s^-2
c = 2.998e8    # m/s
rho_crit = 3 * H0_SI**2 / (8 * np.pi * G)
print(f"Critical density: ρ_c = {rho_crit:.2e} kg/m³")

# Current density parameters (approximate)
Omega_m0 = 0.3   # Matter (dark + baryonic)
Omega_r0 = 9e-5  # Radiation
Omega_Lambda0 = 0.7  # Dark energy (cosmological constant)
Omega_k0 = 1 - Omega_m0 - Omega_r0 - Omega_Lambda0  # Curvature

print(f"\nDensity parameters today:")
print(f"  Ω_m = {Omega_m0:.2f} (matter)")
print(f"  Ω_r = {Omega_r0:.2e} (radiation)")
print(f"  Ω_Λ = {Omega_Lambda0:.2f} (dark energy)")
print(f"  Ω_k = {Omega_k0:.3f} (curvature)")

def friedmann_equation(a, t, Omega_m0, Omega_r0, Omega_Lambda0):
    """
    Friedmann equation: H² = H₀²[Ω_m a⁻³ + Ω_r a⁻⁴ + Ω_Λ + Ω_k a⁻²]
    Returns da/dt
    """
    if a <= 0:
        return 0
    Omega_k0 = 1 - Omega_m0 - Omega_r0 - Omega_Lambda0
    H_squared = Omega_m0 * a**(-3) + Omega_r0 * a**(-4) + Omega_Lambda0 + Omega_k0 * a**(-2)
    if H_squared < 0:
        return 0
    H = np.sqrt(H_squared)
    return a * H  # da/dt = a * H

# Time array (in units of 1/H₀)
t = np.linspace(0.01, 3, 1000)

# Solve for different cosmologies
# Model 1: Matter-dominated (Einstein-de Sitter)
a1 = odeint(friedmann_equation, 0.01, t, args=(1.0, 0, 0))[:, 0]

# Model 2: Current best-fit (ΛCDM)
a2 = odeint(friedmann_equation, 0.01, t, args=(Omega_m0, Omega_r0, Omega_Lambda0))[:, 0]

# Model 3: Matter + cosmological constant (no radiation)
a3 = odeint(friedmann_equation, 0.01, t, args=(0.3, 0, 0.7))[:, 0]

# Model 4: Open universe (negative curvature)
a4 = odeint(friedmann_equation, 0.01, t, args=(0.3, 0, 0))[:, 0]

# Hubble parameter evolution: H(a) = H₀ √[Ω_m a⁻³ + Ω_r a⁻⁴ + Ω_Λ + Ω_k a⁻²]
a_range = np.linspace(0.1, 3, 1000)
H_matter = np.sqrt(1.0 * a_range**(-3))  # Matter only
H_Lambda_CDM = np.sqrt(Omega_m0 * a_range**(-3) + Omega_r0 * a_range**(-4) + Omega_Lambda0)
H_open = np.sqrt(0.3 * a_range**(-3) + (1-0.3) * a_range**(-2))

# Deceleration parameter: q = -aä/ȧ²
# For ΛCDM: q = (Ω_m a⁻³ + Ω_r a⁻⁴)/2 - Ω_Λ
q_Lambda_CDM = (Omega_m0 * a_range**(-3) + Omega_r0 * a_range**(-4))/2 - Omega_Lambda0

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Scale factor evolution
ax1 = axes[0, 0]
ax1.plot(t, a1, color=COLORS['blue'], linewidth=2, label='Matter only (Ω_m=1)')
ax1.plot(t, a2, color=COLORS['red'], linewidth=2, label='ΛCDM (current)')
ax1.plot(t, a3, color=COLORS['orange'], linewidth=2, label='Matter+Λ (no rad)')
ax1.plot(t, a4, color=COLORS['green'], linewidth=2, label='Open (Ω_m=0.3)')
ax1.axhline(y=1, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Today (a=1)')
ax1.set_xlabel('Time (1/H₀)', fontsize=11)
ax1.set_ylabel('Scale factor a(t)', fontsize=11)
ax1.set_title('Cosmic Expansion', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Plot 2: Hubble parameter
ax2 = axes[0, 1]
ax2.plot(a_range, H_matter, color=COLORS['blue'], linewidth=2, label='Matter only')
ax2.plot(a_range, H_Lambda_CDM, color=COLORS['red'], linewidth=2, label='ΛCDM')
ax2.plot(a_range, H_open, color=COLORS['green'], linewidth=2, label='Open')
ax2.axvline(x=1, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Today')
ax2.set_xlabel('Scale factor a', fontsize=11)
ax2.set_ylabel('Hubble parameter H/H₀', fontsize=11)
ax2.set_title('Hubble Parameter Evolution', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# Plot 3: Deceleration parameter
ax3 = axes[0, 2]
ax3.plot(a_range, q_Lambda_CDM, color=COLORS['purple'], linewidth=2)
ax3.axhline(y=0, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.7,
            label='q=0 (transition)')
ax3.axvline(x=1, color=COLORS['gray'], linestyle='--', alpha=0.5)
ax3.fill_between(a_range, -1, 0, alpha=0.1, color=COLORS['red'], label='Accelerating (q<0)')
ax3.fill_between(a_range, 0, 1, alpha=0.1, color=COLORS['blue'], label='Decelerating (q>0)')
ax3.set_xlabel('Scale factor a', fontsize=11)
ax3.set_ylabel('Deceleration parameter q', fontsize=11)
ax3.set_title('Deceleration Parameter (ΛCDM)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)
ax3.set_ylim(-1, 1)
ax3.text(1.5, -0.6, 'Accelerating\nexpansion', fontsize=9, color=COLORS['red'])
ax3.text(0.5, 0.6, 'Decelerating', fontsize=9, color=COLORS['blue'])

# Plot 4: Redshift vs scale factor
ax4 = axes[1, 0]
z = 1/a_range - 1  # Redshift: 1 + z = 1/a
ax4.plot(z, a_range, color=COLORS['cyan'], linewidth=2)
ax4.axhline(y=1, color=COLORS['gray'], linestyle='--', alpha=0.5)
ax4.axvline(x=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
ax4.set_xlabel('Redshift z', fontsize=11)
ax4.set_ylabel('Scale factor a', fontsize=11)
ax4.set_title('Redshift-Scale Factor Relation', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-0.5, 10)
ax4.set_ylim(0, 1.2)
ax4.text(5, 0.3, 'a = 1/(1+z)', fontsize=10, color=COLORS['cyan'])

# Plot 5: Energy density evolution
ax5 = axes[1, 1]
rho_m = Omega_m0 * a_range**(-3)  # Matter density ∝ a⁻³
rho_r = Omega_r0 * a_range**(-4)  # Radiation density ∝ a⁻⁴
rho_Lambda = Omega_Lambda0 * np.ones_like(a_range)  # Constant

ax5.semilogy(a_range, rho_m, color=COLORS['blue'], linewidth=2, label='Matter (∝ a⁻³)')
ax5.semilogy(a_range, rho_r, color=COLORS['orange'], linewidth=2, label='Radiation (∝ a⁻⁴)')
ax5.semilogy(a_range, rho_Lambda, color=COLORS['red'], linewidth=2,
             linestyle='--', label='Dark energy (const)')
ax5.axvline(x=1, color=COLORS['gray'], linestyle='--', alpha=0.5)
ax5.set_xlabel('Scale factor a', fontsize=11)
ax5.set_ylabel('Density parameter Ω(a)', fontsize=11)
ax5.set_title('Energy Density Evolution', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')
ax5.legend(fontsize=9)
ax5.set_xlim(0.1, 3)

# Plot 6: Age of universe vs scale factor (for ΛCDM)
# Proper time: dt = da/(a*H(a))
def age_integrand(a):
    """Integrand for age of universe"""
    if a <= 0:
        return 0
    H = np.sqrt(Omega_m0 * a**(-3) + Omega_r0 * a**(-4) + Omega_Lambda0)
    return 1/(a * H)

# Numerical integration for age
from scipy.integrate import cumtrapz
a_age = np.linspace(0.01, 2, 500)
integrand_vals = np.array([age_integrand(a) for a in a_age])
age = cumtrapz(integrand_vals, a_age, initial=0)

ax6 = axes[1, 2]
ax6.plot(a_age, age, color=COLORS['green'], linewidth=2)
ax6.axvline(x=1, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Today')
idx_today = np.argmin(np.abs(a_age - 1))
age_today = age[idx_today]
ax6.axhline(y=age_today, color=COLORS['red'], linestyle=':', alpha=0.5,
            label=f't₀ = {age_today:.2f}/H₀')
ax6.set_xlabel('Scale factor a', fontsize=11)
ax6.set_ylabel('Age (1/H₀)', fontsize=11)
ax6.set_title('Age of Universe (ΛCDM)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=9)
ax6.text(1.2, age_today*0.8, f'Age today:\n{age_today:.2f}/H₀\n≈ 13.8 Gyr',
         fontsize=9, color=COLORS['green'])

plt.tight_layout()
plt.suptitle('FLRW Cosmology: The Expanding Universe',
             fontsize=14, fontweight='bold', y=0.995)
plt.show()

print("\n" + "=" * 60)
print("FRIEDMANN EQUATIONS")
print("=" * 60)
print("First Friedmann equation:")
print("  H² = (ȧ/a)² = (8πG/3)ρ - k/a² + Λ/3")
print("\nSecond Friedmann equation (acceleration):")
print("  ä/a = -(4πG/3)(ρ + 3p) + Λ/3")
print("\nKey epochs:")
print("  1. Radiation-dominated (a << 1): ρ ∝ a⁻⁴")
print("  2. Matter-dominated (a ~ 0.3-0.7): ρ ∝ a⁻³")
print("  3. Λ-dominated (a > 0.7): ρ_Λ = const")
print(f"\nCurrent acceleration: q(a=1) = {q_Lambda_CDM[np.argmin(np.abs(a_range-1))]:.2f}")
print("  (negative → accelerating!)")
print("=" * 60)
