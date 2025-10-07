#!/usr/bin/env python3
"""
Example: GPS Relativistic Corrections
Demonstrates gravitational and kinematic time dilation effects on GPS satellites
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

print("=" * 60)
print("GPS AND GENERAL RELATIVITY")
print("=" * 60)
print("\nGPS satellites require relativistic corrections!")
print("Without GR/SR corrections: ~10 km/day position error")

# Physical constants
c = 2.998e8  # m/s (speed of light)
G = 6.674e-11  # m^3 kg^-1 s^-2
M_earth = 5.972e24  # kg
R_earth = 6.371e6  # m

# GPS orbital parameters
h_gps = 20200e3  # m (GPS orbit altitude above surface)
r_gps = R_earth + h_gps  # m (orbital radius)
v_gps = np.sqrt(G * M_earth / r_gps)  # Orbital velocity
T_gps = 2 * np.pi * r_gps / v_gps  # Orbital period

print(f"\n" + "=" * 60)
print("GPS SATELLITE PARAMETERS")
print("=" * 60)
print(f"Orbital altitude: h = {h_gps/1000:.0f} km")
print(f"Orbital radius: r = {r_gps/1000:.0f} km")
print(f"Orbital velocity: v = {v_gps:.0f} m/s = {v_gps/1000:.2f} km/s")
print(f"Orbital period: T = {T_gps/3600:.2f} hours")

# Gravitational time dilation (General Relativity)
# Δt/t = -GM/(c²r)
# Clock runs FASTER at higher altitude (weaker gravity)

def grav_time_dilation(r, M=M_earth, c=c, G=G):
    """Gravitational time dilation factor: Δt/t"""
    return -G * M / (c**2 * r)

# At Earth's surface
dt_earth = grav_time_dilation(R_earth)

# At GPS altitude
dt_gps = grav_time_dilation(r_gps)

# Difference (positive means GPS clock runs faster)
dt_grav = dt_gps - dt_earth

# Convert to seconds per day
seconds_per_day = 86400
dt_grav_per_day = dt_grav * seconds_per_day

print(f"\n" + "=" * 60)
print("GRAVITATIONAL TIME DILATION (GR)")
print("=" * 60)
print(f"At Earth surface: Δt/t = {dt_earth:.3e}")
print(f"At GPS altitude: Δt/t = {dt_gps:.3e}")
print(f"Difference: Δt/t = {dt_grav:.3e}")
print(f"\nGPS clock runs FASTER by: {dt_grav_per_day*1e6:.2f} μs/day")
print(f"  (due to weaker gravitational field)")

# Kinematic time dilation (Special Relativity)
# Δt/t = -v²/(2c²)
# Clock runs SLOWER due to motion

dt_kinematic = -v_gps**2 / (2 * c**2)
dt_kinematic_per_day = dt_kinematic * seconds_per_day

print(f"\n" + "=" * 60)
print("KINEMATIC TIME DILATION (SR)")
print("=" * 60)
print(f"Velocity: v = {v_gps:.0f} m/s")
print(f"Time dilation: Δt/t = {dt_kinematic:.3e}")
print(f"\nGPS clock runs SLOWER by: {abs(dt_kinematic_per_day)*1e6:.2f} μs/day")
print(f"  (due to orbital motion)")

# Total relativistic correction
dt_total = dt_grav + dt_kinematic
dt_total_per_day = dt_total * seconds_per_day

print(f"\n" + "=" * 60)
print("TOTAL RELATIVISTIC CORRECTION")
print("=" * 60)
print(f"Gravitational (GR): +{dt_grav_per_day*1e6:.2f} μs/day (faster)")
print(f"Kinematic (SR): {dt_kinematic_per_day*1e6:.2f} μs/day (slower)")
print(f"Net effect: +{dt_total_per_day*1e6:.2f} μs/day")
print(f"\nGPS clock runs FASTER overall!")

# Position error if uncorrected
# Distance error = c × time error
position_error_per_day = c * dt_total_per_day

print(f"\n" + "=" * 60)
print("POSITION ERROR WITHOUT CORRECTION")
print("=" * 60)
print(f"Time error: {dt_total_per_day*1e6:.2f} μs/day")
print(f"Position error: {position_error_per_day/1000:.2f} km/day")
print(f"  (Would accumulate rapidly!)")

# Visualization 1: Time dilation vs altitude
altitudes = np.linspace(0, 40000e3, 1000)  # 0 to 40,000 km
radii = R_earth + altitudes

# Gravitational effect
dt_grav_alt = np.array([grav_time_dilation(r) - dt_earth for r in radii])

# Kinematic effect (circular orbit velocity at each altitude)
v_orbit = np.sqrt(G * M_earth / radii)
dt_kin_alt = -v_orbit**2 / (2 * c**2)

# Total
dt_total_alt = dt_grav_alt + dt_kin_alt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Time dilation components
ax1 = axes[0, 0]
ax1.plot(altitudes/1000, dt_grav_alt * seconds_per_day * 1e6,
         color=COLORS['blue'], linewidth=2, label='Gravitational (GR)')
ax1.plot(altitudes/1000, dt_kin_alt * seconds_per_day * 1e6,
         color=COLORS['orange'], linewidth=2, label='Kinematic (SR)')
ax1.plot(altitudes/1000, dt_total_alt * seconds_per_day * 1e6,
         color=COLORS['green'], linewidth=3, label='Total')
ax1.axvline(x=h_gps/1000, color=COLORS['red'], linestyle='--',
            linewidth=2, alpha=0.5, label=f'GPS orbit ({h_gps/1000:.0f} km)')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.set_xlabel('Altitude (km)', fontsize=11)
ax1.set_ylabel('Time dilation (μs/day)', fontsize=11)
ax1.set_title('Relativistic Time Dilation vs Altitude', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Plot 2: Position error accumulation
days = np.linspace(0, 7, 1000)
position_error = c * dt_total * seconds_per_day * days

ax2 = axes[0, 1]
ax2.plot(days, position_error/1000, color=COLORS['purple'], linewidth=2)
ax2.set_xlabel('Time (days)', fontsize=11)
ax2.set_ylabel('Accumulated position error (km)', fontsize=11)
ax2.set_title('GPS Position Error (Uncorrected)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(3.5, 50, f'~{position_error_per_day/1000:.1f} km/day\naccumulation',
         fontsize=10, color=COLORS['purple'], ha='center')

# Plot 3: Clock frequency correction
# GPS operates at nominal frequency, corrected on ground
f_nominal = 10.23e6  # MHz (nominal GPS frequency)
f_correction = f_nominal * dt_total  # Frequency offset

ax3 = axes[1, 0]
correction_factors = dt_total_alt * seconds_per_day * 1e6

ax3.plot(altitudes/1000, correction_factors, color=COLORS['cyan'], linewidth=2)
ax3.axvline(x=h_gps/1000, color=COLORS['red'], linestyle='--',
            linewidth=2, alpha=0.5, label=f'GPS orbit')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.set_xlabel('Altitude (km)', fontsize=11)
ax3.set_ylabel('Clock correction (μs/day)', fontsize=11)
ax3.set_title('Required Clock Correction', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

# Plot 4: Comparison with other satellites
sat_altitudes = np.array([400, 20200, 35786]) / 1000  # ISS, GPS, GEO in km
sat_radii = R_earth + sat_altitudes * 1000
sat_names = ['ISS', 'GPS', 'GEO']

sat_dt_grav = np.array([grav_time_dilation(r) - dt_earth for r in sat_radii])
sat_v = np.sqrt(G * M_earth / sat_radii)
sat_dt_kin = -sat_v**2 / (2 * c**2)
sat_dt_total = sat_dt_grav + sat_dt_kin

ax4 = axes[1, 1]
x_pos = np.arange(len(sat_names))
width = 0.25

ax4.bar(x_pos - width, sat_dt_grav * seconds_per_day * 1e6, width,
        color=COLORS['blue'], label='GR (gravitational)')
ax4.bar(x_pos, sat_dt_kin * seconds_per_day * 1e6, width,
        color=COLORS['orange'], label='SR (kinematic)')
ax4.bar(x_pos + width, sat_dt_total * seconds_per_day * 1e6, width,
        color=COLORS['green'], label='Total')
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.set_ylabel('Time dilation (μs/day)', fontsize=11)
ax4.set_title('Time Dilation for Different Satellites', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(sat_names)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Additional info
print(f"\n" + "=" * 60)
print("HOW GPS CORRECTS FOR RELATIVITY")
print("=" * 60)
print("\n1. FACTORY SETTING:")
print(f"   Satellite clocks set to run at {f_nominal - f_correction:.2f} Hz")
print(f"   instead of nominal {f_nominal/1e6:.3f} MHz")
print(f"   Offset: {f_correction:.2f} Hz")
print("\n2. ON-ORBIT:")
print("   Relativistic effects make clock run at correct rate")
print("\n3. GROUND STATIONS:")
print("   Continuous monitoring and small adjustments")
print("   Upload ephemeris and clock corrections")

print(f"\n" + "=" * 60)
print("DIFFERENT ORBITS COMPARISON")
print("=" * 60)
for i, name in enumerate(sat_names):
    print(f"\n{name} ({sat_altitudes[i]:.0f} km altitude):")
    print(f"  Velocity: {sat_v[i]:.0f} m/s")
    print(f"  GR effect: +{sat_dt_grav[i]*seconds_per_day*1e6:.2f} μs/day")
    print(f"  SR effect: {sat_dt_kin[i]*seconds_per_day*1e6:.2f} μs/day")
    print(f"  Total: +{sat_dt_total[i]*seconds_per_day*1e6:.2f} μs/day")

print(f"\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("• GPS is a practical test of General Relativity")
print("• BOTH GR and SR effects are significant")
print("• GR dominates (weaker gravity → faster clock)")
print("• SR opposes (motion → slower clock)")
print("• Net: GPS clocks run ~38 μs/day faster")
print("• Without correction: ~10 km/day error")
print("• GR is essential for modern navigation!")
print("=" * 60)
