#!/usr/bin/env python3
"""
GR Phenomena Practice Quiz
Tests understanding of gravitational lensing, GPS, black hole thermodynamics, and observations
"""

import numpy as np

class Quiz:
    def __init__(self):
        self.score = 0
        self.total = 0

    def question(self, prompt, answer_checker, hints=None):
        """Ask a question and check the answer"""
        self.total += 1
        print(f"\n{'='*60}")
        print(f"Question {self.total}:")
        print(prompt)
        print('='*60)

        if hints:
            show_hint = input("\nWould you like a hint? (y/n): ").strip().lower()
            if show_hint == 'y':
                for i, hint in enumerate(hints, 1):
                    print(f"  Hint {i}: {hint}")

        user_answer = input("\nYour answer: ").strip()

        is_correct, feedback = answer_checker(user_answer)

        if is_correct:
            print("\n✓ Correct!")
            self.score += 1
        else:
            print("\n✗ Incorrect.")

        print(feedback)
        input("\nPress Enter to continue...")

    def show_results(self):
        """Display final score"""
        percentage = (self.score / self.total * 100) if self.total > 0 else 0
        print(f"\n{'='*60}")
        print(f"FINAL SCORE: {self.score}/{self.total} ({percentage:.0f}%)")
        print('='*60)

        if percentage >= 90:
            print("Excellent! You've mastered GR phenomena!")
        elif percentage >= 75:
            print("Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("Keep studying! Focus on the fundamentals.")
        else:
            print("Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     General Relativity Phenomena - Practice Quiz        ║
║     Lesson 12: Real-World Applications and Tests        ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Gravitational lensing
• Gravitational redshift
• Frame dragging
• GPS and time dilation
• Black hole thermodynamics
• Cosmological observations
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Light Deflection Scaling
    def check_q1(answer):
        try:
            ans = float(answer)
            # α ∝ 1/b, so α(2R) = α(R)/2 = 1.75/2 = 0.875
            if np.isclose(ans, 0.875, rtol=0.1):
                return True, "Correct! Since deflection angle α ∝ 1/b (inversely proportional to impact parameter), doubling b halves α: 1.75\"/2 = 0.875 arcseconds."
            else:
                return False, "Incorrect. The deflection angle α = 4GM/(c²b) is inversely proportional to impact parameter. Double the distance → half the deflection: 1.75\"/2 = 0.875\""
        except:
            return False, "Please enter a number (in arcseconds)"

    quiz.question(
        "Light is deflected by 1.75 arcseconds grazing the Sun. What would the deflection be at b = 2R_sun? (in arcseconds)",
        check_q1,
        hints=[
            "The deflection angle α is inversely proportional to impact parameter b",
            "α ∝ 1/b, so doubling b halves α"
        ]
    )

    # Question 2: Einstein Ring Formation
    def check_q2(answer):
        ans = answer.lower()
        has_alignment = 'align' in ans or 'line' in ans or 'behind' in ans
        if has_alignment:
            return True, "Correct! An Einstein ring forms when the source, lens, and observer are perfectly aligned. Light from the source is bent equally in all directions around the lens, creating a circular ring image."
        else:
            return False, "Incorrect. An Einstein ring occurs when the background source, the lensing mass, and the observer are perfectly aligned. The source appears as a circular ring around the lens with angular radius θ_E (the Einstein radius)."

    quiz.question(
        "What is an Einstein ring and when does it form? (brief explanation)",
        check_q2,
        hints=[
            "Think about the geometry: source, lens, observer",
            "What happens when they're perfectly lined up?"
        ]
    )

    # Question 3: Neutron Star Redshift
    def check_q3(answer):
        try:
            ans = float(answer)
            # r_s = 2GM/c² for M = 1.4 M_sun ≈ 4.1 km
            # R = 10 km, so r_s/R ≈ 0.41
            # z = 1/√(1 - r_s/R) - 1 ≈ 1/√(0.59) - 1 ≈ 1.30 - 1 = 0.30
            # λ_obs = λ_emit(1 + z) = 500 × 1.30 = 650 nm
            if np.isclose(ans, 650, rtol=0.15):
                return True, f"Correct! The gravitational redshift is z ≈ 0.30, so λ_obs = 500 nm × 1.30 ≈ 650 nm. The photon shifts from green to red!"
            else:
                return False, "Incorrect. First find z from z = 1/√(1 - r_s/R) - 1 where r_s ≈ 4.1 km, R = 10 km. This gives z ≈ 0.30. Then λ_obs = 500(1 + 0.30) = 650 nm"
        except:
            return False, "Please enter a number (wavelength in nm)"

    quiz.question(
        "A photon with λ = 500 nm is emitted from a neutron star (M = 1.4 M_sun, R = 10 km). What wavelength (in nm) is observed at infinity?",
        check_q3,
        hints=[
            "Calculate the Schwarzschild radius: r_s ≈ 4.1 km",
            "Use z = 1/√(1 - r_s/R) - 1, then λ_obs = λ_emit(1 + z)"
        ]
    )

    # Question 4: GPS Dominant Effect
    def check_q4(answer):
        ans = answer.lower()
        is_gr = 'gr' in ans or 'general' in ans or 'gravitational' in ans
        if is_gr:
            return True, "Correct! General Relativity (gravitational time dilation) dominates with +45.9 μs/day, while Special Relativity contributes -7.2 μs/day. The net effect is +38.7 μs/day (GR wins!)."
        else:
            return False, "Incorrect. For GPS satellites, GR (gravitational time dilation) gives +45.9 μs/day while SR (kinematic time dilation) gives -7.2 μs/day. The GR effect is larger and dominates!"

    quiz.question(
        "GPS satellites need relativistic corrections. Which effect dominates: SR or GR? (answer SR or GR)",
        check_q4,
        hints=[
            "SR: clocks run slower due to motion (-7.2 μs/day)",
            "GR: clocks run faster at higher altitude (+45.9 μs/day)"
        ]
    )

    # Question 5: GPS Position Error
    def check_q5(answer):
        try:
            ans = float(answer)
            # Net time error: 38.7 μs/day = 38.7e-6 s/day
            # Per hour: 38.7e-6 / 24 ≈ 1.6e-6 s
            # Position error: c × Δt = 3e8 × 1.6e-6 ≈ 480 m
            if np.isclose(ans, 480, rtol=0.3):  # Accept 340-620 m
                return True, f"Correct! Time error per hour: 38.7 μs/day / 24 ≈ 1.6 μs/hour. Position error: c × Δt ≈ 480 meters. GPS would be off by nearly half a kilometer after just one hour!"
            else:
                return False, "Incorrect. Time error per hour: (38.7 μs/day) / 24 ≈ 1.6 μs. Position error: 3×10⁸ m/s × 1.6×10⁻⁶ s ≈ 480 meters"
        except:
            return False, "Please enter a number (in meters)"

    quiz.question(
        "Without relativistic corrections (38.7 μs/day error), how much GPS position error (in meters) accumulates in one hour?",
        check_q5,
        hints=[
            "Scale the daily error (38.7 μs) to one hour",
            "Position error = c × time error"
        ]
    )

    # Question 6: Frame Dragging
    def check_q6(answer):
        ans = answer.lower()
        has_rotation = 'rotat' in ans or 'spin' in ans or 'drag' in ans or 'twist' in ans
        if has_rotation:
            return True, "Correct! Frame dragging (Lense-Thirring effect) is the phenomenon where rotating masses 'drag' or 'twist' the spacetime around them, causing nearby objects and reference frames to be pulled along in the direction of rotation."
        else:
            return False, "Incorrect. Frame dragging is the effect where a rotating mass drags spacetime around with it, like a ball spinning in honey. Nearby gyroscopes precess and orbits twist due to this 'twisting' of spacetime."

    quiz.question(
        "What is the physical interpretation of frame dragging? (brief explanation)",
        check_q6,
        hints=[
            "Think about what rotating masses do to spacetime",
            "Imagine spinning a ball in honey - what happens to the honey?"
        ]
    )

    # Question 7: Hawking Temperature Scaling
    def check_q7(answer):
        ans = answer.lower()
        has_inverse = 'inverse' in ans or '1/m' in ans or 'smaller' in ans or 'lower' in ans or 'cold' in ans
        if has_inverse:
            return True, "Correct! Hawking temperature T_H ∝ 1/M is inversely proportional to mass. Larger black holes have lower temperatures and radiate less. A supermassive black hole is much colder than a stellar-mass one!"
        else:
            return False, "Incorrect. Hawking temperature T_H = ℏc³/(8πGMk_B) ∝ 1/M. Larger black holes are COLDER! A billion solar mass black hole has temperature ~10⁻¹⁴ K, while a stellar mass one has ~10⁻⁷ K."

    quiz.question(
        "How does Hawking temperature scale with black hole mass? Why do larger black holes have lower temperatures?",
        check_q7,
        hints=[
            "Look at the formula: T_H ∝ 1/M",
            "Temperature is inversely proportional to mass"
        ]
    )

    # Question 8: Black Hole Information Paradox
    def check_q8(answer):
        ans = answer.lower()
        has_info = 'information' in ans or 'quantum' in ans or 'thermal' in ans
        has_lost = 'lost' in ans or 'destroy' in ans or 'paradox' in ans or 'where' in ans
        if has_info:
            return True, "Correct! The information paradox asks: if you drop information (like a book) into a black hole, and the black hole evaporates via thermal Hawking radiation, where does the information go? Quantum mechanics says information cannot be destroyed, but Hawking radiation appears thermal (random). Current thinking involves the holographic principle - information is encoded on the horizon."
        else:
            return False, "Incorrect. The information paradox: Hawking radiation is thermal (appears random), but quantum mechanics says information cannot be destroyed. If a black hole evaporates completely, where did the information about what fell in go? This is still debated, with the holographic principle as a leading resolution."

    quiz.question(
        "What is the black hole information paradox? (brief explanation)",
        check_q8,
        hints=[
            "Think about what happens to information that falls into a black hole",
            "Hawking radiation is thermal - does it carry information out?"
        ]
    )

    # Question 9: Hubble Law
    def check_q9(answer):
        try:
            ans = float(answer)
            # v = H₀ × d = 70 km/s/Mpc × 100 Mpc = 7000 km/s
            if np.isclose(ans, 7000, rtol=0.15):
                return True, f"Correct! v = H₀ × d = 70 km/s/Mpc × 100 Mpc = 7000 km/s. The redshift z ≈ v/c ≈ 0.023."
            else:
                return False, "Incorrect. Use Hubble's law: v = H₀ × d = 70 × 100 = 7000 km/s"
        except:
            return False, "Please enter a number (velocity in km/s)"

    quiz.question(
        "A galaxy is at d = 100 Mpc. Using H₀ = 70 km/s/Mpc, what is its recession velocity (in km/s)?",
        check_q9,
        hints=[
            "Use Hubble's law: v = H₀ × d",
            "Just multiply: 70 × 100"
        ]
    )

    # Question 10: CMB Significance
    def check_q10(answer):
        ans = answer.lower()
        has_big_bang = 'big bang' in ans or 'early universe' in ans or 'recombination' in ans
        has_thermal = 'thermal' in ans or 'radiation' in ans or 'hot' in ans
        if has_big_bang or has_thermal:
            return True, "Correct! The CMB (Cosmic Microwave Background) is thermal radiation from the early universe at T ≈ 2.7 K. It was released when the universe became transparent (recombination, z ≈ 1100), providing direct evidence for the hot Big Bang model. The CMB power spectrum constrains cosmological parameters with incredible precision."
        else:
            return False, "Incorrect. The CMB is the afterglow of the Big Bang - thermal radiation from when the universe was ~380,000 years old. It confirms the hot Big Bang model and provides precise measurements of cosmological parameters (flat universe, dark energy, etc.)."

    quiz.question(
        "What is the Cosmic Microwave Background (CMB) and why is it important? (brief explanation)",
        check_q10,
        hints=[
            "Think about radiation left over from the early universe",
            "It provides evidence for the Big Bang and measures cosmological parameters"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\nCongratulations on completing the GR tutorial!")
    print("Review LESSON.md for any topics you want to explore further.")
    print("You now understand one of the most beautiful theories in physics!\n")


if __name__ == "__main__":
    main()
