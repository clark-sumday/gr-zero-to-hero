#!/usr/bin/env python3
"""
GR Solutions Practice Quiz
Tests understanding of Schwarzschild, Kerr, FLRW, and gravitational wave solutions
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
            print("Excellent! You have mastered GR solutions.")
        elif percentage >= 75:
            print("Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("Keep studying! Focus on the fundamentals.")
        else:
            print("Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     Solutions to Einstein Equations - Practice Quiz     ║
║     Lesson 11: Black Holes, Cosmology, and GWs          ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Schwarzschild solution and black holes
• Kerr solution (rotating black holes)
• FLRW cosmology
• Gravitational wave solutions
• de Sitter space
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Schwarzschild Radius
    def check_q1(answer):
        try:
            ans = float(answer)
            # r_s = 2GM/c² for M = 10 M_sun
            # For solar mass: r_s ≈ 3 km, so 10 M_sun ≈ 30 km
            # More precisely: r_s = 2 * 6.67e-11 * 10*1.989e30 / (3e8)^2 ≈ 29.5 km
            if np.isclose(ans, 30, rtol=0.2):  # Accept 24-36 km
                return True, f"Correct! r_s = 2GM/c² ≈ 30 km for M = 10 M_sun. (Precise value: 29.5 km)"
            else:
                return False, "Incorrect. For solar mass, r_s ≈ 3 km, so for 10 M_sun, r_s ≈ 30 km"
        except:
            return False, "Please enter a number (in km)"

    quiz.question(
        "What is the Schwarzschild radius (in km) for a black hole with M = 10 M_sun?",
        check_q1,
        hints=[
            "Use r_s = 2GM/c². For one solar mass, r_s ≈ 3 km",
            "The Schwarzschild radius scales linearly with mass"
        ]
    )

    # Question 2: Time Dilation Factor
    def check_q2(answer):
        try:
            ans = float(answer)
            # dτ/dt = √(1 - r_s/r) = 1/2
            # So 1 - r_s/r = 1/4, thus r_s/r = 3/4, r = 4r_s/3
            if np.isclose(ans, 4/3, rtol=0.05):
                return True, "Correct! When √(1 - r_s/r) = 1/2, solving gives r = 4r_s/3 ≈ 1.33 r_s"
            else:
                return False, "Incorrect. Time dilation factor dτ/dt = √(1 - r_s/r). Set this equal to 1/2 and solve: 1 - r_s/r = 1/4, so r = 4r_s/3"
        except:
            return False, "Please enter a number (as multiple of r_s, e.g., 1.33)"

    quiz.question(
        "At what radius r (in units of r_s) is the gravitational time dilation factor dτ/dt exactly 1/2?",
        check_q2,
        hints=[
            "Time dilation factor is dτ/dt = √(1 - r_s/r)",
            "Set √(1 - r_s/r) = 1/2 and solve for r"
        ]
    )

    # Question 3: Photon Sphere
    def check_q3(answer):
        ans = answer.lower()
        has_orbit = 'orbit' in ans or 'circular' in ans or 'light' in ans or 'photon' in ans
        if has_orbit:
            return True, "Correct! The photon sphere at r = 1.5 r_s is where photons can orbit the black hole in unstable circular orbits. Any photon at this radius moving tangentially will orbit forever (though the orbit is unstable)."
        else:
            return False, "Incorrect. At the photon sphere (r = 1.5 r_s or 3M), photons can orbit the black hole in circular paths. These orbits are unstable - slight perturbations cause photons to either escape or fall in."

    quiz.question(
        "What is special about the photon sphere at r = 1.5 r_s?",
        check_q3,
        hints=[
            "Think about the path light can take at this radius",
            "The word 'photon' is a hint!"
        ]
    )

    # Question 4: ISCO
    def check_q4(answer):
        try:
            ans = float(answer)
            # ISCO for Schwarzschild is at r = 6M = 3r_s
            if np.isclose(ans, 6, rtol=0.1) or np.isclose(ans, 3, rtol=0.1):
                return True, "Correct! The ISCO (Innermost Stable Circular Orbit) is at r = 6M = 3r_s for a Schwarzschild black hole. Inside this radius, circular orbits are unstable."
            else:
                return False, "Incorrect. The ISCO for a Schwarzschild black hole is at r = 6M = 3r_s. Any closer and orbits spiral inward!"
        except:
            return False, "Please enter a number (in units of M or r_s)"

    quiz.question(
        "What is the radius of the ISCO (Innermost Stable Circular Orbit) for a Schwarzschild black hole? (in units of M or r_s)",
        check_q4,
        hints=[
            "ISCO is where the innermost stable circular orbit exists",
            "It's at r = 6M = 3r_s for Schwarzschild"
        ]
    )

    # Question 5: Inside Event Horizon
    def check_q5(answer):
        ans = answer.lower()
        is_no = 'n' in ans or 'cannot' in ans or 'impossible' in ans
        if is_no:
            return True, "Correct! No observer inside the event horizon can escape, not even with light-speed rockets. Inside the horizon, all future-directed paths lead to the singularity at r = 0. The singularity is in your future, not at a place you can avoid."
        else:
            return False, "Incorrect. Once inside the event horizon (r < r_s), escape is impossible. The r coordinate becomes timelike and the singularity at r = 0 is in your future - as inevitable as moving forward in time!"

    quiz.question(
        "Can an observer inside the event horizon ever escape? (yes/no)",
        check_q5,
        hints=[
            "Think about what 'event horizon' means",
            "Inside the horizon, even light cannot escape"
        ]
    )

    # Question 6: Coordinate vs Physical Singularity
    def check_q6(answer):
        ans = answer.lower()
        has_coord = 'coordinate' in ans
        has_physical = 'physical' in ans or 'curvature' in ans
        if has_coord and has_physical:
            return True, "Correct! A coordinate singularity (like r = r_s in Schwarzschild coordinates) is just an artifact of the coordinate system - it can be removed by choosing different coordinates. A physical singularity (like r = 0) has infinite curvature and cannot be removed by any coordinate choice."
        else:
            return False, "Incorrect. Coordinate singularity: artifact of coordinates (e.g., r = r_s), can be removed by changing coordinates. Physical singularity: real divergence in curvature (e.g., r = 0), cannot be removed."

    quiz.question(
        "What is the difference between a coordinate singularity and a physical singularity? (brief explanation)",
        check_q6,
        hints=[
            "Think about the event horizon vs the center of a black hole",
            "One can be 'fixed' by choosing better coordinates, the other cannot"
        ]
    )

    # Question 7: Maximum Kerr Spin
    def check_q7(answer):
        try:
            ans = float(answer)
            if np.isclose(ans, 1.0, atol=0.05):
                return True, "Correct! The maximum spin parameter is a = M (or a/M = 1). This is called an extremal Kerr black hole. If a > M, you'd get a naked singularity (believed to be forbidden by cosmic censorship)."
            else:
                return False, "Incorrect. The maximum allowed spin parameter is a = M (a/M = 1). Above this, the event horizon disappears and you get a naked singularity."
        except:
            return False, "Please enter a number (a/M ratio)"

    quiz.question(
        "What is the maximum possible spin parameter a/M for a Kerr black hole?",
        check_q7,
        hints=[
            "Think about what happens when a > M",
            "The maximum is when a = M (extremal Kerr)"
        ]
    )

    # Question 8: Ergosphere
    def check_q8(answer):
        ans = answer.lower()
        has_frame = 'frame' in ans or 'drag' in ans or 'rotate' in ans or 'co-rotate' in ans
        has_location = 'outside' in ans or 'horizon' in ans
        if has_frame or has_location:
            return True, "Correct! The ergosphere is the region outside the event horizon where frame-dragging is so strong that nothing can remain stationary - everything must co-rotate with the black hole. It exists between the outer horizon r_+ and the ergosphere boundary r_ergo = M + √(M² - a²cos²θ)."
        else:
            return False, "Incorrect. The ergosphere is the region between the event horizon and the ergosphere boundary where spacetime is dragged so strongly that stationary observers cannot exist - all objects must co-rotate with the black hole."

    quiz.question(
        "Describe the ergosphere. Where is it located?",
        check_q8,
        hints=[
            "Think about frame-dragging effects",
            "It's outside the event horizon but has special properties"
        ]
    )

    # Question 9: Kerr Horizons
    def check_q9(answer):
        try:
            ans = int(answer)
            if ans == 2:
                return True, "Correct! A Kerr black hole (with a < M) has two horizons: the outer event horizon r_+ = M + √(M² - a²) and the inner (Cauchy) horizon r_- = M - √(M² - a²)."
            else:
                return False, "Incorrect. Kerr black holes have 2 horizons: outer event horizon r_+ and inner Cauchy horizon r_-. (For extremal a = M, they coincide.)"
        except:
            return False, "Please enter a number"

    quiz.question(
        "How many horizons does a Kerr black hole have (for 0 < a < M)?",
        check_q9,
        hints=[
            "Think about the outer and inner horizons",
            "There's r_+ and r_-"
        ]
    )

    # Question 10: Scale Factor and Redshift
    def check_q10(answer):
        try:
            ans = float(answer)
            # If a doubles, wavelengths double: z = a_now/a_then - 1
            # If a_now = 2a_then, z = 2 - 1 = 1
            # But question asks about wavelength: λ_obs/λ_emit = a_now/a_then = 2
            # So wavelength doubles
            if np.isclose(ans, 2, rtol=0.1):
                return True, "Correct! Wavelength λ scales with the scale factor: λ ∝ a. If the scale factor doubles, wavelengths double (the universe expands and stretches light)."
            else:
                return False, "Incorrect. Wavelength scales with the scale factor: λ_obs/λ_emit = a_obs/a_emit. If a doubles, so does the wavelength."
        except:
            return False, "Please enter a number (the multiplicative factor)"

    quiz.question(
        "If the scale factor a(t) doubles, by what factor does the wavelength of light change?",
        check_q10,
        hints=[
            "Think about how light stretches as the universe expands",
            "λ ∝ a"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\nReview the lesson material in LESSON.md for any topics you missed.")
    print("Explore the code examples to deepen your understanding.\n")


if __name__ == "__main__":
    main()
