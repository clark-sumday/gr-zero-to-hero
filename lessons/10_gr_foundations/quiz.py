#!/usr/bin/env python3
"""
GR Foundations Practice Quiz
Tests understanding of equivalence principle, Einstein equations, geodesics, and gravitational waves
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
            print("Excellent! You have a strong grasp of GR foundations.")
        elif percentage >= 75:
            print("Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("Keep studying! Focus on the fundamentals.")
        else:
            print("Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     General Relativity Foundations - Practice Quiz      ║
║     Lesson 10: Einstein Equations and Geodesics         ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Equivalence principle
• Stress-energy tensor
• Einstein field equations
• Geodesic equation
• Weak field limit
• Gravitational waves
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Equivalence Principle
    def check_q1(answer):
        ans = answer.lower()
        is_no = 'n' in ans or 'cannot' in ans or 'can\'t' in ans
        if is_no:
            return True, "Correct! Locally, the equivalence principle says gravity and acceleration are indistinguishable. However, tidal forces (variation in gravitational field) can distinguish them over larger regions."
        else:
            return False, "Incorrect. The equivalence principle states that locally (in a small enough region), no experiment can distinguish gravity from acceleration. However, tidal effects break this equivalence over extended regions."

    quiz.question(
        "Can an astronaut in a windowless spacecraft distinguish between (a) sitting on Earth and (b) accelerating at g in deep space? (yes/no)",
        check_q1,
        hints=[
            "Think about what the equivalence principle says about local measurements",
            "The key word is 'locally' - in a small enough region of spacetime"
        ]
    )

    # Question 2: Stress-Energy Tensor
    def check_q2(answer):
        try:
            ans = float(answer)
            # T^00 = rho + p - p = rho in rest frame
            # Actually: T^00 = (rho + p)u^0 u^0 + p*(-1) = (rho+p)*1*1 - p = rho
            # For perfect fluid: T^00 = rho (energy density)
            if np.isclose(ans, 2.0):
                return True, "Correct! For a perfect fluid in its rest frame, T^00 equals the energy density ρ = 2.0"
            else:
                return False, f"Incorrect. For a perfect fluid at rest with u = (1,0,0,0), T^00 = (ρ+p)u^0 u^0 + p η^00 = (2+0.5)·1·1 + 0.5·(-1) = 2.5 - 0.5 = 2.0"
        except:
            return False, "Please enter a number"

    quiz.question(
        "For a perfect fluid at rest with energy density ρ = 2 and pressure p = 0.5, what is T^00?",
        check_q2,
        hints=[
            "Use the perfect fluid formula: T^μν = (ρ + p)u^μ u^ν + p η^μν",
            "In the rest frame: u = (1, 0, 0, 0) and η^00 = -1"
        ]
    )

    # Question 3: Perfect Fluid T^11
    def check_q3(answer):
        try:
            ans = float(answer)
            if np.isclose(ans, 0.5):
                return True, "Correct! T^11 = (ρ+p)·0·0 + p·(+1) = 0 + 0.5 = 0.5"
            else:
                return False, "Incorrect. For the same perfect fluid, T^11 = (ρ+p)u^1 u^1 + p η^11 = 0 + 0.5·1 = 0.5"
        except:
            return False, "Please enter a number"

    quiz.question(
        "For the same perfect fluid (ρ=2, p=0.5) at rest, what is T^11?",
        check_q3,
        hints=[
            "In the rest frame, u^1 = 0 (no spatial velocity)",
            "η^11 = +1 in Minkowski metric"
        ]
    )

    # Question 4: Einstein Equations
    def check_q4(answer):
        ans = answer.lower()
        is_no = 'n' in ans or 'not' in ans or 'curved' in ans
        if is_no:
            return True, "Correct! R_μν = 0 means Ricci-flat, but the full Riemann curvature tensor can still be non-zero. For example, Schwarzschild spacetime has R_μν = 0 (vacuum) but is highly curved!"
        else:
            return False, "Incorrect. The vacuum equations R_μν = 0 do NOT imply flat spacetime. The Ricci tensor is just one part of the full curvature. Schwarzschild and Kerr solutions satisfy R_μν = 0 but are definitely curved!"

    quiz.question(
        "In vacuum, the Einstein equations reduce to R_μν = 0. Does this mean spacetime is flat? (yes/no)",
        check_q4,
        hints=[
            "Consider the Schwarzschild solution - it's a vacuum solution",
            "Is spacetime around a black hole flat?"
        ]
    )

    # Question 5: Christoffel Symbols
    def check_q5(answer):
        ans = answer.lower()
        is_yes = ('y' in ans or 'zero' in ans or 'flat' in ans or
                 'cartesian' in ans or 'straight' in ans)
        if is_yes:
            return True, "Correct! If all Γ^μ_νλ = 0, the geodesic equation becomes d²x^μ/dτ² = 0, which describes straight lines (constant velocity). This occurs in flat spacetime with Cartesian coordinates."
        else:
            return False, "Incorrect. When all Christoffel symbols vanish, the geodesic equation d²x^μ/dτ² + Γ^μ_νλ (dx^ν/dτ)(dx^λ/dτ) = 0 reduces to d²x^μ/dτ² = 0, which means particles move in straight lines - i.e., flat spacetime!"

    quiz.question(
        "In the geodesic equation, if all Christoffel symbols Γ^μ_νλ = 0, do particles move in straight lines? (yes/no)",
        check_q5,
        hints=[
            "Look at the geodesic equation: d²x^μ/dτ² + Γ^μ_νλ (dx^ν/dτ)(dx^λ/dτ) = 0",
            "What happens when you set Γ = 0?"
        ]
    )

    # Question 6: Weak Field Limit
    def check_q6(answer):
        try:
            # g_00 = -(1 + 2Φ/c²) where Φ = -GM/r
            # At Earth surface: Φ = -GM/R
            # g_00 = -(1 - 2GM/(Rc²))
            # Deviation from -1 is 2GM/(Rc²)
            # For Earth: GM/R ≈ 7e14 m³/s² / 6.4e6 m ≈ 1.1e8 m²/s²
            # 2GM/(Rc²) ≈ 2.4e-9
            ans = float(answer)
            if np.isclose(ans, -1.0, atol=0.01):
                return True, "Correct! At Earth's surface, g_00 ≈ -1.000000001, extremely close to flat spacetime value of -1. The weak field approximation is very good!"
            else:
                return False, "The answer is approximately -1 (very close to flat space). The exact value is g_00 = -(1 - 2GM_Earth/(R_Earth·c²)) ≈ -1.000000001"
        except:
            return False, "Please enter a number (approximately -1)"

    quiz.question(
        "In the weak field limit, g_00 = -(1 + 2Φ/c²). At Earth's surface, approximately what is g_00? (hint: very close to -1)",
        check_q6,
        hints=[
            "The Newtonian potential Φ = -GM/r is very small compared to c²",
            "For Earth, |Φ|/c² ~ 10^-9, so g_00 ≈ -1.000000001"
        ]
    )

    # Question 7: Gravitational Wave Polarizations
    def check_q7(answer):
        try:
            ans = int(answer)
            if ans == 2:
                return True, "Correct! Gravitational waves have two independent polarizations: + (plus) and × (cross). This is because GWs are transverse and traceless in the appropriate gauge."
            else:
                return False, "Incorrect. Gravitational waves have 2 independent polarizations: + and ×. Unlike EM waves which also have 2, GWs are quadrupole radiation."
        except:
            return False, "Please enter a number"

    quiz.question(
        "How many independent polarizations do gravitational waves have?",
        check_q7,
        hints=[
            "Think about the + and × polarizations",
            "GWs are transverse waves, like EM waves"
        ]
    )

    # Question 8: Gravitational Wave Strain
    def check_q8(answer):
        try:
            # h = ΔL/L, so ΔL = h × L
            ans = float(answer)
            h = 1e-21
            L = 4e3  # 4 km = 4000 m
            delta_L = h * L
            # delta_L = 4e-18 m
            if np.isclose(ans, delta_L, rtol=0.5):
                return True, f"Correct! ΔL = h × L = 10^(-21) × 4000 m = 4×10^(-18) m, about 1/1000 the diameter of a proton! This shows the incredible sensitivity of LIGO."
            else:
                return False, f"Incorrect. The strain h = ΔL/L, so ΔL = h × L = 10^(-21) × 4000 m = 4×10^(-18) m"
        except:
            return False, "Please enter a number (use scientific notation, e.g., 4e-18)"

    quiz.question(
        "A gravitational wave has strain h = 10^(-21). If a LIGO arm is L = 4 km, what is ΔL in meters? (use scientific notation, e.g., 4e-18)",
        check_q8,
        hints=[
            "The strain h is defined as h = ΔL/L",
            "Solve for ΔL: ΔL = h × L"
        ]
    )

    # Question 9: Newtonian Limit
    def check_q9(answer):
        ans = answer.lower()
        has_poisson = 'poisson' in ans or 'nabla' in ans or '∇²' in ans or 'laplacian' in ans
        if has_poisson:
            return True, "Correct! In the weak field limit, the Einstein equations reduce to Poisson's equation: ∇²Φ = 4πGρ, which is exactly Newton's law of gravity!"
        else:
            return False, "Incorrect. The Einstein equations reduce to Poisson's equation: ∇²Φ = 4πGρ in the weak field, slow motion limit. This is the fundamental equation of Newtonian gravity."

    quiz.question(
        "What Newtonian equation do the Einstein equations reduce to in the weak field limit? (name the equation)",
        check_q9,
        hints=[
            "This equation relates the gravitational potential Φ to the mass density ρ",
            "It's ∇²Φ = 4πGρ"
        ]
    )

    # Question 10: Geodesic Equation Physical Meaning
    def check_q10(answer):
        ans = answer.lower()
        has_free = ('free' in ans or 'inertial' in ans or 'no force' in ans or
                   'f=0' in ans or 'zero' in ans)
        if has_free:
            return True, "Correct! The geodesic equation describes the motion of free particles (no forces except gravity). It's the GR equivalent of Newton's F = ma with F = 0, but in curved spacetime!"
        else:
            return False, "Incorrect. The geodesic equation d²x^μ/dτ² + Γ^μ_νλ (dx^ν/dτ)(dx^λ/dτ) = 0 describes free particles in curved spacetime. It's the GR version of 'force-free motion' or Newton's first law in curved geometry."

    quiz.question(
        "What does the geodesic equation describe physically? (hint: think about Newton's laws)",
        check_q10,
        hints=[
            "Consider a particle moving with no forces acting on it",
            "It's like Newton's F = 0, but in curved spacetime"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\nReview the lesson material in LESSON.md for any topics you missed.")
    print("Try the example scripts in the examples/ directory to explore more.\n")


if __name__ == "__main__":
    main()
