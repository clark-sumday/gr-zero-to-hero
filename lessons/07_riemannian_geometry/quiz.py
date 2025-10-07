#!/usr/bin/env python3
"""
Riemannian Geometry Practice Quiz
Tests understanding of metric tensor, Christoffel symbols, curvature, and geodesics
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
            print("🌟 Excellent! You understand Riemannian geometry.")
        elif percentage >= 75:
            print("👍 Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("📚 Keep studying! Focus on curvature and geodesics.")
        else:
            print("🔄 Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║      Riemannian Geometry - Practice Quiz                 ║
║      Lesson 7: Curved Spaces and Geodesics               ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Metric tensor and line elements
• Christoffel symbols
• Covariant derivatives
• Riemann curvature tensor
• Ricci tensor and scalar curvature
• Geodesics
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Line element on sphere
    def check_q1(answer):
        ans = answer.lower()
        if 'r²dθ²' in ans.replace(' ', '') and 'r²sin²θdφ²' in ans.replace(' ', ''):
            return True, "Correct! ds² = R²dθ² + R²sin²θ dφ² is the line element for a 2-sphere of radius R."
        else:
            return False, "Incorrect. For a sphere: ds² = R²dθ² + R²sin²θ dφ². The metric is diag(R², R²sin²θ)."

    quiz.question(
        "What is the line element for a 2-sphere of radius R in (θ, φ) coordinates? (Write in form: ds² = ...)",
        check_q1,
        hints=[
            "The metric tensor is g = diag(R², R²sin²θ)",
            "Line element: ds² = g_μν dx^μ dx^ν",
            "ds² = R²dθ² + R²sin²θ dφ²"
        ]
    )

    # Question 2: Metric component at pole
    def check_q2(answer):
        try:
            ans = float(answer)
            correct = 0.0
            if np.isclose(ans, correct):
                return True, f"Correct! g_φφ = R²sin²θ = 0 at θ=0 (north pole). Circumference → 0 at pole!"
            else:
                return False, f"Incorrect. g_φφ = R²sin²(0) = 0 at the north pole."
        except:
            return False, "Please enter a number"

    quiz.question(
        "For a 2-sphere of radius R, what is g_φφ at the north pole (θ = 0)?",
        check_q2,
        hints=[
            "The metric component is g_φφ = R²sin²θ",
            "Evaluate at θ = 0",
            "What is sin(0)?"
        ]
    )

    # Question 3: Christoffel symbols in flat space
    def check_q3(answer):
        ans = answer.lower()
        if 'zero' in ans or '0' in ans or 'all' in ans:
            return True, "Correct! In flat space with Cartesian coords, g_μν = δ_μν (constant), so ∂g = 0, thus Γ = 0."
        else:
            return False, "Incorrect. All Christoffel symbols vanish in flat space with Cartesian coordinates because the metric is constant."

    quiz.question(
        "In flat Euclidean space with Cartesian coordinates, what are all the Christoffel symbols?",
        check_q3,
        hints=[
            "Cartesian coords: g_μν = δ_μν (constant)",
            "Christoffel symbols depend on ∂_μ g_νλ",
            "What's the derivative of a constant?"
        ]
    )

    # Question 4: Computing Christoffel symbol
    def check_q4(answer):
        ans = answer.lower()
        if '-r' in ans.replace(' ', '') or 'r' in ans:
            try:
                # Try to extract number/sign
                if '-' in ans:
                    return True, "Correct! Γ^r_θθ = -r for the polar metric g = diag(1, r²)."
                else:
                    return False, "Close, but check the sign. Γ^r_θθ = -r (negative!)."
            except:
                return True, "Correct! Γ^r_θθ = -r"
        else:
            return False, "Incorrect. For polar coords, Γ^r_θθ = -r. This represents the centrifugal effect."

    quiz.question(
        "Compute Γ^r_θθ for the 2D polar metric g = diag(1, r²). (Give the symbolic answer)",
        check_q4,
        hints=[
            "Use Γ^λ_μν = (1/2)g^λσ(∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)",
            "For Γ^r_θθ: λ=r, μ=θ, ν=θ",
            "g_θθ = r², and ∂g_θθ/∂r = 2r"
        ]
    )

    # Question 5: Covariant derivative of metric
    def check_q5(answer):
        ans = answer.lower()
        if 'zero' in ans or '0' in ans:
            return True, "Correct! ∇_μ g_νλ = 0 (metric compatibility). This is a defining property of the Levi-Civita connection."
        else:
            return False, "Incorrect. The covariant derivative of the metric always vanishes: ∇_μ g_νλ = 0. This is metric compatibility."

    quiz.question(
        "What is ∇_μ g_νλ for the Levi-Civita connection?",
        check_q5,
        hints=[
            "The Levi-Civita connection is metric-compatible",
            "This means the metric is covariantly constant",
            "∇_μ g_νλ = ?"
        ]
    )

    # Question 6: Riemann tensor components
    def check_q6(answer):
        try:
            ans = int(answer)
            correct = 20
            if ans == correct:
                return True, f"Correct! In n=4 dimensions: n²(n²-1)/12 = 16×15/12 = {correct} independent components."
            else:
                return False, f"Incorrect. Use formula: n²(n²-1)/12 for n dimensions. In 4D: 4²×15/12 = {correct}."
        except:
            return False, "Please enter a number"

    quiz.question(
        "How many independent components does the Riemann tensor have in 4D spacetime?",
        check_q6,
        hints=[
            "Use the formula: n²(n²-1)/12 for n dimensions",
            "For n=4: 4² = 16, 4²-1 = 15",
            "16 × 15 / 12 = ?"
        ]
    )

    # Question 7: Meaning of zero Riemann tensor
    def check_q7(answer):
        ans = answer.lower()
        if 'flat' in ans or 'euclidean' in ans:
            return True, "Correct! R^ρ_σμν = 0 everywhere means the space is flat (locally Euclidean). No intrinsic curvature."
        else:
            return False, "Incorrect. R = 0 everywhere means the space is FLAT - we can find global Cartesian coordinates."

    quiz.question(
        "What does R^ρ_σμν = 0 everywhere tell you about the space?",
        check_q7,
        hints=[
            "Think about parallel transport and geodesic deviation",
            "What kind of space has no curvature?",
            "Hint: rhymes with 'mat'"
        ]
    )

    # Question 8: Ricci scalar for sphere
    def check_q8(answer):
        ans = answer.lower().replace(' ', '')
        if '2/r²' in ans or '2/r^2' in ans:
            return True, "Correct! For a 2-sphere: R = 2/R². For unit sphere (R=1), R = 2."
        else:
            return False, "Incorrect. The Ricci scalar for a 2-sphere of radius R is R = 2/R². Constant positive curvature!"

    quiz.question(
        "What is the Ricci scalar R for a 2-sphere of radius R?",
        check_q8,
        hints=[
            "The Ricci scalar is constant on the sphere",
            "For unit sphere (R=1), R = 2",
            "Answer has form: 2/R²"
        ]
    )

    # Question 9: Geodesics on sphere
    def check_q9(answer):
        ans = answer.lower()
        if 'n' in ans or 'only equator' in ans or 'just equator' in ans:
            return True, "Correct! Only the equator is a geodesic among latitude circles. Other latitudes have geodesic curvature."
        else:
            return False, "Incorrect. Only great circles are geodesics on a sphere. Latitude circles (except equator) are NOT geodesics."

    quiz.question(
        "On a sphere, are latitude circles (except the equator) geodesics? (yes/no)",
        check_q9,
        hints=[
            "Geodesics satisfy ∇_u u = 0",
            "Great circles are geodesics",
            "Walking along a latitude requires 'turning'"
        ]
    )

    # Question 10: Geodesic equation in flat space
    def check_q10(answer):
        ans = answer.lower().replace(' ', '')
        if 'd²x/dt²=0' in ans or 'x=at+b' in ans or 'straight' in ans or 'constant' in ans:
            return True, "Correct! In flat space (Cartesian), Γ = 0, so d²x^λ/dt² = 0. This gives straight lines!"
        else:
            return False, "Incorrect. In flat space with Cartesian coords, all Γ^λ_μν = 0, so the geodesic equation becomes d²x^λ/dt² = 0 (straight line with constant velocity)."

    quiz.question(
        "What is the geodesic equation in flat space with Cartesian coordinates?",
        check_q10,
        hints=[
            "Geodesic equation: d²x^λ/dt² + Γ^λ_μν dx^μ/dt dx^ν/dt = 0",
            "In flat Cartesian space, Γ^λ_μν = 0",
            "What's left?"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\n📚 Review the lesson material in LESSON.md for any topics you missed.")
    print("🔬 Practice computing Christoffel symbols and Riemann tensor components.\n")


if __name__ == "__main__":
    main()
