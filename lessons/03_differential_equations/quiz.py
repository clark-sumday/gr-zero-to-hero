#!/usr/bin/env python3
"""
Differential Equations Practice Quiz
Tests understanding of ODEs, systems, phase portraits, and numerical methods
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
            print("🌟 Excellent! You have a strong grasp of differential equations.")
        elif percentage >= 75:
            print("👍 Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("📚 Keep studying! Focus on the fundamentals.")
        else:
            print("🔄 Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   Differential Equations - Practice Quiz                ║
║   Lesson 3: ODEs, Systems, Phase Portraits, Numerics    ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• First-order and second-order ODEs
• Systems of differential equations
• Phase portraits and stability
• Numerical methods (Euler, RK4)
• Boundary value problems
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Exponential decay solution
    def check_q1(answer):
        ans = answer.lower().replace(' ', '')
        # Looking for e^(-kt) or exp(-kt) or exponential
        is_exponential = 'e^(-' in ans or 'exp(-' in ans or 'e^-' in ans
        if is_exponential:
            return True, "Correct! dy/dt = -ky has solution y(t) = y₀e^(-kt)"
        else:
            return False, "Incorrect. The solution is y(t) = y₀e^(-kt) (exponential decay)"

    quiz.question(
        "What is the general solution form to dy/dt = -ky? (Use e^ notation)",
        check_q1,
        hints=[
            "This is a separable ODE: dy/y = -k dt",
            "Integrate both sides: ln|y| = -kt + C"
        ]
    )

    # Question 2: Characteristic equation
    def check_q2(answer):
        try:
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            parts = sorted(parts)
            # r² - 3r + 2 = 0 gives (r-1)(r-2) = 0, so r = 1, 2
            correct = sorted([1.0, 2.0])
            if np.allclose(parts, correct):
                return True, f"Correct! Roots are r = 1 and r = 2"
            else:
                return False, f"The correct roots are r = 1 and r = 2"
        except:
            return False, "Please enter two numbers separated by comma: r₁, r₂"

    quiz.question(
        "For y'' - 3y' + 2y = 0, find the roots of the characteristic equation.\nFormat: r₁, r₂",
        check_q2,
        hints=[
            "Characteristic equation: r² - 3r + 2 = 0",
            "Factor: (r - 1)(r - 2) = 0"
        ]
    )

    # Question 3: Damping classification
    def check_q3(answer):
        ans = answer.lower()
        is_underdamped = 'under' in ans
        if is_underdamped:
            return True, "Correct! When γ < ω, the system is underdamped (oscillates with decay)."
        else:
            return False, "Incorrect. γ < ω gives underdamped motion (oscillation with decay)."

    quiz.question(
        "For damped harmonic oscillator y'' + 2γy' + ω²y = 0, if γ < ω, is it:\n(a) underdamped, (b) critically damped, or (c) overdamped?",
        check_q3,
        hints=[
            "Damping ratio ζ = γ/ω",
            "ζ < 1 means underdamped"
        ]
    )

    # Question 4: System eigenvalues
    def check_q4(answer):
        try:
            # Accept i, j, I, J for imaginary unit
            ans_clean = answer.lower().replace('i', 'j').replace(' ', '')
            # Parse ±i or ±j
            if ans_clean in ['+j,-j', '-j,+j', 'j,-j', '-j,j', '±j', '±i']:
                return True, "Correct! Eigenvalues are λ = ±i (pure imaginary)"
            else:
                # Try complex number parsing
                parts = ans_clean.split(',')
                if len(parts) == 2:
                    try:
                        vals = [complex(p.strip()) for p in parts]
                        if np.allclose(sorted([v.imag for v in vals]), [-1, 1]) and \
                           np.allclose([v.real for v in vals], [0, 0]):
                            return True, "Correct! Eigenvalues are λ = ±i"
                    except:
                        pass
                return False, "The correct eigenvalues are λ = ±i"
        except:
            return False, "Please enter the eigenvalues (e.g., +i, -i)"

    quiz.question(
        "For the system d𝐫/dt = A𝐫 where A = [[0, 1], [-1, 0]], what are the eigenvalues?",
        check_q4,
        hints=[
            "det(A - λI) = 0",
            "λ² + 1 = 0"
        ]
    )

    # Question 5: Phase portrait classification
    def check_q5(answer):
        ans = answer.lower()
        is_center = 'center' in ans or 'centre' in ans
        if is_center:
            return True, "Correct! Pure imaginary eigenvalues give a center (circular orbits)."
        else:
            return False, "Incorrect. λ = ±i (pure imaginary) corresponds to a center with closed orbits."

    quiz.question(
        "What type of equilibrium does the system from Question 4 have at the origin?\n(node, saddle, spiral, center)",
        check_q5,
        hints=[
            "Pure imaginary eigenvalues λ = ±i",
            "No real part means no growth or decay"
        ]
    )

    # Question 6: Stability
    def check_q6(answer):
        try:
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            parts = sorted(parts)
            # λ² + 4λ + 3 = 0 gives (λ+1)(λ+3) = 0, so λ = -1, -3
            correct = sorted([-3.0, -1.0])
            if np.allclose(parts, correct):
                return True, f"Correct! λ = -1 and λ = -3 (both negative → stable node)"
            else:
                return False, f"The correct eigenvalues are λ = -1 and λ = -3"
        except:
            return False, "Please enter two numbers separated by comma"

    quiz.question(
        "For A = [[-2, 1], [1, -2]], find the eigenvalues and classify stability.\nFormat: λ₁, λ₂",
        check_q6,
        hints=[
            "Characteristic equation: (-2-λ)² - 1 = 0",
            "λ² + 4λ + 3 = 0"
        ]
    )

    # Question 7: Euler vs RK4 accuracy
    def check_q7(answer):
        ans = answer.lower()
        is_rk4 = 'rk4' in ans or 'runge' in ans
        if is_rk4:
            return True, "Correct! RK4 is O(h⁴) accurate, much better than Euler's O(h)."
        else:
            return False, "Incorrect. RK4 is more accurate (O(h⁴) vs O(h) for Euler)."

    quiz.question(
        "Which numerical method is more accurate: Euler or RK4?",
        check_q7,
        hints=[
            "Euler is O(h) - first order",
            "RK4 is O(h⁴) - fourth order"
        ]
    )

    # Question 8: Order of accuracy
    def check_q8(answer):
        try:
            ans = int(answer)
            if ans == 4:
                return True, "Correct! RK4 is fourth-order accurate: error ~ O(h⁴)"
            else:
                return False, "Incorrect. RK4 is fourth-order: O(h⁴)"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the order of accuracy of the RK4 (Runge-Kutta 4th order) method?\n(Enter a number: 1, 2, 3, 4, etc.)",
        check_q8,
        hints=[
            "The name gives it away!",
            "RK4 = Runge-Kutta 4th order"
        ]
    )

    # Question 9: Boundary value problem
    def check_q9(answer):
        ans = answer.lower().replace(' ', '')
        # Looking for y = x^3 or y=x³
        is_cubic = 'x^3' in ans or 'x³' in ans or 'x**3' in ans
        if is_cubic:
            return True, "Correct! y(x) = x³ satisfies y'' = 6x with y(0) = 0, y(1) = 1"
        else:
            return False, "Incorrect. The solution is y(x) = x³"

    quiz.question(
        "Solve the BVP: y'' = 6x with y(0) = 0 and y(1) = 1",
        check_q9,
        hints=[
            "Integrate twice: y'' = 6x → y' = 3x² + C₁ → y = x³ + C₁x + C₂",
            "Apply boundary conditions to find C₁ and C₂"
        ]
    )

    # Question 10: Predator-prey behavior
    def check_q10(answer):
        ans = answer.lower()
        is_periodic = 'period' in ans or 'oscillat' in ans or 'cycl' in ans
        if is_periodic:
            return True, "Correct! Lotka-Volterra exhibits periodic oscillations (closed orbits in phase space)."
        else:
            return False, "Incorrect. The Lotka-Volterra model shows periodic/oscillatory behavior."

    quiz.question(
        "What type of behavior does the Lotka-Volterra predator-prey model exhibit?\n(periodic, exponential, stable, chaotic)",
        check_q10,
        hints=[
            "Look at the phase portrait",
            "Predator and prey populations cycle"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\n📚 Review the lesson material in LESSON.md for any topics you missed.")
    print("🔬 Try the example scripts in the examples/ directory to explore more.\n")


if __name__ == "__main__":
    main()
