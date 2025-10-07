#!/usr/bin/env python3
"""
Manifolds Practice Quiz
Tests understanding of manifolds, charts, tangent spaces, and differential geometry
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
            print("🌟 Excellent! You have a strong grasp of manifold theory.")
        elif percentage >= 75:
            print("👍 Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("📚 Keep studying! Focus on the fundamentals.")
        else:
            print("🔄 Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   Manifolds - Practice Quiz                             ║
║   Lesson 5: Charts, Tangent Spaces, Vector Fields       ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Manifolds and their properties
• Charts and atlases
• Differentiable manifolds
• Tangent vectors and tangent spaces
• Vector fields
• Cotangent space and one-forms
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Torus dimension
    def check_q1(answer):
        try:
            ans = int(answer)
            if ans == 2:
                return True, "Correct! The torus is a 2-dimensional manifold (2 angles: u, v)"
            else:
                return False, "The correct answer is 2"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the dimension of the torus (donut surface) as a manifold?",
        check_q1,
        hints=[
            "How many coordinates needed to specify a point?",
            "Think of two angles: around the tube and around the main circle"
        ]
    )

    # Question 2: Minimum charts for sphere
    def check_q2(answer):
        try:
            ans = int(answer)
            if ans == 2:
                return True, "Correct! You need at least 2 charts to cover a sphere (e.g., stereographic from N and S poles)"
            else:
                return False, "The correct answer is 2"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the minimum number of charts needed to cover a sphere S²?",
        check_q2,
        hints=[
            "Can you make a single flat map of Earth without distortion?",
            "Stereographic projection from north covers all but north pole"
        ]
    )

    # Question 3: Circle as manifold
    def check_q3(answer):
        try:
            ans = int(answer)
            if ans == 1:
                return True, "Correct! The circle S¹ is 1-dimensional (parametrized by angle θ)"
            else:
                return False, "The correct answer is 1"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the dimension of the circle S¹ as a manifold?",
        check_q3,
        hints=[
            "How many coordinates to specify a point on a circle?",
            "Think of angle θ"
        ]
    )

    # Question 4: Tangent space dimension
    def check_q4(answer):
        try:
            ans = int(answer)
            if ans == 2:
                return True, "Correct! T_p S² is 2-dimensional (same as manifold dimension)"
            else:
                return False, "The correct answer is 2"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the dimension of the tangent space T_p S² at a point p on the 2-sphere?",
        check_q4,
        hints=[
            "Tangent space dimension = manifold dimension",
            "S² is 2-dimensional"
        ]
    )

    # Question 5: Tangent vectors as operators
    def check_q5(answer):
        ans = answer.lower()
        is_derivative = 'derivative' in ans or 'directional' in ans or 'operator' in ans
        if is_derivative:
            return True, "Correct! Tangent vectors are directional derivative operators"
        else:
            return False, "Incorrect. Tangent vectors ARE directional derivative operators"

    quiz.question(
        "What are tangent vectors interpreted as on a manifold?\n(arrows, derivatives, points, functions)",
        check_q5,
        hints=[
            "They act on functions to give rates of change",
            "Think: v[f] = directional derivative"
        ]
    )

    # Question 6: Coordinate basis
    def check_q6(answer):
        ans = answer.lower().replace(' ', '').replace('^', '')
        # Looking for ∂/∂x or d/dx notation
        is_partial = '∂/' in ans or 'd/d' in ans or 'partial' in ans
        if is_partial:
            return True, "Correct! Basis vectors are ∂/∂x^i (partial derivatives)"
        else:
            return False, "Incorrect. The coordinate basis is {∂/∂x¹, ∂/∂x², ..., ∂/∂xⁿ}"

    quiz.question(
        "In coordinates (x¹, x², ..., xⁿ), what forms the basis for the tangent space?\n(Answer with ∂ or d/d notation)",
        check_q6,
        hints=[
            "Tangent vectors are derivative operators",
            "Partial derivatives with respect to coordinates"
        ]
    )

    # Question 7: Vector field definition
    def check_q7(answer):
        ans = answer.lower()
        is_assignment = 'assign' in ans or 'each point' in ans or 'every point' in ans
        if is_assignment:
            return True, "Correct! A vector field assigns a tangent vector to each point on M"
        else:
            return False, "Incorrect. A vector field assigns a tangent vector to each point on the manifold"

    quiz.question(
        "What is a vector field on a manifold M?",
        check_q7,
        hints=[
            "Think: wind velocity at each point on Earth",
            "A smooth assignment of vectors to points"
        ]
    )

    # Question 8: One-form evaluation
    def check_q8(answer):
        try:
            ans = float(answer)
            # ω(v) = (2 dθ)(3 ∂/∂θ) = 2 × 3 = 6
            if np.isclose(ans, 6):
                return True, "Correct! ω(v) = 2 × 3 × dθ(∂/∂θ) = 6"
            else:
                return False, "The correct answer is 6"
        except:
            return False, "Please enter a number"

    quiz.question(
        "If v = 3 ∂/∂θ is a tangent vector and ω = 2 dθ is a one-form, what is ω(v)?",
        check_q8,
        hints=[
            "One-forms eat vectors and output numbers",
            "dθ(∂/∂θ) = 1 (Kronecker delta)"
        ]
    )

    # Question 9: Differential of function
    def check_q9(answer):
        ans = answer.lower().replace(' ', '')
        # Looking for 2x dx + 2y dy
        has_x = '2x' in ans and 'dx' in ans
        has_y = '2y' in ans and 'dy' in ans
        if has_x and has_y:
            return True, "Correct! df = 2x dx + 2y dy"
        else:
            return False, "The correct answer is df = 2x dx + 2y dy"

    quiz.question(
        "For f(x,y) = x² + y², what is the differential df (as a one-form)?",
        check_q9,
        hints=[
            "df = (∂f/∂x) dx + (∂f/∂y) dy",
            "Compute partial derivatives"
        ]
    )

    # Question 10: Configuration space dimension
    def check_q10(answer):
        try:
            ans = int(answer)
            # 3 particles × 3 dimensions = 9
            if ans == 9:
                return True, "Correct! 3 particles in 3D space → 3×3 = 9 dimensional configuration space"
            else:
                return False, "The correct answer is 9 (3 particles × 3 coordinates each)"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the dimension of the configuration space for 3 particles moving freely in 3D space?",
        check_q10,
        hints=[
            "Each particle needs 3 coordinates (x, y, z)",
            "Total = number of particles × 3"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\n📚 Review the lesson material in LESSON.md for any topics you missed.")
    print("🔬 Try the example scripts in the examples/ directory to explore more.\n")


if __name__ == "__main__":
    main()
