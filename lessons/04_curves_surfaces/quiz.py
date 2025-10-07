#!/usr/bin/env python3
"""
Curves and Surfaces Practice Quiz
Tests understanding of parametric curves, curvature, torsion, and surface geometry
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
            print("🌟 Excellent! You have a strong grasp of curves and surfaces.")
        elif percentage >= 75:
            print("👍 Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("📚 Keep studying! Focus on the fundamentals.")
        else:
            print("🔄 Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   Curves and Surfaces - Practice Quiz                   ║
║   Lesson 4: Curvature, Torsion, Fundamental Forms       ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Parametric curves and velocity vectors
• Arc length and reparametrization
• Curvature and osculating circles
• Torsion and Frenet-Serret frame
• Parametric surfaces
• First and second fundamental forms
• Gaussian curvature
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Velocity vector
    def check_q1(answer):
        try:
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            # r(t) = (t, t², t³), r'(1) = (1, 2t, 3t²)|_{t=1} = (1, 2, 3)
            correct = np.array([1.0, 2.0, 3.0])
            if np.allclose(parts, correct):
                return True, f"Correct! r'(1) = (1, 2, 3)"
            else:
                return False, f"The correct answer is r'(1) = (1, 2, 3)"
        except:
            return False, "Please enter your answer in the format: (a, b, c)"

    quiz.question(
        "For r(t) = (t, t², t³), what is the velocity vector r'(t) at t = 1?",
        check_q1,
        hints=[
            "r'(t) = (dx/dt, dy/dt, dz/dt)",
            "Differentiate each component"
        ]
    )

    # Question 2: Arc length
    def check_q2(answer):
        try:
            ans = float(answer)
            # r(t) = (3t, 4t, 0), |r'(t)| = √(9+16) = 5, L = 5×1 = 5
            correct = 5.0
            if np.isclose(ans, correct):
                return True, f"Correct! Arc length L = 5 (speed = 5, time = 1)"
            else:
                return False, f"The correct answer is L = 5"
        except:
            return False, "Please enter a number"

    quiz.question(
        "A particle moves along r(t) = (3t, 4t, 0) from t=0 to t=1. What is the arc length?",
        check_q2,
        hints=[
            "r'(t) = (3, 4, 0)",
            "Speed = |r'(t)| = √(3² + 4²) = 5",
            "L = speed × time"
        ]
    )

    # Question 3: Curvature of circle
    def check_q3(answer):
        try:
            ans = float(answer)
            # κ = 1/R for circle of radius 3
            correct = 1.0/3.0
            if np.isclose(ans, correct, rtol=0.01):
                return True, f"Correct! κ = 1/R = 1/3 ≈ {correct:.3f}"
            else:
                return False, f"The correct answer is κ = 1/3 ≈ 0.333"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the curvature κ of a circle with radius R = 3?",
        check_q3,
        hints=[
            "For a circle, κ = 1/R",
            "Substitute R = 3"
        ]
    )

    # Question 4: Curvature of straight line
    def check_q4(answer):
        try:
            ans = float(answer)
            if np.isclose(ans, 0):
                return True, "Correct! A straight line has zero curvature (κ = 0)"
            else:
                return False, "The correct answer is κ = 0"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the curvature of a straight line?",
        check_q4,
        hints=[
            "Does a straight line bend?",
            "r''(t) = 0 for a line"
        ]
    )

    # Question 5: Torsion of planar curve
    def check_q5(answer):
        try:
            ans = float(answer)
            if np.isclose(ans, 0):
                return True, "Correct! Planar curves have zero torsion (τ = 0)"
            else:
                return False, "The correct answer is τ = 0"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the torsion τ of a planar curve (lying in the xy-plane)?",
        check_q5,
        hints=[
            "Does a planar curve twist out of its plane?",
            "Torsion measures twisting out of the osculating plane"
        ]
    )

    # Question 6: Frenet-Serret frame dimension
    def check_q6(answer):
        try:
            ans = int(answer)
            if ans == 3:
                return True, "Correct! The Frenet-Serret frame has 3 vectors: T, N, B"
            else:
                return False, "The correct answer is 3 (T, N, B)"
        except:
            return False, "Please enter a number"

    quiz.question(
        "How many vectors are in the Frenet-Serret frame? (T, N, B)",
        check_q6,
        hints=[
            "T = tangent, N = normal, B = binormal",
            "Count them!"
        ]
    )

    # Question 7: Surface dimension
    def check_q7(answer):
        try:
            ans = int(answer)
            if ans == 2:
                return True, "Correct! A sphere is a 2-dimensional manifold/surface"
            else:
                return False, "The correct answer is 2"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the dimension of the sphere S² as a manifold?",
        check_q7,
        hints=[
            "How many coordinates needed to specify a point on the sphere?",
            "Think of latitude and longitude"
        ]
    )

    # Question 8: First fundamental form for plane
    def check_q8(answer):
        ans = answer.lower().replace(' ', '')
        # Looking for identity matrix or E=1, F=0, G=1
        is_identity = ('1,0' in ans and '0,1' in ans) or \
                      ('e=1' in ans and 'f=0' in ans and 'g=1' in ans) or \
                      'identity' in ans
        if is_identity:
            return True, "Correct! For a plane: E=1, F=0, G=1 (identity matrix)"
        else:
            return False, "Incorrect. The first fundamental form is I = [[1, 0], [0, 1]] (identity)"

    quiz.question(
        "For a flat plane r(u,v) = (u, v, 0), what is the first fundamental form?\n(Answer: E, F, G values or 'identity')",
        check_q8,
        hints=[
            "Compute r_u = (1, 0, 0) and r_v = (0, 1, 0)",
            "E = r_u · r_u, F = r_u · r_v, G = r_v · r_v"
        ]
    )

    # Question 9: Gaussian curvature of sphere
    def check_q9(answer):
        try:
            ans = float(answer)
            # K = 1/R² for sphere of radius 2
            correct = 1.0/4.0
            if np.isclose(ans, correct, rtol=0.01):
                return True, f"Correct! K = 1/R² = 1/4 = {correct}"
            else:
                return False, f"The correct answer is K = 1/4 = 0.25"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the Gaussian curvature K of a sphere with radius R = 2?",
        check_q9,
        hints=[
            "For a sphere: K = 1/R²",
            "Substitute R = 2"
        ]
    )

    # Question 10: Gaussian curvature of plane
    def check_q10(answer):
        try:
            ans = float(answer)
            if np.isclose(ans, 0):
                return True, "Correct! A flat plane has K = 0 (zero curvature)"
            else:
                return False, "The correct answer is K = 0"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the Gaussian curvature of a flat plane?",
        check_q10,
        hints=[
            "Does a plane curve?",
            "All second derivatives r_uu, r_uv, r_vv are zero"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\n📚 Review the lesson material in LESSON.md for any topics you missed.")
    print("🔬 Try the example scripts in the examples/ directory to explore more.\n")


if __name__ == "__main__":
    main()
