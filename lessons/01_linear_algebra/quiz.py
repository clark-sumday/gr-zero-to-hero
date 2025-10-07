#!/usr/bin/env python3
"""
Linear Algebra Practice Quiz
Tests understanding of vectors, matrices, dot/cross products, and eigenvalues
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
            print("🌟 Excellent! You have a strong grasp of linear algebra.")
        elif percentage >= 75:
            print("👍 Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("📚 Keep studying! Focus on the fundamentals.")
        else:
            print("🔄 Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     Linear Algebra Foundations - Practice Quiz          ║
║     Lesson 1: Vectors, Matrices, Transformations        ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Vector operations and properties
• Linear independence and basis
• Dot and cross products
• Matrix operations and eigenvalues
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Vector addition
    def check_q1(answer):
        try:
            # Parse answer like "(6, -1, -1)" or "6, -1, -1"
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            u = np.array([2, -3, 1])
            v = np.array([4, 1, -2])
            correct = u + v
            if np.allclose(parts, correct):
                return True, f"Correct! u + v = {correct}"
            else:
                return False, f"The correct answer is u + v = {correct}"
        except:
            return False, "Please enter your answer in the format: (x, y, z)"

    quiz.question(
        "Given u = (2, -3, 1) and v = (4, 1, -2), compute u + v",
        check_q1,
        hints=[
            "Add corresponding components: (u₁+v₁, u₂+v₂, u₃+v₃)",
            "u + v = (2+4, -3+1, 1+(-2))"
        ]
    )

    # Question 2: Magnitude
    def check_q2(answer):
        try:
            ans = float(answer)
            u = np.array([2, -3, 1])
            mag = np.linalg.norm(u)
            if np.isclose(ans, mag):
                return True, f"Correct! |u| = √(4+9+1) = √14 ≈ {mag:.3f}"
            else:
                return False, f"The correct answer is |u| = √14 ≈ {mag:.3f}"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the magnitude |u| of vector u = (2, -3, 1)?",
        check_q2,
        hints=[
            "|u| = √(u₁² + u₂² + u₃²)",
            "|u| = √(4 + 9 + 1) = √14"
        ]
    )

    # Question 3: Linear independence
    def check_q3(answer):
        ans = answer.lower()
        is_yes = 'y' in ans or 'independent' in ans
        if not is_yes:
            return True, "Correct! v₂ = 2v₁, so they are linearly dependent."
        else:
            return False, "Incorrect. v₂ = (2, 4, 6) = 2(1, 2, 3) = 2v₁, so they're dependent."

    quiz.question(
        "Are vectors v₁ = (1, 2, 3) and v₂ = (2, 4, 6) linearly independent? (yes/no)",
        check_q3,
        hints=[
            "Check if one vector is a scalar multiple of the other",
            "Does v₂ = c·v₁ for some scalar c?"
        ]
    )

    # Question 4: Dot product
    def check_q4(answer):
        try:
            ans = float(answer)
            u = np.array([1, -2, 3])
            v = np.array([2, 1, -1])
            dot = np.dot(u, v)
            if np.isclose(ans, dot):
                return True, f"Correct! u·v = (1)(2) + (-2)(1) + (3)(-1) = 2 - 2 - 3 = {dot}"
            else:
                return False, f"The correct answer is {dot}"
        except:
            return False, "Please enter a number"

    quiz.question(
        "Given u = (1, -2, 3) and v = (2, 1, -1), compute u · v",
        check_q4,
        hints=[
            "u·v = u₁v₁ + u₂v₂ + u₃v₃",
            "u·v = (1)(2) + (-2)(1) + (3)(-1)"
        ]
    )

    # Question 5: Orthogonality
    def check_q5(answer):
        ans = answer.lower()
        is_yes = 'y' in ans or 'orthogonal' in ans or 'perpendicular' in ans
        if not is_yes:
            return True, "Correct! u·v = -3 ≠ 0, so they're not orthogonal."
        else:
            return False, "Incorrect. For vectors to be orthogonal, u·v must equal 0. Here u·v = -3."

    quiz.question(
        "From the previous question, are u and v orthogonal? (yes/no)",
        check_q5,
        hints=[
            "Vectors are orthogonal if their dot product equals 0",
            "Is u·v = 0?"
        ]
    )

    # Question 6: Cross product
    def check_q6(answer):
        try:
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            u = np.array([2, 1, 0])
            v = np.array([1, 0, 3])
            cross = np.cross(u, v)
            if np.allclose(parts, cross):
                return True, f"Correct! u × v = {cross}"
            else:
                return False, f"The correct answer is u × v = {cross}"
        except:
            return False, "Please enter your answer in the format: (x, y, z)"

    quiz.question(
        "Compute the cross product (2, 1, 0) × (1, 0, 3)",
        check_q6,
        hints=[
            "u × v = (u₂v₃ - u₃v₂, u₃v₁ - u₁v₃, u₁v₂ - u₂v₁)",
            "u × v = (1·3 - 0·0, 0·1 - 2·3, 2·0 - 1·1)"
        ]
    )

    # Question 7: Area from cross product
    def check_q7(answer):
        try:
            ans = float(answer)
            u = np.array([3, 0, 0])
            v = np.array([0, 4, 0])
            cross = np.cross(u, v)
            area = np.linalg.norm(cross)
            if np.isclose(ans, area):
                return True, f"Correct! Area = |u × v| = {area}"
            else:
                return False, f"The correct answer is {area}"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is the area of the parallelogram formed by u = (3, 0, 0) and v = (0, 4, 0)?",
        check_q7,
        hints=[
            "Area = |u × v|",
            "u × v = (0, 0, 12), so area = 12"
        ]
    )

    # Question 8: Matrix multiplication
    def check_q8(answer):
        try:
            # Accept answers like "[[5,5],[5,10]]" or "5,5,5,10"
            answer = answer.strip('[]()').replace('][', ',').replace('];[', ',')
            parts = [float(x.strip()) for x in answer.split(',')]
            A = np.array([[2, 1], [1, 3]])
            A_squared = A @ A
            correct = A_squared.flatten()
            if np.allclose(parts, correct):
                return True, f"Correct! A² = \n{A_squared}"
            else:
                return False, f"The correct answer is:\n{A_squared}"
        except:
            return False, "Please enter your answer in the format: [[a,b],[c,d]]"

    quiz.question(
        "Given matrix A = [[2, 1], [1, 3]], compute A² (A times itself)",
        check_q8,
        hints=[
            "A² = A × A. Use matrix multiplication rules.",
            "First row, first column: 2×2 + 1×1 = 5"
        ]
    )

    # Question 9: Determinant
    def check_q9(answer):
        try:
            ans = float(answer)
            A = np.array([[3, 1, 2], [0, 2, 1], [1, 0, 4]])
            det = np.linalg.det(A)
            if np.isclose(ans, det):
                return True, f"Correct! det(A) = {det:.0f}"
            else:
                return False, f"The correct answer is {det:.0f}"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is det([[3, 1, 2], [0, 2, 1], [1, 0, 4]])?",
        check_q9,
        hints=[
            "Use cofactor expansion along the second row (has a zero)",
            "det(A) = -0×(minor) + 2×(minor of [3,2],[1,4]) - 1×(minor of [3,1],[1,0])"
        ]
    )

    # Question 10: Eigenvalues
    def check_q10(answer):
        try:
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            parts = sorted(parts)
            A = np.array([[4, 1], [0, 3]])
            eigenvalues = sorted(np.linalg.eigvals(A).real)
            if np.allclose(parts, eigenvalues):
                return True, f"Correct! Eigenvalues are {eigenvalues[0]:.0f} and {eigenvalues[1]:.0f}"
            else:
                return False, f"The correct eigenvalues are {eigenvalues[0]:.0f} and {eigenvalues[1]:.0f}"
        except:
            return False, "Please enter two numbers separated by comma: λ₁, λ₂"

    quiz.question(
        "For the matrix [[4, 1], [0, 3]], what are the eigenvalues? (format: λ₁, λ₂)",
        check_q10,
        hints=[
            "This is an upper triangular matrix",
            "For triangular matrices, eigenvalues are the diagonal elements"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\n📚 Review the lesson material in LESSON.md for any topics you missed.")
    print("🔬 Try the example scripts in the examples/ directory to explore more.\n")


if __name__ == "__main__":
    main()
