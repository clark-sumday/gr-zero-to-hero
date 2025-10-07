#!/usr/bin/env python3
"""
Multivariable Calculus Practice Quiz
Tests understanding of gradients, directional derivatives, divergence/curl, and optimization
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
            print("🌟 Excellent! You have a strong grasp of multivariable calculus.")
        elif percentage >= 75:
            print("👍 Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("📚 Keep studying! Focus on the fundamentals.")
        else:
            print("🔄 Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   Multivariable Calculus - Practice Quiz                ║
║   Lesson 2: Gradients, Optimization, Div/Curl           ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Partial derivatives and gradients
• Directional derivatives
• Divergence and curl
• Optimization and critical points
• Lagrange multipliers
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Partial derivatives
    def check_q1(answer):
        try:
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            # f(x,y) = x^2 + y^2, partial derivatives are 2x, 2y
            # At (1,2): (2, 4)
            correct = np.array([2.0, 4.0])
            if np.allclose(parts, correct):
                return True, f"Correct! ∂f/∂x = 2x = 2, ∂f/∂y = 2y = 4"
            else:
                return False, f"The correct answer is ∂f/∂x = 2, ∂f/∂y = 4"
        except:
            return False, "Please enter your answer in the format: (∂f/∂x, ∂f/∂y)"

    quiz.question(
        "For f(x,y) = x² + y², compute the partial derivatives at point (1, 2).\nFormat: (∂f/∂x, ∂f/∂y)",
        check_q1,
        hints=[
            "∂f/∂x = 2x (treat y as constant)",
            "∂f/∂y = 2y (treat x as constant)"
        ]
    )

    # Question 2: Gradient vector
    def check_q2(answer):
        try:
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            # ∇f = [y, x] at (2,3) = [3, 2]
            correct = np.array([3.0, 2.0])
            if np.allclose(parts, correct):
                return True, f"Correct! ∇f = [y, x] = [3, 2] at (2,3)"
            else:
                return False, f"The correct answer is ∇f = [3, 2]"
        except:
            return False, "Please enter your answer in the format: (a, b)"

    quiz.question(
        "For f(x,y) = xy, what is the gradient ∇f at point (2, 3)?",
        check_q2,
        hints=[
            "∇f = [∂f/∂x, ∂f/∂y]",
            "∂f/∂x = y and ∂f/∂y = x"
        ]
    )

    # Question 3: Direction of steepest ascent
    def check_q3(answer):
        ans = answer.lower()
        is_gradient = 'gradient' in ans or 'nabla' in ans or '∇f' in ans
        if is_gradient:
            return True, "Correct! The gradient ∇f points in the direction of steepest ascent."
        else:
            return False, "Incorrect. The gradient ∇f points in the direction of steepest ascent."

    quiz.question(
        "In what direction does the gradient ∇f point?",
        check_q3,
        hints=[
            "Think about what the gradient measures",
            "It points in the direction of maximum increase"
        ]
    )

    # Question 4: Directional derivative
    def check_q4(answer):
        try:
            ans = float(answer)
            # ∇f = [3, 2], û = [0.6, 0.8], D_u f = ∇f · û = 3(0.6) + 2(0.8) = 3.4
            correct = 3.4
            if np.isclose(ans, correct):
                return True, f"Correct! D_u f = ∇f · û = 3(0.6) + 2(0.8) = 3.4"
            else:
                return False, f"The correct answer is 3.4"
        except:
            return False, "Please enter a number"

    quiz.question(
        "For f(x,y) = xy at point (2,3), find the directional derivative in the direction u = [3, 4].\n(Hint: First normalize u)",
        check_q4,
        hints=[
            "∇f = [3, 2] at (2,3)",
            "û = [3,4]/5 = [0.6, 0.8]",
            "D_u f = ∇f · û"
        ]
    )

    # Question 5: Perpendicular to level curves
    def check_q5(answer):
        ans = answer.lower()
        is_yes = 'y' in ans or 'perpendicular' in ans or 'orthogonal' in ans
        if is_yes:
            return True, "Correct! ∇f is perpendicular to level curves f = constant."
        else:
            return False, "Incorrect. The gradient ∇f is always perpendicular to level curves."

    quiz.question(
        "Is the gradient ∇f perpendicular to level curves of f? (yes/no)",
        check_q5,
        hints=[
            "Level curves are where f = constant",
            "The gradient points in direction of maximum increase"
        ]
    )

    # Question 6: Divergence
    def check_q6(answer):
        try:
            ans = float(answer)
            # F = [x, y, z], div F = ∂x/∂x + ∂y/∂y + ∂z/∂z = 1 + 1 + 1 = 3
            correct = 3.0
            if np.isclose(ans, correct):
                return True, f"Correct! div F = ∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z = 1 + 1 + 1 = 3"
            else:
                return False, f"The correct answer is 3"
        except:
            return False, "Please enter a number"

    quiz.question(
        "For the vector field F = [x, y, z], compute div F",
        check_q6,
        hints=[
            "div F = ∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z",
            "∂x/∂x = 1, ∂y/∂y = 1, ∂z/∂z = 1"
        ]
    )

    # Question 7: Curl
    def check_q7(answer):
        try:
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            # F = [y, -x, 0], curl F = [0, 0, -2]
            correct = np.array([0.0, 0.0, -2.0])
            if np.allclose(parts, correct):
                return True, f"Correct! curl F = [0, 0, -2] - rotation around z-axis"
            else:
                return False, f"The correct answer is curl F = [0, 0, -2]"
        except:
            return False, "Please enter your answer in the format: (x, y, z)"

    quiz.question(
        "For the rotational field F = [y, -x, 0], compute curl F",
        check_q7,
        hints=[
            "curl F = [∂F₃/∂y - ∂F₂/∂z, ∂F₁/∂z - ∂F₃/∂x, ∂F₂/∂x - ∂F₁/∂y]",
            "∂(-x)/∂x = -1, ∂y/∂y = 1"
        ]
    )

    # Question 8: Critical point classification
    def check_q8(answer):
        ans = answer.lower()
        is_saddle = 'saddle' in ans
        if is_saddle:
            return True, "Correct! With D < 0, this is a saddle point."
        else:
            return False, "Incorrect. When D = f_xx·f_yy - f_xy² < 0, the critical point is a saddle point."

    quiz.question(
        "For f(x,y) = x² - y², the critical point at (0,0) has Hessian det(H) = -4 < 0.\nWhat type of critical point is this?",
        check_q8,
        hints=[
            "Check the sign of D = f_xx·f_yy - f_xy²",
            "D < 0 indicates a saddle point"
        ]
    )

    # Question 9: Finding critical points
    def check_q9(answer):
        try:
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            # ∇f = [2x-4, 4y-4] = 0 gives x=2, y=1
            correct = np.array([2.0, 1.0])
            if np.allclose(parts, correct):
                return True, f"Correct! Critical point at (2, 1) where ∇f = 0"
            else:
                return False, f"The correct answer is (2, 1)"
        except:
            return False, "Please enter your answer in the format: (x, y)"

    quiz.question(
        "Find the critical point of f(x,y) = x² + 2y² - 4x - 4y + 5",
        check_q9,
        hints=[
            "Solve ∇f = 0: [2x-4, 4y-4] = [0, 0]",
            "2x - 4 = 0 and 4y - 4 = 0"
        ]
    )

    # Question 10: Lagrange multipliers
    def check_q10(answer):
        try:
            ans = float(answer)
            # Maximum of x + 2y on x² + y² = 5 occurs at (1, 2) with f = 5
            correct = 5.0
            if np.isclose(ans, correct):
                return True, f"Correct! Maximum value is 5 at point (1, 2)"
            else:
                return False, f"The correct answer is 5"
        except:
            return False, "Please enter a number"

    quiz.question(
        "Using Lagrange multipliers, what is the maximum value of f(x,y) = x + 2y subject to x² + y² = 5?",
        check_q10,
        hints=[
            "Set up ∇f = λ∇g where g = x² + y² - 5",
            "[1, 2] = λ[2x, 2y] gives y = 2x",
            "Substitute into constraint: x² + (2x)² = 5"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\n📚 Review the lesson material in LESSON.md for any topics you missed.")
    print("🔬 Try the example scripts in the examples/ directory to explore more.\n")


if __name__ == "__main__":
    main()
