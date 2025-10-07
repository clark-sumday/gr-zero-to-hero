#!/usr/bin/env python3
"""
Single-Variable Calculus Practice Quiz
Tests understanding of limits, derivatives, chain rule, integration, FTC, and exponentials
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
            print("🌟 Excellent! You have a strong grasp of calculus.")
        elif percentage >= 75:
            print("👍 Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("📚 Keep studying! Focus on the fundamentals.")
        else:
            print("🔄 Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     Single-Variable Calculus - Practice Quiz            ║
║     Lesson 0: Limits, Derivatives, Integration          ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Limits and continuity
• Derivatives and rates of change
• Chain rule and differentiation rules
• Integration and area under curves
• Fundamental Theorem of Calculus
• Exponentials and logarithms
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Simple limit
    def check_q1(answer):
        try:
            ans = float(answer)
            if np.isclose(ans, 7):
                return True, "Correct! For continuous functions, lim[x→a] f(x) = f(a), so 2(3) + 1 = 7"
            else:
                return False, "The correct answer is 7. Just substitute x = 3 into 2x + 1."
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is lim[x→3] (2x + 1)?",
        check_q1,
        hints=[
            "For polynomial functions, you can substitute directly",
            "lim[x→3] (2x + 1) = 2(3) + 1"
        ]
    )

    # Question 2: Limit with indeterminate form
    def check_q2(answer):
        try:
            ans = float(answer)
            if np.isclose(ans, 4):
                return True, "Correct! Factor: (x²-4)/(x-2) = (x+2)(x-2)/(x-2) = x+2, so limit = 2+2 = 4"
            else:
                return False, "The correct answer is 4. Factor the numerator: x²-4 = (x+2)(x-2)"
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is lim[x→2] (x² - 4)/(x - 2)?",
        check_q2,
        hints=[
            "This is 0/0 form. Factor the numerator.",
            "x² - 4 = (x+2)(x-2), so the (x-2) cancels"
        ]
    )

    # Question 3: Basic derivative
    def check_q3(answer):
        try:
            # Accept answers like "6x" or "6*x"
            answer_clean = answer.lower().replace('*', '').replace(' ', '')
            if '6x' in answer_clean or answer_clean == '6':
                return True, "Correct! Using power rule: d/dx(3x²) = 3·2x = 6x"
            else:
                return False, "The correct answer is 6x. Use the power rule: d/dx(ax^n) = nax^(n-1)"
        except:
            return False, "Please enter the derivative (e.g., '6x')"

    quiz.question(
        "Find f'(x) for f(x) = 3x²",
        check_q3,
        hints=[
            "Use the power rule: d/dx(x^n) = nx^(n-1)",
            "d/dx(3x²) = 3·2x^(2-1) = 6x"
        ]
    )

    # Question 4: Derivative at a point
    def check_q4(answer):
        try:
            ans = float(answer)
            # f(x) = x² - 4x + 1, f'(x) = 2x - 4, f'(3) = 2(3) - 4 = 2
            if np.isclose(ans, 2):
                return True, "Correct! f'(x) = 2x - 4, so f'(3) = 2(3) - 4 = 2"
            else:
                return False, "The correct answer is 2. First find f'(x) = 2x - 4, then evaluate at x = 3."
        except:
            return False, "Please enter a number"

    quiz.question(
        "If f(x) = x² - 4x + 1, what is f'(3)?",
        check_q4,
        hints=[
            "First find the derivative f'(x) using power rule",
            "f'(x) = 2x - 4, now substitute x = 3"
        ]
    )

    # Question 5: Chain rule
    def check_q5(answer):
        try:
            # Accept various formats
            answer_clean = answer.lower().replace('*', '').replace(' ', '').replace('^', '')
            # Looking for 6(2x+3)² or equivalent
            if '6' in answer_clean and '2x+3' in answer_clean and '2' in answer_clean:
                return True, "Correct! Chain rule: h'(x) = 3(2x+3)² · 2 = 6(2x+3)²"
            else:
                return False, "The correct answer is 6(2x+3)². Use chain rule: outer derivative × inner derivative"
        except:
            return False, "Please enter the derivative"

    quiz.question(
        "Find the derivative of f(x) = (2x + 3)³ using the chain rule",
        check_q5,
        hints=[
            "Chain rule: if h(x) = f(g(x)), then h'(x) = f'(g(x)) · g'(x)",
            "Outer: u³ has derivative 3u², Inner: 2x+3 has derivative 2"
        ]
    )

    # Question 6: Indefinite integral
    def check_q6(answer):
        try:
            # Accept answers like "x³ + x² + C" or "x^3 + x^2 + C"
            answer_clean = answer.lower().replace('^', '').replace(' ', '').replace('*', '')
            if ('x3' in answer_clean or 'x³' in answer_clean) and ('x2' in answer_clean or 'x²' in answer_clean):
                return True, "Correct! ∫(3x² + 2x)dx = x³ + x² + C"
            else:
                return False, "The correct answer is x³ + x² + C. Use ∫x^n dx = x^(n+1)/(n+1)"
        except:
            return False, "Please enter the antiderivative"

    quiz.question(
        "Find ∫(3x² + 2x)dx",
        check_q6,
        hints=[
            "Use the power rule backwards: ∫x^n dx = x^(n+1)/(n+1) + C",
            "∫3x² dx = x³ and ∫2x dx = x²"
        ]
    )

    # Question 7: Definite integral using FTC
    def check_q7(answer):
        try:
            ans = float(answer)
            # ∫[0,2] (x² + 1)dx = [x³/3 + x] from 0 to 2 = (8/3 + 2) - 0 = 14/3
            correct = 14.0/3.0
            if np.isclose(ans, correct, atol=0.1):
                return True, f"Correct! F(x) = x³/3 + x, so F(2) - F(0) = (8/3 + 2) = 14/3 ≈ {correct:.3f}"
            else:
                return False, f"The correct answer is 14/3 ≈ {correct:.3f}"
        except:
            return False, "Please enter a number (you can use decimal)"

    quiz.question(
        "Compute ∫[0,2] (x² + 1)dx using the Fundamental Theorem of Calculus",
        check_q7,
        hints=[
            "Find antiderivative F(x), then compute F(2) - F(0)",
            "F(x) = x³/3 + x"
        ]
    )

    # Question 8: Derivative of exponential
    def check_q8(answer):
        try:
            # Accept various formats for e^x(1+x) or (1+x)e^x
            answer_clean = answer.lower().replace('*', '').replace(' ', '').replace('^', '')
            if ('ex' in answer_clean or 'e' in answer_clean) and ('1+x' in answer_clean or 'x+1' in answer_clean):
                return True, "Correct! Using product rule: (x·e^x)' = 1·e^x + x·e^x = e^x(1+x)"
            else:
                return False, "The correct answer is e^x(1+x). Use product rule: (fg)' = f'g + fg'"
        except:
            return False, "Please enter the derivative"

    quiz.question(
        "Find d/dx[x·e^x]",
        check_q8,
        hints=[
            "Use product rule: (fg)' = f'g + fg'",
            "(x)' = 1 and (e^x)' = e^x"
        ]
    )

    # Question 9: Derivative of natural log
    def check_q9(answer):
        try:
            # Accept "2/x" or "2x^-1"
            answer_clean = answer.lower().replace('*', '').replace(' ', '').replace('^', '')
            if '2/x' in answer_clean or '2x-1' in answer_clean:
                return True, "Correct! ln(x²) = 2ln(x), so d/dx[2ln(x)] = 2/x"
            else:
                return False, "The correct answer is 2/x. Use chain rule or simplify first: ln(x²) = 2ln(x)"
        except:
            return False, "Please enter the derivative"

    quiz.question(
        "Find d/dx[ln(x²)]",
        check_q9,
        hints=[
            "Simplify first: ln(x²) = 2ln(x)",
            "d/dx[ln(x)] = 1/x, so d/dx[2ln(x)] = 2/x"
        ]
    )

    # Question 10: Area interpretation
    def check_q10(answer):
        try:
            ans = float(answer)
            # ∫[0,3] 2dx = 2x from 0 to 3 = 6
            # This is area of rectangle: width 3, height 2
            if np.isclose(ans, 6):
                return True, "Correct! ∫[0,3] 2dx = 2x|₀³ = 6. This is the area of a 3×2 rectangle."
            else:
                return False, "The correct answer is 6. This represents the area of a rectangle with width 3 and height 2."
        except:
            return False, "Please enter a number"

    quiz.question(
        "What is ∫[0,3] 2dx? (Think geometrically: area under f(x) = 2)",
        check_q10,
        hints=[
            "f(x) = 2 is a horizontal line at height 2",
            "The area under this line from 0 to 3 is a rectangle"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\n📚 Review the lesson material in LESSON.md for any topics you missed.")
    print("🔬 Try the example scripts in the examples/ directory to explore more.\n")


if __name__ == "__main__":
    main()
