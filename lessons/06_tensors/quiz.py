#!/usr/bin/env python3
"""
Tensors Practice Quiz
Tests understanding of tensor basics, Einstein summation, metric tensor, and index manipulation
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
            print("🌟 Excellent! You have mastered tensors.")
        elif percentage >= 75:
            print("👍 Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("📚 Keep studying! Focus on index notation and the metric.")
        else:
            print("🔄 Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║           Tensors - Practice Quiz                        ║
║           Lesson 6: Tensor Fundamentals                  ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Tensor definition and rank
• Einstein summation convention
• Contravariant and covariant indices
• Metric tensor and raising/lowering indices
• Tensor operations
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Tensor rank/components
    def check_q1(answer):
        try:
            ans = int(answer)
            correct = 64
            if ans == correct:
                return True, f"Correct! A (2,1)-tensor has 3 indices, each running 0-3 in 4D, so 4³ = 64 components."
            else:
                return False, f"Incorrect. In 4D, each index has 4 values. For 3 indices: 4³ = {correct} components."
        except:
            return False, "Please enter a number"

    quiz.question(
        "How many independent components does a (2,1)-tensor have in 4-dimensional spacetime?",
        check_q1,
        hints=[
            "Count total indices: 2 upper + 1 lower = 3 total",
            "Each index runs from 0 to 3 (four values in 4D)",
            "Number of components = 4³"
        ]
    )

    # Question 2: Einstein summation
    def check_q2(answer):
        ans = answer.lower()
        if 'scalar' in ans or 'contraction' in ans or '(0,0)' in ans:
            return True, "Correct! Both μ and ν are summed over (repeated indices), producing a scalar."
        else:
            return False, "Incorrect. T^μν g_μν sums over both μ and ν, contracting the (2,0)-tensor with the (0,2)-metric to give a scalar."

    quiz.question(
        "What does the expression T^μν g_μν represent? (T is a (2,0)-tensor, g is the metric)",
        check_q2,
        hints=[
            "Look at which indices are repeated",
            "Both μ and ν appear once up and once down",
            "This is a double contraction - what rank is the result?"
        ]
    )

    # Question 3: Contraction
    def check_q3(answer):
        ans = answer.lower()
        if 'scalar' in ans or '(0,0)' in ans or '0' in ans:
            return True, "Correct! Contracting T_μν removes both indices after raising one. Trace gives a scalar."
        else:
            return False, "Incorrect. Starting with (0,2)-tensor, raise one index to get (1,1), then contract to get a (0,0)-tensor (scalar)."

    quiz.question(
        "If T_μν is a (0,2)-tensor and we contract it to get S = T^μ_μ, what type of tensor is S?",
        check_q3,
        hints=[
            "Contraction reduces rank by 2",
            "Need to raise one index first using the metric",
            "The trace operation produces what?"
        ]
    )

    # Question 4: Minkowski metric index lowering
    def check_q4(answer):
        try:
            # Parse answer like "(-2, 1, 0, 0)" or "-2, 1, 0, 0"
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            correct = np.array([-2, 1, 0, 0])
            if np.allclose(parts, correct):
                return True, f"Correct! u_μ = η_μν u^ν. Time component flips sign: u_0 = -2, spatial stay same: u_i = 1, 0, 0."
            else:
                return False, f"The correct answer is u_μ = (-2, 1, 0, 0). Note the sign flip on the time component!"
        except:
            return False, "Please enter your answer in format: -2, 1, 0, 0"

    quiz.question(
        "In Minkowski spacetime with metric η_μν = diag(-1,1,1,1), if u^μ = (2, 1, 0, 0), what is u_μ?",
        check_q4,
        hints=[
            "Use u_μ = η_μν u^ν",
            "The metric is diagonal, so u_μ = η_μμ u^μ (no sum)",
            "η_00 = -1, η_11 = η_22 = η_33 = 1"
        ]
    )

    # Question 5: Trace computation
    def check_q5(answer):
        try:
            ans = float(answer)
            T = np.array([[2, 1], [3, 4]])
            trace = np.trace(T)
            if np.isclose(ans, trace):
                return True, f"Correct! Trace = T^μ_μ = T^0_0 + T^1_1 = 2 + 4 = {trace}"
            else:
                return False, f"Incorrect. Sum diagonal elements: 2 + 4 = {trace}"
        except:
            return False, "Please enter a number"

    quiz.question(
        "Compute the trace of the matrix [[2, 1], [3, 4]].",
        check_q5,
        hints=[
            "Trace = sum of diagonal elements",
            "T^μ_μ = T^0_0 + T^1_1"
        ]
    )

    # Question 6: Symmetrization
    def check_q6(answer):
        ans = answer.lower()
        if 'y' in ans or 'symmetric' in ans:
            return True, "Correct! T_μν = T_νμ (T_01 = T_10 = 2), so it's symmetric."
        else:
            return False, "Incorrect. Check if T_μν = T_νμ. Here T_01 = T_10 = 2, so the tensor is symmetric."

    quiz.question(
        "Is the tensor T_μν = [[1, 2], [2, 1]] symmetric? (yes/no)",
        check_q6,
        hints=[
            "A tensor is symmetric if T_μν = T_νμ",
            "Check if the matrix equals its transpose"
        ]
    )

    # Question 7: Raising index in Euclidean space
    def check_q7(answer):
        try:
            answer = answer.strip('()[]')
            parts = [float(x.strip()) for x in answer.split(',')]
            correct = np.array([3, 4, 0])
            if np.allclose(parts, correct):
                return True, f"Correct! In Euclidean space with δ_ij, raising/lowering doesn't change components: v^i = v_i."
            else:
                return False, f"Incorrect. In Euclidean space g_ij = δ_ij, so v^i = δ^ij v_j = v_i = (3, 4, 0)"
        except:
            return False, "Please enter answer in format: 3, 4, 0"

    quiz.question(
        "In Euclidean 3D space with g_ij = δ_ij, if v_i = (3, 4, 0), what is v^i?",
        check_q7,
        hints=[
            "Use v^i = g^ij v_j",
            "In Euclidean space, g^ij = δ^ij (identity)",
            "Raising/lowering with δ doesn't change components"
        ]
    )

    # Question 8: Norm computation
    def check_q8(answer):
        try:
            ans = float(answer)
            # v^μ = (1, 1, 0, 0), η_μν = diag(-1, 1, 1, 1)
            # v^μ v_μ = η_μν v^μ v^ν = -1 + 1 + 0 + 0 = 0
            correct = 0.0
            if np.isclose(ans, correct):
                return True, f"Correct! v^μ v_μ = -(1)² + (1)² + 0 + 0 = -1 + 1 = 0. This is a null/lightlike vector!"
            else:
                return False, f"Incorrect. Compute η_μν v^μ v^ν = -(1)² + (1)² = {correct}. This is lightlike!"
        except:
            return False, "Please enter a number"

    quiz.question(
        "For Minkowski metric η_μν = diag(-1,1,1,1) and v^μ = (1, 1, 0, 0), compute v^μ v_μ.",
        check_q8,
        hints=[
            "v^μ v_μ = η_μν v^μ v^ν",
            "Expand: -(v^0)² + (v^1)² + (v^2)² + (v^3)²",
            "Substitute values"
        ]
    )

    # Question 9: Metric properties
    def check_q9(answer):
        ans = answer.lower()
        if 'y' in ans or 'identity' in ans or 'kronecker' in ans or 'δ' in ans:
            return True, "Correct! g_μρ g^ρν = δ_μ^ν (Kronecker delta/identity). This is how we define the inverse metric."
        else:
            return False, "Incorrect. The metric and its inverse satisfy g_μρ g^ρν = δ_μ^ν (identity matrix)."

    quiz.question(
        "What does g_μρ g^ρν equal? (This is a fundamental property of the metric)",
        check_q9,
        hints=[
            "The inverse metric g^μν is defined so that...",
            "Think about matrix multiplication: A A^(-1) = ?",
            "Result is the Kronecker delta δ_μ^ν"
        ]
    )

    # Question 10: Physical tensor example
    def check_q10(answer):
        ans = answer.lower()
        if 'symmetric' in ans or '10' in ans or 'ten' in ans:
            return True, "Correct! Ricci tensor R_μν is symmetric, so in 4D it has 4×5/2 = 10 independent components."
        else:
            return False, "Incorrect. The Ricci tensor is symmetric (R_μν = R_νμ), so it has n(n+1)/2 = 4×5/2 = 10 independent components in 4D."

    quiz.question(
        "The Ricci tensor R_μν in General Relativity is symmetric. How many independent components does it have in 4D spacetime?",
        check_q10,
        hints=[
            "Symmetric means R_μν = R_νμ",
            "Count: diagonal elements (4) + upper triangular elements (?)",
            "Formula: n(n+1)/2 for symmetric n×n matrix"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\n📚 Review the lesson material in LESSON.md for any topics you missed.")
    print("🔬 Practice computing Christoffel symbols and working with tensors.\n")


if __name__ == "__main__":
    main()
