#!/usr/bin/env python3
"""
Classical Mechanics Practice Quiz
Tests understanding of Lagrangian/Hamiltonian mechanics and Noether's theorem
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
            print("🌟 Excellent! You've mastered Lagrangian mechanics.")
        elif percentage >= 75:
            print("👍 Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("📚 Keep studying! Focus on the action principle.")
        else:
            print("🔄 Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║      Classical Mechanics - Practice Quiz                 ║
║      Lesson 8: Lagrangian and Hamiltonian Mechanics      ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Lagrangian formulation (L = T - V)
• Principle of least action
• Euler-Lagrange equations
• Hamiltonian mechanics
• Noether's theorem
• Conservation laws
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Free particle path
    def check_q1(answer):
        ans = answer.lower()
        if 'straight' in ans or 'constant' in ans or 'line' in ans:
            return True, "Correct! For a free particle, L = (1/2)mv². The action is minimized by constant velocity (straight line)."
        else:
            return False, "Incorrect. Free particle Lagrangian is L = (1/2)mv². Minimizing ∫v²dt gives constant velocity = straight line."

    quiz.question(
        "For a free particle (V=0), what path minimizes the action?",
        check_q1,
        hints=[
            "Free particle: L = T = (1/2)mv²",
            "Minimize ∫v²dt with fixed endpoints",
            "What velocity profile minimizes this integral?"
        ]
    )

    # Question 2: Lagrangian for pendulum
    def check_q2(answer):
        ans = answer.lower().replace(' ', '')
        # Accept various forms
        if ('mr²θ' in ans or 'ml²θ' in ans) and ('cosθ' in ans or 'cos(θ)' in ans):
            return True, "Correct! L = (1/2)mR²θ̇² - mgR(1-cosθ), where T = (1/2)mR²θ̇² and V = mgR(1-cosθ)."
        else:
            return False, "Incorrect. For pendulum: L = (1/2)mR²θ̇² - mgR(1-cosθ). Kinetic in terms of angle, potential from height."

    quiz.question(
        "What is the Lagrangian for a simple pendulum of length R and mass m? (In terms of angle θ from vertical)",
        check_q2,
        hints=[
            "Use angle θ as generalized coordinate",
            "T = (1/2)m v² where v = Rθ̇",
            "V = mgh where h = R(1-cosθ) from lowest point"
        ]
    )

    # Question 3: Generalized momentum
    def check_q3(answer):
        ans = answer.lower()
        if 'constant' in ans or 'conserved' in ans:
            return True, "Correct! For cyclic coordinate q_i (where ∂L/∂q_i = 0), the conjugate momentum p_i = ∂L/∂q̇_i is conserved."
        else:
            return False, "Incorrect. When ∂L/∂q_i = 0 (cyclic coordinate), the momentum p_i = ∂L/∂q̇_i is CONSERVED (constant in time)."

    quiz.question(
        "What is the generalized momentum for a cyclic coordinate?",
        check_q3,
        hints=[
            "Cyclic coordinate: ∂L/∂q_i = 0",
            "Generalized momentum: p_i = ∂L/∂q̇_i",
            "From Euler-Lagrange: d/dt(p_i) = ∂L/∂q_i = ?"
        ]
    )

    # Question 4: Hamiltonian for particle in gravity
    def check_q4(answer):
        ans = answer.lower().replace(' ', '')
        # Accept p²/2m + mgx or similar
        if ('p²' in ans or 'p^2' in ans) and ('2m' in ans) and ('mgx' in ans or 'mg' in ans):
            return True, "Correct! H = p²/(2m) + mgx. This is T + V (total energy) for this system."
        else:
            return False, "Incorrect. L = (1/2)mẋ² - mgx. Find p = mẋ, then H = pẋ - L = p²/(2m) + mgx."

    quiz.question(
        "For Lagrangian L = (1/2)mẋ² - mgx, find the Hamiltonian H(x,p).",
        check_q4,
        hints=[
            "Find p = ∂L/∂ẋ = mẋ, so ẋ = p/m",
            "Hamiltonian: H = pẋ - L",
            "Substitute ẋ = p/m into H"
        ]
    )

    # Question 5: Hamilton's equations
    def check_q5(answer):
        ans = answer.lower()
        if 'zero' in ans or '0' in ans or 'conserved' in ans:
            return True, "Correct! If ∂H/∂t = 0, then dH/dt = ∂H/∂t = 0, so H is conserved (energy conservation)."
        else:
            return False, "Incorrect. From Hamilton's equations: dH/dt = ∂H/∂t. If H doesn't depend explicitly on time, dH/dt = 0 (conserved)."

    quiz.question(
        "In Hamiltonian mechanics, what is dH/dt if H doesn't explicitly depend on time?",
        check_q5,
        hints=[
            "Use Hamilton's equations to compute dH/dt",
            "dH/dt = ∂H/∂t + ... (other terms cancel)",
            "If ∂H/∂t = 0, then..."
        ]
    )

    # Question 6: Conservation from symmetry
    def check_q6(answer):
        ans = answer.lower()
        # Check for energy and x-momentum
        if ('energy' in ans or 'e' in ans) and ('momentum' in ans or 'p' in ans or 'x' in ans):
            return True, "Correct! Energy (time translation: ∂L/∂t=0) and x-momentum (x-translation: ∂L/∂x=0) are conserved."
        else:
            return False, "Incorrect. L = (1/2)m(ẋ²+ẏ²) - mgy. Time symmetry → Energy conserved. x-translation symmetry → p_x conserved. But ∂L/∂y ≠ 0 (gravity) → p_y NOT conserved."

    quiz.question(
        "For L = (1/2)m(ẋ² + ẏ²) - mgy, what quantities are conserved?",
        check_q6,
        hints=[
            "Check for time, x-translation, and y-translation symmetries",
            "∂L/∂t = ?",
            "∂L/∂x = ? and ∂L/∂y = ?"
        ]
    )

    # Question 7: Noether's theorem
    def check_q7(answer):
        ans = answer.lower()
        if 'symmetry' in ans and 'conservation' in ans:
            return True, "Correct! Noether's theorem: Every continuous symmetry of the action corresponds to a conservation law."
        else:
            return False, "Incorrect. Noether's theorem states: Every continuous SYMMETRY implies a CONSERVATION LAW (and vice versa)."

    quiz.question(
        "State Noether's theorem in one sentence.",
        check_q7,
        hints=[
            "It relates two concepts: symmetries and conservation laws",
            "Continuous symmetries of the action...",
            "...correspond to what?"
        ]
    )

    # Question 8: Time symmetry consequence
    def check_q8(answer):
        ans = answer.lower()
        if 'energy' in ans:
            return True, "Correct! Time translation symmetry (∂L/∂t = 0) implies energy conservation via Noether's theorem."
        else:
            return False, "Incorrect. Time translation symmetry → ENERGY is conserved. This is one of the key results of Noether's theorem."

    quiz.question(
        "What is conserved if a system has time translation symmetry (∂L/∂t = 0)?",
        check_q8,
        hints=[
            "Time translation means physics is the same at all times",
            "Apply Noether's theorem",
            "The conserved quantity is the Hamiltonian, which is usually..."
        ]
    )

    # Question 9: Relativistic particle Lagrangian
    def check_q9(answer):
        ans = answer.lower().replace(' ', '')
        if '-mc²' in ans and ('√' in ans or 'sqrt' in ans) and ('1-v²/c²' in ans or '1-v^2/c^2' in ans):
            return True, "Correct! L = -mc²√(1-v²/c²) is the relativistic Lagrangian for a free particle."
        else:
            return False, "Incorrect. For relativistic particle: L = -mc²√(1-v²/c²). This reduces to (1/2)mv² - mc² for v << c."

    quiz.question(
        "What is the Lagrangian for a relativistic free particle?",
        check_q9,
        hints=[
            "Action: S = -mc² ∫√(1-v²/c²) dt",
            "Lagrangian is the integrand",
            "L = -mc²√(1-v²/c²)"
        ]
    )

    # Question 10: Einstein-Hilbert action
    def check_q10(answer):
        ans = answer.lower().replace(' ', '')
        # Accept various forms
        if ('einstein' in ans or 'g_μν' in ans or 'g_mu' in ans or 'metric' in ans) and ('vary' in ans or 'variation' in ans or 'field' in ans):
            return True, "Correct! Vary the Einstein-Hilbert action S = ∫R√(-g)d⁴x with respect to metric g_μν to get G_μν = 8πGT_μν."
        else:
            return False, "Incorrect. In GR, we vary the Einstein-Hilbert action S = (1/16πG)∫R√(-g)d⁴x with respect to the metric g_μν to derive Einstein's field equations."

    quiz.question(
        "In General Relativity, what action do we vary to get Einstein's field equations?",
        check_q10,
        hints=[
            "Named after Einstein and Hilbert",
            "Involves the Ricci scalar R and metric determinant g",
            "S = ∫R√(-g)d⁴x"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\n📚 Review the lesson material in LESSON.md for any topics you missed.")
    print("🔬 Practice deriving equations of motion from Lagrangians.\n")


if __name__ == "__main__":
    main()
