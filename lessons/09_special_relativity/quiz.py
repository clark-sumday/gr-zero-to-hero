#!/usr/bin/env python3
"""
Special Relativity Practice Quiz
Tests understanding of Lorentz transformations, time dilation, four-vectors, and energy-momentum
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
            print("🌟 Excellent! You understand Special Relativity.")
        elif percentage >= 75:
            print("👍 Good work! Review the topics you missed.")
        elif percentage >= 60:
            print("📚 Keep studying! Focus on Lorentz transformations.")
        else:
            print("🔄 Review the lesson material and try again.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║      Special Relativity - Practice Quiz                  ║
║      Lesson 9: Spacetime and Lorentz Transformations     ║
╚══════════════════════════════════════════════════════════╝

This quiz tests your understanding of:
• Einstein's postulates
• Spacetime intervals and causality
• Lorentz transformations
• Time dilation and length contraction
• Four-vectors
• Relativistic energy and momentum
    """)

    input("Press Enter to begin...")

    quiz = Quiz()

    # Question 1: Einstein's postulates
    def check_q1(answer):
        ans = answer.lower()
        if ('relativity' in ans or 'laws' in ans or 'inertial' in ans) and ('light' in ans or 'c' in ans or 'constant' in ans):
            return True, "Correct! The two postulates are: (1) Laws of physics are the same in all inertial frames, and (2) Speed of light is constant in all frames."
        else:
            return False, "Incorrect. Einstein's postulates: (1) Principle of Relativity - physics is same in all inertial frames; (2) Light speed c is constant for all observers."

    quiz.question(
        "What are Einstein's two postulates of special relativity? (Brief answer)",
        check_q1,
        hints=[
            "One about the laws of physics in different frames",
            "One about the speed of light",
            "Both are required for SR"
        ]
    )

    # Question 2: Lorentz factor
    def check_q2(answer):
        try:
            ans = float(answer)
            correct = 5.0/3.0  # 1.667
            if np.isclose(ans, correct, rtol=0.01):
                return True, f"Correct! γ = 1/√(1-0.64) = 1/√0.36 = 1/0.6 = 5/3 ≈ {correct:.3f}"
            else:
                return False, f"Incorrect. γ = 1/√(1-v²/c²) = 1/√(1-0.64) = {correct:.3f}"
        except:
            return False, "Please enter a number"

    quiz.question(
        "Calculate the Lorentz factor γ for v = 0.8c.",
        check_q2,
        hints=[
            "γ = 1/√(1 - v²/c²)",
            "v/c = 0.8, so v²/c² = 0.64",
            "γ = 1/√0.36 = ?"
        ]
    )

    # Question 3: Spacetime interval
    def check_q3(answer):
        ans = answer.lower()
        if 'timelike' in ans:
            return True, "Correct! Δs² = -(5)² + (3)² = -16 < 0, so it's TIMELIKE. A massive particle can travel between these events."
        else:
            return False, "Incorrect. Δs² = -c²Δt² + Δx² = -25 + 9 = -16 < 0, which is TIMELIKE (not spacelike or lightlike)."

    quiz.question(
        "Two events have Δt = 5s and Δx = 3 light-seconds. Is the interval timelike, lightlike, or spacelike?",
        check_q3,
        hints=[
            "Compute Δs² = -c²Δt² + Δx²",
            "Δs² < 0 → timelike; Δs² = 0 → lightlike; Δs² > 0 → spacelike",
            "What's -25 + 9?"
        ]
    )

    # Question 4: Time dilation
    def check_q4(answer):
        try:
            ans = float(answer)
            # γ = 1/√(1-0.95²) ≈ 3.20, so τ_lab ≈ 3.20 × 2.2 ≈ 7.04
            gamma = 1/np.sqrt(1 - 0.95**2)
            correct = gamma * 2.2
            if np.isclose(ans, correct, rtol=0.05):
                return True, f"Correct! τ_lab = γτ₀ = {gamma:.2f} × 2.2 ≈ {correct:.2f} μs"
            else:
                return False, f"Incorrect. γ ≈ {gamma:.2f}, so τ_lab = {gamma:.2f} × 2.2 ≈ {correct:.2f} μs"
        except:
            return False, "Please enter a number (in microseconds)"

    quiz.question(
        "A muon has lifetime τ₀ = 2.2 μs in its rest frame. At v = 0.95c, what lifetime do we measure? (in μs)",
        check_q4,
        hints=[
            "Use Δt = γΔτ (time dilation)",
            "First find γ = 1/√(1-0.95²)",
            "Then multiply by 2.2 μs"
        ]
    )

    # Question 5: Length contraction
    def check_q5(answer):
        try:
            ans = float(answer)
            # L = L₀/γ = 1/1.25 = 0.8 m
            gamma = 1/np.sqrt(1 - 0.6**2)
            correct = 1.0 / gamma
            if np.isclose(ans, correct, rtol=0.01):
                return True, f"Correct! L = L₀/γ = 1/{gamma:.2f} = {correct:.2f} m"
            else:
                return False, f"Incorrect. γ = {gamma:.2f}, so L = 1/{gamma:.2f} = {correct:.2f} m"
        except:
            return False, "Please enter a number (in meters)"

    quiz.question(
        "A meter stick moves past you at 0.6c. How long does it appear? (in meters)",
        check_q5,
        hints=[
            "Use L = L₀/γ (length contraction)",
            "Find γ for v = 0.6c",
            "Divide 1 m by γ"
        ]
    )

    # Question 6: Four-velocity norm
    def check_q6(answer):
        ans = answer.lower().replace(' ', '')
        if '-c²' in ans or '-c^2' in ans or '-1' in ans:
            return True, "Correct! u·u = u_μ u^μ = -c² for ALL velocities. This is a Lorentz invariant!"
        else:
            return False, "Incorrect. The four-velocity always satisfies u·u = -c² (or -1 in units where c=1). This is true for any velocity!"

    quiz.question(
        "What is the inner product of the four-velocity with itself? (u·u = ?)",
        check_q6,
        hints=[
            "u^μ = γ(c, v⃗)",
            "u·u = -γ²c² + γ²v²",
            "Simplify using γ² = 1/(1-v²/c²)"
        ]
    )

    # Question 7: Photon momentum
    def check_q7(answer):
        ans = answer.lower().replace(' ', '')
        if 'e/c' in ans or 'e/c' in ans or 'pc=e' in ans:
            return True, "Correct! For photons (m=0): E² = (pc)², so E = pc, thus p = E/c."
        else:
            return False, "Incorrect. For massless particles: E² = (pc)² + 0, so E = pc, giving p = E/c."

    quiz.question(
        "A photon has energy E. What is its momentum p? (Give formula)",
        check_q7,
        hints=[
            "For photons, m = 0",
            "Use E² = (pc)² + (mc²)²",
            "Solve for p"
        ]
    )

    # Question 8: Energy-momentum relation
    def check_q8(answer):
        try:
            ans = float(answer)
            # E = 4 GeV, pc = √(16-1) = √15 ≈ 3.87 GeV
            correct = np.sqrt(16 - 1)
            if np.isclose(ans, correct, rtol=0.02):
                return True, f"Correct! E = 4 GeV, so pc = √(E² - m²c⁴) = √(16-1) ≈ {correct:.2f} GeV"
            else:
                return False, f"Incorrect. E = mc² + K = 4 GeV. Use E² = (pc)² + (mc²)² to get pc = √15 ≈ {correct:.2f} GeV"
        except:
            return False, "Please enter a number (in GeV)"

    quiz.question(
        "A particle has rest mass m = 1 GeV/c² and kinetic energy K = 3 GeV. What is its momentum pc? (in GeV)",
        check_q8,
        hints=[
            "Total energy: E = mc² + K",
            "Use E² = (pc)² + (mc²)²",
            "Solve for pc"
        ]
    )

    # Question 9: Causality
    def check_q9(answer):
        ans = answer.lower()
        if 'no' in ans or 'cannot' in ans or 'impossible' in ans:
            return True, "Correct! Events with spacelike separation (Δs² > 0) cannot be causally connected. No signal can travel between them."
        else:
            return False, "Incorrect. For Δs² = 7 > 0 (spacelike), the events CANNOT be causally connected. Would require v > c!"

    quiz.question(
        "Two events have Δs² = 7 (light-seconds)². Can they be causally connected? (yes/no)",
        check_q9,
        hints=[
            "Δs² > 0 means spacelike separation",
            "What would be required to travel between spacelike events?",
            "Is that possible?"
        ]
    )

    # Question 10: Relativistic energy
    def check_q10(answer):
        ans = answer.lower().replace(' ', '')
        if ('γmc²' in ans or 'gamma' in ans) or 'e=mc²' in ans:
            return True, "Correct! Total relativistic energy is E = γmc². For v=0, γ=1, giving E=mc² (rest energy)."
        else:
            return False, "Incorrect. Total energy E = γmc². This includes rest energy mc² plus kinetic energy (γ-1)mc²."

    quiz.question(
        "What is the formula for total relativistic energy? (In terms of γ, m, c)",
        check_q10,
        hints=[
            "Not just kinetic energy!",
            "Total energy = rest energy + kinetic energy",
            "E = γmc²"
        ]
    )

    # Show final results
    quiz.show_results()

    print("\n📚 Review the lesson material in LESSON.md for any topics you missed.")
    print("🔬 Practice with spacetime diagrams and Lorentz transformations.\n")


if __name__ == "__main__":
    main()
