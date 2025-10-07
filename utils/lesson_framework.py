"""
Base framework for interactive lessons.
Provides structure for Read → Ask → See → Code → Practice flow.
"""

import sys
from typing import List, Callable, Optional
from .ai_assistant import get_assistant
from .progress_tracker import get_tracker


class Lesson:
    def __init__(self, name: str, topic: str):
        self.name = name
        self.topic = topic
        self.concepts_covered = []
        self.ai_assistant = get_assistant()
        self.progress_tracker = get_tracker()
        self.sections_completed = []

        # Show progress summary
        self.progress_tracker.show_summary()

        # Set initial context for AI assistant
        if self.ai_assistant:
            self.ai_assistant.set_lesson_context(name, topic, [])

        self.print_header()

    def print_header(self):
        """Print lesson header."""
        print("\n" + "="*80)
        print(f"  {self.name}")
        print(f"  Topic: {self.topic}")
        print("="*80)
        print("\nCommands:")
        print("  /ask <question> - Ask the AI assistant anything")
        print("  /skip - Skip current section")
        print("  /quit - Exit lesson")
        print("="*80 + "\n")

    def section(self, title: str):
        """Start a new section within the lesson."""
        # Mark previous section as complete if exists
        if self.sections_completed:
            self.progress_tracker.mark_section_complete(
                self.name,
                self.sections_completed[-1]
            )

        # Track current section
        self.sections_completed.append(title)

        # Check if already completed
        completed_sections = self.progress_tracker.get_lesson_progress(self.name)
        is_completed = title in completed_sections

        status = "✅" if is_completed else "📖"

        print(f"\n{'─'*80}")
        print(f"{status} {title}")
        print(f"{'─'*80}\n")

    def explain(self, text: str):
        """Display explanatory text."""
        print(text)
        print()

    def concept(self, concept_name: str):
        """Mark a concept as covered and update AI context."""
        self.concepts_covered.append(concept_name)
        if self.ai_assistant:
            self.ai_assistant.set_lesson_context(
                self.name,
                self.topic,
                self.concepts_covered
            )

    def code_example(self, description: str, code_func: Callable):
        """
        Run a code example demonstration.

        Args:
            description: What this example demonstrates
            code_func: Function containing the example code to run
        """
        print(f"💻 Code Example: {description}\n")
        try:
            code_func()
            print("\n📊 Plot displayed in separate window. You can interact with it.")
            print("   Close the plot window or return here to continue...")
        except Exception as e:
            print(f"❌ Error running example: {e}")
        print()

    def challenge(self, description: str, hints: Optional[List[str]] = None):
        """
        Present a coding challenge for the student.

        Args:
            description: What to implement
            hints: Optional list of hints
        """
        print(f"🎯 Challenge: {description}\n")
        if hints:
            print("Hints available - type '/hint' to see them\n")

        while True:
            response = input("When you're done, press Enter to continue (or /hint, /skip, /ask): ").strip()

            if response == "":
                break
            elif response == "/skip":
                print("Skipping challenge...\n")
                break
            elif response == "/hint" and hints:
                for i, hint in enumerate(hints, 1):
                    print(f"  Hint {i}: {hint}")
                print()
            elif response.startswith("/ask "):
                question = response[5:].strip()
                if question:
                    self._handle_ask(question)
                else:
                    print("Usage: /ask <your question>")
                    print("Example: /ask how do I do this in matplotlib?")
            elif response == "/ask":
                print("Usage: /ask <your question>")
                print("Example: /ask how do I do this in matplotlib?")
            else:
                print("Unknown command. Use /hint, /skip, /ask <question>, or press Enter.\n")

    def practice_question(self, question: str, answer: str, hints: Optional[List[str]] = None):
        """
        Ask a practice question.

        Args:
            question: The question to ask
            answer: The correct answer (hidden until revealed)
            hints: Optional list of hints
        """
        print(f"❓ Practice Question:\n{question}\n")
        if hints:
            print("Hints available - type '/hint' to see them")

        while True:
            response = input("\nEnter your answer (or /answer, /hint, /skip, /ask): ").strip()

            if response == "/answer":
                print(f"\n✓ Answer: {answer}\n")
                break
            elif response == "/skip":
                print("Skipping question...\n")
                break
            elif response == "/hint" and hints:
                for i, hint in enumerate(hints, 1):
                    print(f"  Hint {i}: {hint}")
                print()
            elif response.startswith("/ask "):
                question_text = response[5:].strip()
                if question_text:
                    self._handle_ask(question_text)
                else:
                    print("Usage: /ask <your question>")
                    print("Example: /ask what is an eigenvector?")
            elif response == "/ask":
                print("Usage: /ask <your question>")
                print("Example: /ask what is an eigenvector?")
            elif response:
                # Student provided an answer
                check = input("Check answer? (y/n): ").strip().lower()
                if check == 'y':
                    print(f"\n✓ Correct answer: {answer}")
                    user_correct = input("Did you get it right? (y/n): ").strip().lower()
                    if user_correct == 'y':
                        print("Great job! 🎉\n")
                    else:
                        print("Review the solution and try again if needed.\n")
                    break
            else:
                print("Please provide an answer or use a command.\n")

    def pause(self, message: str = "Press Enter to continue..."):
        """Pause for user to read or complete a task."""
        while True:
            response = input(f"\n{message} ").strip()
            if response == "":
                break
            elif response.startswith("/ask "):
                question = response[5:].strip()
                if question:
                    self._handle_ask(question)
                else:
                    print("Usage: /ask <your question>")
                    print("Example: /ask how do I visualize vector addition?")
            elif response == "/ask":
                print("Usage: /ask <your question>")
                print("Example: /ask how do I visualize vector addition?")
            else:
                print("Press Enter to continue, or use /ask <question>")

    def _handle_ask(self, question: str):
        """Handle /ask command."""
        if not self.ai_assistant:
            print("\n⚠️  AI assistant not available (missing API key).")
            print("Please set ANTHROPIC_API_KEY in your .env file.\n")
            return

        print(f"\n🤖 AI Assistant (thinking...)\n")
        try:
            response = self.ai_assistant.ask(question)
            print(response)
            print("\n" + "─"*80 + "\n")
        except Exception as e:
            print(f"\n❌ Error communicating with AI assistant: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print()

    def complete(self):
        """Mark lesson as complete."""
        # Mark final section complete
        if self.sections_completed:
            self.progress_tracker.mark_section_complete(
                self.name,
                self.sections_completed[-1]
            )

        # Mark entire lesson complete
        self.progress_tracker.mark_lesson_complete(self.name)

        print("\n" + "="*80)
        print(f"✅ Lesson Complete: {self.name}")
        print(f"Concepts covered: {', '.join(self.concepts_covered)}")
        print("="*80 + "\n")

        # Show updated progress
        self.progress_tracker.show_summary()


def wait_for_user(prompt: str = "Press Enter when ready to continue..."):
    """Simple utility to pause execution."""
    input(f"\n{prompt} ")
