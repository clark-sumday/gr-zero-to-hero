"""
AI Assistant for interactive Q&A during lessons.
Uses Anthropic's Claude API to answer questions contextually.
"""

import os
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class AIAssistant:
    def __init__(self, api_key=None):
        """Initialize the AI assistant with Claude API."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Please set it in your .env file or environment."
            )
        self.client = Anthropic(api_key=self.api_key)
        self.conversation_history = []
        self.current_lesson_context = ""

    def set_lesson_context(self, lesson_name, current_topic, concepts_covered):
        """Set context about the current lesson for more relevant answers."""
        self.current_lesson_context = f"""
Current Lesson: {lesson_name}
Current Topic: {current_topic}
Concepts Covered So Far: {', '.join(concepts_covered)}
"""

    def ask(self, question):
        """
        Ask the AI assistant a question.
        Returns the assistant's response as a string.
        """
        # Build system prompt with lesson context
        system_prompt = f"""You are a patient, knowledgeable physics and mathematics tutor helping a student learn General Relativity from scratch.

{self.current_lesson_context}

Guidelines:
- Provide clear, accurate explanations suitable for someone learning the topic
- Use analogies and examples when helpful
- If the question relates to the current lesson, reference concepts they've already learned
- Encourage mathematical rigor but explain intuition first
- Cite open sources when making specific claims (ArXiv, MIT OCW, etc.)
- If asked for code examples, use numpy/matplotlib/manim
- Be encouraging and supportive of the learning process
"""

        # Add user question to history
        self.conversation_history.append({
            "role": "user",
            "content": question
        })

        # Get response from Claude
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=self.conversation_history
        )

        # Extract assistant's response
        assistant_message = response.content[0].text

        # Add to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def clear_history(self):
        """Clear conversation history (useful when starting a new lesson)."""
        self.conversation_history = []


def get_assistant():
    """Factory function to create an AI assistant instance."""
    try:
        assistant = AIAssistant()
        print("✓ AI Assistant initialized successfully")
        return assistant
    except ValueError as e:
        print(f"\n⚠️  Warning: {e}")
        print("AI assistant will not be available. You can still complete the lessons.")
        return None
    except Exception as e:
        print(f"\n⚠️  Unexpected error initializing AI assistant: {e}")
        print("AI assistant will not be available. You can still complete the lessons.")
        return None
