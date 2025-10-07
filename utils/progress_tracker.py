"""
Progress tracking for GR tutorial lessons.
Saves progress to JSON file so you can resume where you left off.
"""

import json
from pathlib import Path
from datetime import datetime


class ProgressTracker:
    def __init__(self):
        # Store progress in project directory
        project_root = Path(__file__).parent.parent
        self.progress_file = project_root / '.progress.json'
        self.progress = self.load_progress()

    def load_progress(self):
        """Load progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "lessons_completed": [],
            "current_lesson": None,
            "sections_completed": {},
            "last_updated": None
        }

    def save_progress(self):
        """Save progress to file."""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def mark_section_complete(self, lesson_name, section_name):
        """Mark a section as completed."""
        if lesson_name not in self.progress["sections_completed"]:
            self.progress["sections_completed"][lesson_name] = []

        if section_name not in self.progress["sections_completed"][lesson_name]:
            self.progress["sections_completed"][lesson_name].append(section_name)

        self.progress["current_lesson"] = lesson_name
        self.save_progress()

    def mark_lesson_complete(self, lesson_name):
        """Mark an entire lesson as completed."""
        if lesson_name not in self.progress["lessons_completed"]:
            self.progress["lessons_completed"].append(lesson_name)
        self.save_progress()

    def get_lesson_progress(self, lesson_name):
        """Get sections completed for a specific lesson."""
        return self.progress["sections_completed"].get(lesson_name, [])

    def is_lesson_complete(self, lesson_name):
        """Check if a lesson is complete."""
        return lesson_name in self.progress["lessons_completed"]

    def show_summary(self):
        """Display progress summary."""
        print("\n" + "="*80)
        print("📊 YOUR PROGRESS")
        print("="*80)

        if self.progress["lessons_completed"]:
            print(f"\n✅ Completed Lessons: {', '.join(self.progress['lessons_completed'])}")
        else:
            print("\n✅ Completed Lessons: None yet")

        if self.progress["current_lesson"]:
            current = self.progress["current_lesson"]
            sections = self.progress["sections_completed"].get(current, [])
            print(f"\n📖 Current Lesson: {current}")
            if sections:
                print(f"   Sections completed: {', '.join(sections)}")

        if self.progress["last_updated"]:
            print(f"\n🕐 Last updated: {self.progress['last_updated']}")

        print("="*80 + "\n")

    def reset(self):
        """Reset all progress."""
        self.progress = {
            "lessons_completed": [],
            "current_lesson": None,
            "sections_completed": {},
            "last_updated": None
        }
        self.save_progress()


# Global tracker instance
_tracker = None

def get_tracker():
    """Get global progress tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ProgressTracker()
    return _tracker
