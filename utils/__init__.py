"""Utility modules for GR tutorial."""

from .ai_assistant import AIAssistant, get_assistant
from .colorblind_colors import COLORS, get_color_cycle, setup_plot_style, get_distinct_style

# Legacy imports (deprecated - kept for backwards compatibility)
try:
    from .lesson_framework import Lesson, wait_for_user
    from .progress_tracker import ProgressTracker, get_tracker
    _legacy_available = True
except ImportError:
    _legacy_available = False

__all__ = [
    'AIAssistant',
    'get_assistant',
    'COLORS',
    'get_color_cycle',
    'setup_plot_style',
    'get_distinct_style'
]

# Add legacy exports if available
if _legacy_available:
    __all__.extend(['Lesson', 'wait_for_user', 'ProgressTracker', 'get_tracker'])
