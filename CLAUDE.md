# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **interactive, text-based tutorial** for learning General Relativity from scratch. Unlike traditional Python tutorials, this is designed as a **computer-native textbook** where users:
- Read theory in markdown files
- Copy/paste code snippets into a Python terminal
- Visualize concepts with matplotlib/numpy
- Practice with built-in quizzes
- Get help from an AI assistant when stuck

The tutorial is a **fast-tracked curriculum** from linear algebra → General Relativity with **no fluff** - every concept is essential.

## Project Structure

```
gr/
├── README.md              # Main project overview + usage guide (start here!)
├── CLAUDE.md              # This file - developer/AI instructions
├── lessons/              # 12 lesson modules
│   ├── 01_linear_algebra/
│   │   ├── LESSON.md          # 📖 Main lesson content (THE TEXTBOOK)
│   │   ├── quiz.py            # 🎯 Interactive practice quiz
│   │   ├── examples/          # 💻 Standalone runnable scripts
│   │   │   └── 01_vector_addition.py
│   │   ├── README.md          # Implementation notes
│   │   └── lesson.py          # [DEPRECATED] Old interactive format
│   ├── 02_multivariable_calculus/
│   ├── 03_differential_equations/
│   ├── 04_curves_surfaces/
│   ├── 05_manifolds/
│   ├── 06_tensors/
│   ├── 07_riemannian_geometry/
│   ├── 08_classical_mechanics/
│   ├── 09_special_relativity/
│   ├── 10_gr_foundations/
│   ├── 11_gr_solutions/
│   └── 12_gr_phenomena/
├── utils/
│   ├── colorblind_colors.py   # Accessible color schemes
│   ├── ai_assistant.py        # Claude API integration
│   ├── lesson_framework.py    # [DEPRECATED] Old Lesson class
│   └── __init__.py
├── requirements.txt
├── .env                       # ANTHROPIC_API_KEY
└── venv/                      # Virtual environment
```

## Learning Philosophy: Three-Panel Setup

Users are expected to have **three panels open**:

```
┌─────────────────┬─────────────────┬─────────────────┐
│   📖 LESSON     │   💻 TERMINAL   │   🤖 AI HELP    │
│                 │                 │                 │
│   Read theory   │   Run code      │   Ask questions │
│   Study math    │   Visualize     │   Get hints     │
│   Work problems │   Experiment    │   Clarify       │
└─────────────────┴─────────────────┴─────────────────┘
```

### Panel 1: The Textbook (LESSON.md)
- Open in markdown viewer, browser, or editor
- Contains concept explanations, theory, math
- Includes ready-to-copy code snippets
- Has practice questions with progressive hints

### Panel 2: Python Terminal
- Keep Python running throughout lesson
- Copy/paste code from LESSON.md
- Run and see output/visualizations
- Modify values and experiment

### Panel 3: AI Assistant (Optional)
```python
from utils.ai_assistant import AIAssistant
assistant = AIAssistant()
assistant.ask("Why does the cross product only work in 3D?")
```

## File Formats

### NEW FORMAT (Current): LESSON.md
Each lesson has a `LESSON.md` file - this is the main content.

**Structure:**
```markdown
# Lesson N: Topic Name

## 1. Section Name

### 📖 Concept
[Theory and intuition]

### 💻 Code Example
```python
# Copy-paste ready code
import numpy as np
u = np.array([3, 4])
print(f"|u| = {np.linalg.norm(u)}")
```

### 📊 Visualization
```python
# Matplotlib visualization code
import matplotlib.pyplot as plt
plt.quiver(...)
plt.show()
```

### 🔬 Explore
Try these experiments...

### 🎯 Practice Question
**Q:** [Question]
<details><summary>Hint</summary>[Hint]</details>
<details><summary>Answer</summary>[Solution]</details>
```

### OLD FORMAT (Deprecated): lesson.py
The old format used interactive Python scripts with the `Lesson` class. **Do not create new lessons in this format.** These are being phased out but remain for reference.

## Development Commands

### Setup
```bash
cd /Users/clarkcarter/Claude/personal/gr
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Set up API key (optional)
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY
```

### Using a Lesson
```bash
# View the lesson
open lessons/01_linear_algebra/LESSON.md
# Or: grip lessons/01_linear_algebra/LESSON.md

# Start Python terminal
python

# Copy/paste code from LESSON.md as you read
```

### Running Practice Quiz
```bash
cd lessons/01_linear_algebra
python quiz.py
```

### Running Example Scripts
```bash
cd lessons/01_linear_algebra/examples
python 01_vector_addition.py
```

## Creating New Lessons

When creating a new lesson, follow this structure:

### 1. Create LESSON.md

```markdown
# Lesson N: Topic Name

**Topics:** Brief list
**Prerequisites:** What student should know
**Time:** Estimated hours

## Table of Contents
[Links to sections]

## 1. First Section

### 📖 Concept
Clear explanation of theory...

### 💻 Code Example
```python
import numpy as np
# Working code ready to copy
```

### 📊 Visualization
```python
import matplotlib.pyplot as plt
# Visualization code
```

### 🔬 Explore
Experiments to try...

### 🎯 Practice Question
Question with progressive hints...
```

### 2. Create quiz.py

```python
#!/usr/bin/env python3
"""Practice quiz for Lesson N"""

class Quiz:
    # Quiz implementation (see 01_linear_algebra/quiz.py as template)
    pass

if __name__ == "__main__":
    main()
```

### 3. Create examples/ directory

```python
# examples/01_concept_name.py
#!/usr/bin/env python3
"""
Example: Concept Name
Brief description of what this demonstrates
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS

# Standalone working code
```

### 4. Update README.md (lesson-specific)

Document implementation notes, known issues, topics covered.

## Visualization Guidelines

### Always Use Colorblind-Friendly Colors

```python
from utils.colorblind_colors import COLORS

# Use COLORS dictionary instead of basic colors
plt.plot(x, y, color=COLORS['blue'])   # ✅ Good
plt.plot(x, y, color='blue')           # ❌ Avoid

# Available colors:
COLORS = {
    'blue': '#0173B2',      # Strong blue
    'orange': '#DE8F05',    # Vivid orange
    'green': '#029E73',     # Bluish green
    'red': '#CC3311',       # Vermillion red
    'purple': '#9C3587',    # Purple
    'yellow': '#ECE133',    # Yellow
    'cyan': '#56B4E9',      # Sky blue
    'pink': '#CC79A7',      # Reddish purple
    'black': '#000000',
    'gray': '#888888'
}
```

### Choose Vectors with Distinct Directions

When demonstrating vector operations (addition, subtraction, etc.), use vectors that point in visually distinct directions to emphasize the concept:

```python
# ✅ Good: Vectors point in different directions (easier to see parallelogram)
u = np.array([3, 1])  # Mostly horizontal
v = np.array([1, 3])  # Mostly vertical

# ❌ Avoid: Vectors point in similar directions
u = np.array([3, 4])  # Both up and to the right
v = np.array([1, 2])  # Also up and to the right
```

### Label All Arrows/Vectors

```python
# Add text labels at arrow endpoints
plt.quiver(0, 0, u[0], u[1], color=COLORS['blue'])
plt.text(u[0], u[1], ' u', fontsize=12, color=COLORS['blue'],
         verticalalignment='bottom')
```

### Standard Plot Setup

```python
plt.figure(figsize=(8, 6))
plt.grid(True, alpha=0.3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Clear Descriptive Title")
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.gca().set_aspect('equal')  # For geometric plots
plt.legend()
plt.show()
```

## Content Guidelines

### Accuracy Requirements

All content must be accurate and verifiable:

- **Source from open resources only:**
  - MIT OpenCourseWare (8.962 GR, 18.06 Linear Algebra)
  - ArXiv preprints
  - Sean Carroll's GR lecture notes (arXiv:gr-qc/9712019)
  - Leonard Susskind's lectures
  - 3Blue1Brown (for visual intuition)
  - Wikipedia for general concepts

- Mathematical equations are public domain facts
- Create **original explanations** - no copyrighted text
- Cross-reference against multiple sources
- Use established libraries (numpy, scipy, einsteinpy)
- Verify numerical results against known solutions

### Pedagogical Requirements

Based on lesson-pedagogy-reviewer agent feedback:

1. **Progressive Complexity:** Build from simple → complex
2. **Geometric Intuition:** Always provide visual/physical interpretation
3. **Active Learning:** Include experiments and practice questions
4. **Physics Connections:** Relate math to GR concepts
5. **Complete Visualizations:** Show proper geometric constructions (e.g., tip-to-tail for vectors)
6. **Physical Intuition:** Use analogies (rubber sheet stretching, etc.)
7. **Clear Notation:** Explain all symbols before using them

### Required Sections

Every lesson must have:

1. **Concept explanations** - Theory with intuition
2. **Code examples** - Working, copy-paste ready
3. **Visualizations** - At least 3-5 per lesson
4. **Practice questions** - With progressive hints
5. **Quiz script** - 8-10 questions
6. **Example scripts** - Standalone demos
7. **GR connections** - Why this matters for relativity

### Writing Style

- **Concise and direct** - No unnecessary prose
- **Math-forward** - Show equations, explain them
- **Code-ready** - All snippets must work as-is
- **Progressive hints** - Don't give away answers immediately
- **Clear instructions** - "Copy this into your terminal"

## Code Style

### Python Code

```python
# Clear variable names in explanations
velocity = np.array([3, 4, 5])  # ✅ Good for teaching
v = np.array([3, 4, 5])         # ✅ OK in compact examples

# Add comments for physics/math
# Compute Christoffel symbols Γ^μ_νλ
christoffel = ...

# Use print statements to show results
print(f"Magnitude: {np.linalg.norm(v):.2f}")

# Keep functions focused and demonstrative
def show_vector_addition(u, v):
    """Visualize u + v using tip-to-tail method."""
    # Implementation
```

### Quiz Questions

```python
def check_answer(user_answer):
    """
    Returns (is_correct: bool, feedback: str)
    """
    try:
        # Parse user input
        ans = float(user_answer)

        # Check correctness
        if np.isclose(ans, correct_answer):
            return True, f"Correct! Explanation of why..."
        else:
            return False, f"Incorrect. The answer is {correct_answer} because..."
    except:
        return False, "Please enter a number"
```

## AI Assistant Integration

### In LESSON.md Files

Show users how to use it:

```markdown
**Need Help?** Use the AI assistant:
```python
from utils.ai_assistant import AIAssistant
assistant = AIAssistant()
assistant.set_lesson_context("Lesson 1", "Linear Algebra", ["vectors", "dot product"])
assistant.ask("Can you explain why eigenvalues are important for GR?")
```
```

### Key Methods

```python
# Create assistant
assistant = AIAssistant()

# Set lesson context (helps it understand what you're learning)
assistant.set_lesson_context(
    lesson_name="Lesson 1: Linear Algebra",
    topic="Vectors and Matrices",
    concepts=["vectors", "linear independence", "eigenvalues"]
)

# Ask questions
response = assistant.ask("Why is linear independence important?")

# Clear history if switching topics
assistant.clear_history()
```

## Dependencies

### Core Requirements
- numpy - Numerical computations
- matplotlib - 2D/3D plotting
- scipy - Scientific algorithms
- sympy - Symbolic mathematics
- anthropic - AI assistant API
- python-dotenv - Environment variables

### Physics-Specific
- einsteinpy - GR calculations (Lessons 10-12)

### Optional
- jupyter - Notebook interface
- ipython - Enhanced terminal
- manim - Animations (requires `brew install cairo`)

## Common Tasks

### Converting Old Lesson to New Format

1. Read the old `lesson.py`
2. Extract all `lesson.explain()` calls → `### 📖 Concept` sections
3. Extract `code_example()` functions → `### 💻 Code Example` blocks
4. Extract visualization code → `### 📊 Visualization` blocks
5. Extract `practice_question()` calls → `### 🎯 Practice Question` with collapsible hints
6. Create `quiz.py` from the practice questions
7. Create standalone scripts in `examples/` from code examples

### Adding Colorblind-Friendly Colors to Existing Code

Find: `color='red'` or `color='blue'`
Replace with: `color=COLORS['red']` or `color=COLORS['blue']`

Add import:
```python
from utils.colorblind_colors import COLORS
```

### Testing a Lesson

1. Open LESSON.md in markdown viewer
2. Start Python terminal
3. Copy each code block sequentially
4. Verify all code runs without errors
5. Check all plots display correctly
6. Run `python quiz.py` and verify questions work
7. Run example scripts: `python examples/*.py`

## Troubleshooting

### Matplotlib Issues
```python
# Try different backend
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### Path Issues
```python
# Add project to path
import sys
sys.path.append('/Users/clarkcarter/Claude/personal/gr')
from utils.colorblind_colors import COLORS
```

### AI Assistant Not Working
- Check `.env` has valid `ANTHROPIC_API_KEY`
- Lessons work fine without it (no `/ask` feature)

## Notes

- All 12 lessons are complete in new LESSON.md format
- Each lesson includes inline code explanations for all numpy/matplotlib/scipy functions
- Old `lesson.py` files kept for reference but deprecated
- All new development should use LESSON.md format
- Focus on **no fluff** - every sentence must be essential
- This is a **fast-tracked curriculum** - direct path to GR
- Target: 50-80 hours total from zero → General Relativity

## Curriculum Path

1. **Linear Algebra** ✅ (Complete)
1.5. **Single-Variable Calculus** ✅ (Complete - NEW! Essential foundation)
2. **Multivariable Calculus** ✅ (Complete - includes 3D curl visualization)
3. **Differential Equations** ✅ (Complete)
4. **Curves & Surfaces** ✅ (Complete)
5. **Manifolds** ✅ (Complete)
6. **Tensors** ✅ (Complete)
7. **Riemannian Geometry** ✅ (Complete - enhanced parallel transport visualization)
8. **Classical Mechanics** ✅ (Complete)
9. **Special Relativity** ✅ (Complete)
10. **GR Foundations** ✅ (Complete)
11. **GR Solutions** ✅ (Complete - includes Penrose diagram)
12. **GR Phenomena** ✅ (Complete)

Each lesson takes ~4-6 hours to complete. Lesson 1.5 takes ~6-8 hours.

### Recent Enhancements

**NEW LESSON ADDED (Critical Gap Fixed):**
- **Lesson 1.5: Single-Variable Calculus** (6-8 hours)
  - Fills critical gap identified by pedagogy review
  - Covers limits, derivatives, integration, fundamental theorem, exponentials
  - Makes curriculum truly "from scratch" - no prior calculus needed
  - Includes 4 visualization examples and 10-question quiz
  - Total curriculum now 56-88 hours (up from 50-80)

**Accuracy & Clarity Improvements:**
- Sign convention notes in Lesson 7 (Riemann tensor)
- Metric signature declarations in all GR lessons (7, 9, 10, 11, 12)
- GPS approximation notation in Lesson 12 (weak-field, first-order)
- Frame-dragging vector formula clarification in Lesson 12
- Ergosphere geometric units specification in Lesson 12

**Visualization Enhancements:**
- 3D curl visualization in Lesson 2 (shows rotation axis clearly)
- Enhanced parallel transport in Lesson 7 (vectors at all triangle vertices)
- Penrose diagram in Lesson 11 (causal structure of Schwarzschild black hole)

**Code Quality:**
- All code snippets include inline explanations of numpy/matplotlib/scipy functions
- Pattern: `np.array()  # np.array() converts Python list to efficient numpy array`
