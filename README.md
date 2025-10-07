# General Relativity: Zero to Hero

**A fast-tracked, computer-native textbook to learn General Relativity from first principles**

> No fluff. Just rigorous physics, interactive code, and hands-on learning.

---

## 🎯 What This Is

An **interactive, text-based tutorial** designed specifically for learning on a computer. Think of it as a modern textbook where you:
- **Read** theory in markdown files
- **Code** examples in a Python terminal
- **Visualize** concepts with matplotlib/numpy
- **Practice** with built-in quizzes
- **Ask** an AI assistant when stuck

**This is NOT:**
- A video course
- A traditional Python tutorial
- A series of Jupyter notebooks
- A watered-down introduction

**This IS:**
- A complete curriculum from linear algebra → General Relativity
- Designed for self-paced learning (50-80 hours total)
- Built around experimentation and visualization
- Focused on deep understanding, not memorization

---

## 🚀 Quick Start

### 1. Installation

```bash
cd /path/to/gr
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Optional: AI Assistant Setup

```bash
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here
```

### 3. Set Up Three Panels

For the best learning experience, open three side-by-side panels:

```
┌─────────────────┬─────────────────┬─────────────────┐
│   📖 LESSON     │   💻 TERMINAL   │   🤖 AI HELP    │
│   (Textbook)    │   (Code/Play)   │   (If stuck)    │
│                 │                 │                 │
│   LESSON.md     │   python        │   assistant.py  │
│   Read concepts │   Run snippets  │   Ask questions │
│   Study theory  │   Visualize     │   Get hints     │
│   Work problems │   Experiment    │   Clarify       │
└─────────────────┴─────────────────┴─────────────────┘
```

### 4. Start Learning

```bash
# Panel 1: Open the lesson textbook
open lessons/01_linear_algebra/LESSON.md
# Or: grip lessons/01_linear_algebra/LESSON.md (for live preview)

# Panel 2: Start Python terminal
python

# Panel 3 (optional): Open AI assistant in Python
# >>> from utils.ai_assistant import AIAssistant
# >>> assistant = AIAssistant()
```

**First lesson:** [Linear Algebra Foundations](lessons/01_linear_algebra/LESSON.md)

---

## 💻 How the Three-Panel Setup Works

### Panel 1: The Textbook (LESSON.md)

Open the lesson markdown file in your editor, browser, or markdown viewer.

Each `LESSON.md` contains:
1. **📖 Concept** - Theory and intuition
2. **💻 Code Example** - Copy/paste ready snippets
3. **📊 Visualization** - Matplotlib graphics code
4. **🔬 Explore** - Experiments to try
5. **🎯 Practice Questions** - Test understanding with progressive hints

### Panel 2: Python Terminal

Keep Python running throughout the lesson:

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from utils.colorblind_colors import COLORS

# Copy snippets from LESSON.md
>>> u = np.array([3, 1])
>>> v = np.array([1, 3])
>>> print(f"u + v = {u + v}")
u + v = [4 4]

# Run visualizations
>>> plt.quiver(0, 0, u[0], u[1], color=COLORS['blue'])
>>> plt.show()
```

**Terminal Tips:**
- Use `ipython` for better interactive experience
- Use `%matplotlib` for auto-displaying plots
- Modify values and re-run to experiment
- Keep it open - don't restart between sections!

### Panel 3: AI Assistant (Optional)

If you get stuck, use the AI assistant:

```python
from utils.ai_assistant import AIAssistant

assistant = AIAssistant()
assistant.set_lesson_context(
    "Lesson 1: Linear Algebra",
    "Vectors and Matrices",
    ["vectors", "dot product", "cross product"]
)

# Ask questions
assistant.ask("Why does the cross product only work in 3D?")
assistant.ask("Can you give me another example of eigenvectors?")
assistant.ask("I don't understand the parallelogram method")
```

---

## 📚 Curriculum: 12 Lessons

### Foundation Mathematics (Lessons 1-3)
1. **Linear Algebra** ✅ - Vectors, matrices, eigenvalues, transformations
2. **Multivariable Calculus** ✅ - Gradients, divergence, curl, Lagrange multipliers
3. **Differential Equations** ✅ - ODEs, PDEs, numerical methods, phase portraits

### Differential Geometry (Lessons 4-7)
4. **Curves & Surfaces** ✅ - Parametric curves, curvature, Frenet-Serret frames
5. **Manifolds** ✅ - Abstract spaces, tangent spaces, coordinate charts
6. **Tensors** ✅ - Multilinear algebra, index notation, tensor operations
7. **Riemannian Geometry** ✅ - Metric tensor, connections, curvature tensors

### Physics Prerequisites (Lessons 8-9)
8. **Classical Mechanics** ✅ - Lagrangian, Hamiltonian, Noether's theorem
9. **Special Relativity** ✅ - Spacetime, Lorentz transformations, 4-vectors

### General Relativity (Lessons 10-12)
10. **GR Foundations** ✅ - Equivalence principle, Einstein field equations
11. **GR Solutions** ✅ - Schwarzschild, Kerr, FLRW, gravitational waves
12. **GR Phenomena** ✅ - Black holes, lensing, GPS, cosmology

**Total time:** ~50-80 hours (depending on prior knowledge)
**Each lesson:** ~4-6 hours

---

## 📖 How Lessons Work

### Lesson Structure

Each lesson folder contains:

```
lessons/01_linear_algebra/
├── LESSON.md           # 📖 Main content (THE TEXTBOOK - read this!)
├── quiz.py             # 🎯 Practice quiz (run after completing lesson)
├── examples/           # 💻 Standalone runnable scripts
│   ├── 01_vector_addition.py
│   ├── 02_span_visualization.py
│   └── ...
└── README.md           # Implementation notes
```

### Example Workflow

1. **Read** a section in LESSON.md
2. **Copy** code snippet to Python terminal
3. **Run** and see output/visualizations
4. **Modify** values and re-run to experiment
5. **Answer** practice questions (use hints if stuck)
6. **Move** to next section
7. **Quiz** yourself when done: `python quiz.py`

### Alternative: Run Example Scripts

If you prefer complete standalone scripts:

```bash
cd lessons/01_linear_algebra/examples
python 01_vector_addition.py
python 02_span_visualization.py
# etc.
```

These demonstrate each concept as a full program.

---

## 🎓 Learning Philosophy

### No Fluff
- Every concept is essential for GR
- No tangents or "interesting facts"
- Direct path from prerequisites → Einstein equations

### Computer-Native
- Designed for learning at a computer
- Interactive code, not passive reading
- Build intuition through visualization
- Experiment with parameters in real-time

### First Principles
- Start from basic math
- Build up rigorously
- No hand-waving
- Understand *why*, not just *what*

### Active Learning
- You write code, not just read it
- Practice problems throughout
- Quizzes to test understanding
- AI assistant for stuck moments

---

## 🎯 Prerequisites

### Required
- **Programming:** Basic Python (variables, functions, loops)
- **Math:** High school algebra, basic trigonometry
- **Time:** Commitment to work through 50-80 hours of material

### NOT Required
- College-level mathematics (you'll learn it)
- Physics degree (explained from scratch)
- Previous GR exposure

### Ideal Background
- Undergraduate STEM student or equivalent
- Comfortable with mathematical notation
- Curious about spacetime and gravity!

---

## 💡 Tips for Success

### Study Strategies

1. **Don't Skip** - Each lesson builds on previous ones
2. **Code Everything** - Type out examples, don't just read
3. **Visualize Often** - Run all visualization code
4. **Experiment** - Change values, break things, explore
5. **Do Practice Problems** - They solidify understanding
6. **Use AI Assistant** - No shame in asking for help
7. **Take Breaks** - GR is dense; let it sink in
8. **Review** - Revisit earlier lessons if confused

### Common Terminal Workflow

```python
# Keep Python running, import at the start
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/path/to/gr')
from utils.colorblind_colors import COLORS

# For better plotting in iPython
%matplotlib

# Then just copy/paste snippets from LESSON.md as you read!
```

---

## 🔧 Key Features

### ✅ Colorblind-Friendly Visualizations
All plots use scientifically-validated color schemes that work for all types of colorblindness.

### ✅ Progressive Difficulty
Each lesson builds on previous ones. Concepts are introduced when needed, not before.

### ✅ Practice Questions
Every section has questions with progressive hints and detailed solutions.

### ✅ AI Assistant Integration
Get help when stuck without leaving your learning flow.

### ✅ Based on Open Resources
All content derived from MIT OCW, arXiv, and other open educational materials.

---

## 📊 Example: What You'll Code

**Lesson 1 (Linear Algebra):**
```python
import numpy as np
import matplotlib.pyplot as plt
from utils.colorblind_colors import COLORS

# Vector addition with parallelogram visualization
u = np.array([3, 1])
v = np.array([1, 3])
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
           color=COLORS['blue'], label='u')
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color=COLORS['orange'], label='v')
plt.quiver(0, 0, (u+v)[0], (u+v)[1], angles='xy', scale_units='xy', scale=1,
           color=COLORS['green'], label='u+v')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()
```

**Lesson 6 (Tensors):**
```python
# Metric tensor for 2D surface in polar coordinates
g = np.array([[1, 0], [0, r**2]])
g_inv = np.linalg.inv(g)

# Raise/lower indices
v_lower = g @ v_upper
```

**Lesson 11 (Schwarzschild Black Hole):**
```python
from einsteinpy.metric import Schwarzschild

# Black hole spacetime
bh = Schwarzschild(M=1.0)
geodesic = bh.calculate_trajectory(initial_conditions)
visualize_orbit(geodesic)
```

---

## 🧪 Experiments You'll Run

- Visualize vector spaces and spans
- See how matrices transform geometric shapes
- Plot scalar fields and gradients
- Visualize curvature of surfaces
- Compute geodesics on curved spaces
- Simulate orbits around black holes
- Visualize gravitational waves

---

## 🐛 Troubleshooting

### Plots not showing
```python
# Try different backend
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
plt.show()
```

### Import errors
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Verify you're in project directory
pwd  # should show /path/to/gr

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Test imports
python -c "from utils import colorblind_colors; print('✓ Working!')"
```

### AI assistant not working
```bash
# Check .env file exists and has API key
cat .env  # Should show: ANTHROPIC_API_KEY=sk-...

# Test it
python -c "from utils.ai_assistant import AIAssistant; a = AIAssistant(); print('✓ Working!')"
```

**Note:** Lessons work fine without the AI assistant - it's just an optional help feature!

### Path issues in code snippets
```python
# Add project to Python path at start of session
import sys
sys.path.append('/path/to/gr')  # Use your actual path
from utils.colorblind_colors import COLORS
```

---

## 📚 Additional Resources

Content based on:
- **MIT OCW 8.962** - General Relativity (Scott Hughes)
- **MIT OCW 18.06** - Linear Algebra (Gilbert Strang)
- **Sean Carroll** - Lecture Notes on GR (arXiv:gr-qc/9712019)
- **Leonard Susskind** - Theoretical Minimum lectures
- **3Blue1Brown** - Essence of Linear Algebra (YouTube)
- **EinsteinPy** - Python library for GR calculations

---

## 📈 What You'll Understand By The End

- Why spacetime is curved by mass/energy
- How Einstein's field equations work
- What happens at black hole event horizons
- Why gravitational waves exist
- How GPS satellites account for GR effects
- Why time runs slower in gravity
- The mathematics of curved geometry
- How to compute in tensor notation

**You'll go from "What's a vector?" to deriving the Schwarzschild solution.**

---

## 🎓 Ready to Start?

1. ✅ Set up your environment (see Quick Start above)
2. ✅ Open three panels (textbook, terminal, AI assistant)
3. ✅ Begin with [Lesson 1: Linear Algebra](lessons/01_linear_algebra/LESSON.md)

**The journey from linear algebra to General Relativity starts now. Let's go! 🚀**

---

## 📂 Project Structure

```
gr/
├── README.md              # This file (start here!)
├── CLAUDE.md              # Developer/AI instructions
├── lessons/               # 12 lesson modules
│   ├── 01_linear_algebra/
│   │   ├── LESSON.md      # Main textbook content
│   │   ├── quiz.py        # Practice quiz
│   │   ├── examples/      # Runnable scripts
│   │   └── README.md      # Implementation notes
│   ├── 02_multivariable_calculus/
│   └── ...
├── utils/
│   ├── colorblind_colors.py   # Accessible color schemes
│   ├── ai_assistant.py        # Claude API integration
│   └── __init__.py
├── requirements.txt
├── .env                   # API key (optional)
└── venv/                  # Virtual environment
```

**For developers/contributors:** See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.
