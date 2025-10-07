#!/usr/bin/env python3
"""
Script to add inline code explanations to all LESSON.md files.
Adds comments explaining what numpy/scipy/matplotlib functions do.
"""

import re
from pathlib import Path

# Common function explanations
EXPLANATIONS = {
    # NumPy
    'import numpy as np': 'import numpy as np  # NumPy for numerical arrays and linear algebra operations',
    'np.array(': '# np.array() converts Python list/tuple to efficient numpy array',
    'np.dot(': '# np.dot() computes dot product of two arrays',
    'np.cross(': '# np.cross() computes cross product (3D vectors only)',
    'np.linalg.norm(': '# np.linalg.norm() computes vector magnitude (Euclidean norm)',
    'np.linalg.inv(': '# np.linalg.inv() computes matrix inverse',
    'np.linalg.det(': '# np.linalg.det() computes matrix determinant',
    'np.linalg.eig(': '# np.linalg.eig() computes eigenvalues and eigenvectors',
    'np.linalg.eigvals(': '# np.linalg.eigvals() computes eigenvalues only',
    'np.linspace(': '# np.linspace() creates evenly spaced array between start and end',
    'np.meshgrid(': '# np.meshgrid() creates coordinate matrices from coordinate vectors',
    'np.sin(': '# np.sin() computes sine (element-wise for arrays)',
    'np.cos(': '# np.cos() computes cosine (element-wise for arrays)',
    'np.exp(': '# np.exp() computes exponential e^x',
    'np.sqrt(': '# np.sqrt() computes square root',
    'np.isclose(': '# np.isclose() tests if values are approximately equal (handles floating point)',
    'np.allclose(': '# np.allclose() tests if all array elements are approximately equal',
    'np.zeros(': '# np.zeros() creates array filled with zeros',
    'np.ones(': '# np.ones() creates array filled with ones',
    'np.column_stack(': '# np.column_stack() stacks 1-D arrays as columns into 2-D array',
    'np.vstack(': '# np.vstack() stacks arrays vertically (row-wise)',
    '@': '  # @ is matrix multiplication operator (equivalent to np.dot for matrices)',

    # Matplotlib
    'import matplotlib.pyplot as plt': 'import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization',
    'plt.figure(': '# plt.figure() creates a new figure for plotting',
    'plt.subplot(': '# plt.subplot() creates subplot in grid layout (rows, cols, position)',
    'plt.plot(': '# plt.plot() draws line plot',
    'plt.scatter(': '# plt.scatter() draws scatter plot (individual points)',
    'plt.quiver(': '# plt.quiver() draws arrow/vector field plot',
    'plt.contour(': '# plt.contour() draws contour lines (level curves)',
    'plt.contourf(': '# plt.contourf() draws filled contours',
    'plt.xlabel(': '# plt.xlabel() sets x-axis label',
    'plt.ylabel(': '# plt.ylabel() sets y-axis label',
    'plt.title(': '# plt.title() sets plot title',
    'plt.legend(': '# plt.legend() displays legend with labels',
    'plt.grid(': '# plt.grid() adds grid lines to plot',
    'plt.xlim(': '# plt.xlim() sets x-axis limits',
    'plt.ylim(': '# plt.ylim() sets y-axis limits',
    'plt.axhline(': '# plt.axhline() draws horizontal line across plot',
    'plt.axvline(': '# plt.axvline() draws vertical line across plot',
    'plt.show(': '# plt.show() displays the figure window',
    'plt.tight_layout(': '# plt.tight_layout() automatically adjusts subplot spacing',
    'plt.colorbar(': '# plt.colorbar() adds color scale bar to plot',
    'plt.text(': '# plt.text() adds text annotation at specified coordinates',
    'plt.arrow(': '# plt.arrow() draws arrow from (x,y) with specified dx, dy',
    'plt.gca()': '# plt.gca() gets current axes object',
    '.set_aspect(': '# .set_aspect() sets aspect ratio of axes',
    '.plot_surface(': '# .plot_surface() draws 3D surface plot',

    # SciPy
    'from scipy.integrate import odeint': 'from scipy.integrate import odeint  # ODE solver for initial value problems',
    'from scipy.integrate import solve_ivp': 'from scipy.integrate import solve_ivp  # Modern ODE solver with events',
    'from scipy.integrate import quad': 'from scipy.integrate import quad  # Numerical integration (quadrature)',
    'from scipy.integrate import solve_bvp': 'from scipy.integrate import solve_bvp  # Boundary value problem solver',
    'from scipy.optimize import minimize': 'from scipy.optimize import minimize  # Function minimization',
    'odeint(': '# odeint() solves ODE system with given initial conditions',
    'solve_ivp(': '# solve_ivp() solves ODE initial value problem (modern interface)',
    'quad(': '# quad() performs numerical integration using adaptive quadrature',

    # SymPy
    'import sympy as sp': 'import sympy as sp  # SymPy for symbolic mathematics',
    'from sympy import': 'from sympy import  # Import symbolic math functions',
    'symbols(': '# symbols() creates symbolic variables',
    'Matrix(': '# Matrix() creates symbolic matrix',
    'simplify(': '# simplify() algebraically simplifies expression',
    'expand(': '# expand() expands algebraic expression',
    'diff(': '# diff() computes symbolic derivative',

    # 3D plotting
    'from mpl_toolkits.mplot3d import Axes3D': 'from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit',
    "projection='3d'": "projection='3d'  # Create 3D axes",
}

def add_explanations_to_code_block(code_block):
    """Add inline explanations to a code block."""
    lines = code_block.split('\n')
    result = []

    for line in lines:
        # Skip if line already has a comment
        if '#' in line:
            result.append(line)
            continue

        # Check for known patterns
        explained = False
        for pattern, explanation in EXPLANATIONS.items():
            if pattern in line:
                # For import statements, replace the whole line
                if line.strip().startswith('import') or line.strip().startswith('from'):
                    result.append(explanation)
                else:
                    # Add comment at end of line
                    result.append(line + '  ' + explanation)
                explained = True
                break

        if not explained:
            result.append(line)

    return '\n'.join(result)

def process_lesson_file(filepath):
    """Process a single LESSON.md file to add code explanations."""
    print(f"Processing {filepath.name}...")

    with open(filepath, 'r') as f:
        content = f.read()

    # Find all Python code blocks
    pattern = r'```python\n(.*?)\n```'

    def replace_code_block(match):
        original_code = match.group(1)
        explained_code = add_explanations_to_code_block(original_code)
        return f'```python\n{explained_code}\n```'

    # Replace all code blocks
    new_content = re.sub(pattern, replace_code_block, content, flags=re.DOTALL)

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  ✓ Updated {filepath.name}")

def main():
    """Process all LESSON.md files in lessons directories."""
    base_dir = Path('/Users/clarkcarter/Claude/personal/gr/lessons')

    lesson_files = sorted(base_dir.glob('*/LESSON.md'))

    if not lesson_files:
        print("No LESSON.md files found!")
        return

    print(f"Found {len(lesson_files)} lesson files to process\n")

    for lesson_file in lesson_files:
        try:
            process_lesson_file(lesson_file)
        except Exception as e:
            print(f"  ✗ Error processing {lesson_file.name}: {e}")

    print(f"\n✓ Completed! Processed {len(lesson_files)} files")

if __name__ == '__main__':
    main()
