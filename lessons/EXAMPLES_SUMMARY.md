# Example Scripts Summary - Lessons 6-9

## Overview
Created standalone, executable Python example scripts for lessons 6-9, following the Lesson 1 template structure.

## Structure
Each example script includes:
- Shebang (`#!/usr/bin/env python3`)
- Descriptive docstring
- Standalone, runnable code
- Educational print statements
- Colorblind-friendly visualizations
- Conceptual summaries

## Lesson 6: Tensors (4 examples)

### 01_tensor_transformations.py
**Topic:** Coordinate transformations (Cartesian ↔ Polar)
**Demonstrates:**
- Jacobian matrices for contravariant/covariant transformations
- Visual comparison of vector components in different coordinates
- Invariance of dot products under coordinate changes

### 02_metric_tensor.py
**Topic:** Metric tensor defining geometry
**Demonstrates:**
- Euclidean, Minkowski, polar, and spherical metrics
- Distance calculations using metrics
- Comparison of flat vs curved geometries
- Spacetime intervals (timelike, spacelike, null)

### 03_raising_lowering_indices.py
**Topic:** Index manipulation with metric tensor
**Demonstrates:**
- Converting contravariant ↔ covariant indices
- Sign flips in Minkowski space
- Musical isomorphism (♯ and ♭)
- Multi-index tensor transformations

### 04_electromagnetic_tensor.py
**Topic:** EM field tensor F_μν
**Demonstrates:**
- Constructing F_μν from E and B fields
- Antisymmetric tensor structure
- Electromagnetic invariants (B² - E²)
- Lorentz transformations mixing E and B
- Maxwell's equations in tensor form

## Lesson 7: Riemannian Geometry (4 examples)

### 01_christoffel_symbols.py
**Topic:** Connection coefficients
**Demonstrates:**
- Computing Christoffel symbols from metric
- Examples: polar coordinates, sphere surface
- Physical interpretation (basis vector changes)
- Symbolic + numerical calculations

### 02_covariant_derivative.py
**Topic:** Differentiation on curved manifolds
**Demonstrates:**
- Covariant vs ordinary derivative
- Parallel transport on sphere
- Divergence in polar coordinates
- Christoffel correction terms

### 03_riemann_curvature_tensor.py
**Topic:** Curvature measurement
**Demonstrates:**
- Computing Riemann tensor from Christoffel symbols
- Sphere (K=1) vs flat plane (K=0)
- Gaussian curvature
- Parallel transport detecting curvature
- Symmetries and contractions

### 04_geodesics_sphere.py
**Topic:** Shortest paths in curved space
**Demonstrates:**
- Geodesic equation integration
- Great circles on sphere
- Meridians, equator, general great circles
- Non-geodesic paths (latitude circles)
- Comparison with straight lines

## Lesson 8: Classical Mechanics (4 examples)

### 01_lagrangian_mechanics.py
**Topic:** Lagrangian formulation and action principle
**Demonstrates:**
- L = T - V for harmonic oscillator and pendulum
- Euler-Lagrange equations
- Principle of least action
- Comparing true path vs varied paths
- Energy conservation

### 02_hamiltonian_mechanics.py
**Topic:** Phase space formulation
**Demonstrates:**
- H = T + V (Hamiltonian)
- Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
- Phase space trajectories (ellipses for SHO)
- Energy conservation
- Lagrangian vs Hamiltonian comparison

### 03_noethers_theorem.py
**Topic:** Symmetries and conservation laws
**Demonstrates:**
- Time translation → Energy conservation
- Space translation → Momentum conservation
- Free particle example
- Noether's theorem applications

### 04_action_principle.py
**Topic:** Variational calculus
**Demonstrates:**
- Action S = ∫L dt for projectile motion
- Parabolic trajectory minimizes action
- Comparing action for different paths
- Hamilton's principle

## Lesson 9: Special Relativity (4 examples)

### 01_lorentz_transformations.py
**Topic:** Spacetime coordinate transformations
**Demonstrates:**
- Lorentz boost matrices Λ^μ_ν
- Lorentz factor γ = 1/√(1-v²/c²)
- Event transformations between frames
- Spacetime diagrams with worldlines
- Invariance of light speed

### 02_time_dilation.py
**Topic:** Relativistic time and length effects
**Demonstrates:**
- Time dilation: Δt' = γΔt₀
- Length contraction: L = L₀/γ
- Muon decay example
- Spacecraft contraction
- Twin paradox spacetime diagram

### 03_spacetime_diagrams.py
**Topic:** Minkowski diagrams and causality
**Demonstrates:**
- Light cones (future, past, elsewhere)
- Causal structure of spacetime
- Timelike, spacelike, null intervals
- Different reference frame axes
- Causality protection

### 04_four_vectors.py
**Topic:** 4-vector formalism
**Demonstrates:**
- 4-velocity u^μ = γ(1, v/c)
- 4-momentum p^μ = (E/c, p⃗)
- E = γmc², p = γmv
- E² = (pc)² + (mc²)² relation
- Energy-momentum diagrams
- Mass shell condition

## Key Features

### Educational Quality
- Progressive complexity within each lesson
- Clear explanations with physical intuition
- Connection to General Relativity highlighted
- Practice-oriented with numerical examples

### Visualization
- All plots use colorblind-friendly colors from `utils/colorblind_colors.py`
- Multiple subplot layouts showing different aspects
- 3D visualizations where appropriate (spheres, surfaces)
- Annotated diagrams with labels and legends

### Code Quality
- Standalone and executable
- No external file dependencies
- Proper imports with path handling
- Docstrings and inline comments
- Print statements explain calculations

### Consistency
- All scripts follow same structure as Lesson 1 examples
- Numbered sequentially (01, 02, 03, 04)
- Clear file naming convention
- Executable permissions set

## Usage

Run any example directly:
```bash
cd /Users/clarkcarter/Claude/personal/gr
source venv/bin/activate
python lessons/06_tensors/examples/01_tensor_transformations.py
```

Or make executable and run:
```bash
./lessons/06_tensors/examples/01_tensor_transformations.py
```

## Total Count
- **16 standalone example scripts**
- **4 per lesson** (Lessons 6-9)
- All scripts are executable and documented
- All use colorblind-friendly visualizations

## Topics Coverage

### Lesson 6 - Tensors
- Coordinate transformations
- Metric tensor
- Index raising/lowering
- EM field tensor

### Lesson 7 - Riemannian Geometry
- Christoffel symbols
- Covariant derivatives
- Riemann curvature
- Geodesics

### Lesson 8 - Classical Mechanics
- Lagrangian mechanics
- Hamiltonian mechanics
- Noether's theorem
- Action principle

### Lesson 9 - Special Relativity
- Lorentz transformations
- Time dilation
- Spacetime diagrams
- 4-vectors

## Next Steps
Students can:
1. Read LESSON.md files while running examples in Python terminal
2. Modify example parameters to explore concepts
3. Use examples as templates for quiz questions
4. Reference for GR foundations (Lessons 10-12)
