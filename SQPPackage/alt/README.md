# SQPPackage.jl

A Julia package implementing a Sequential Quadratic Programming (SQP) method for solving nonlinear optimization problems.

## Overview

This package provides a comprehensive implementation of a filter-line-search SQP method that can solve difficult nonlinear optimization problems of the form:

```
min  f(x)
s.t. cℓ ≤ c(x) ≤ cu
     xℓ ≤ x ≤ xu
```

where f : ℝⁿ → ℝ and c : ℝⁿ → ℝᵐ are at least twice continuously differentiable functions.

## Key Features

- **Multiple QP Solvers**: Supports OSQP, Clarabel, Ipopt, and MadNLP for solving quadratic programming subproblems
- **Hessian Convexification**: Multiple strategies including Levenberg-Marquardt, projection, mirroring, and Gershgorin circles
- **Globalization Strategy**: Filter-line-search method with Second-Order Corrections (SOC)
- **Robust Implementation**: Handles ill-conditioned problems and provides comprehensive error handling
- **Benchmarking Tools**: Integrated performance profiling and comparison capabilities

## Installation

Since this is a local package, you can install it by navigating to the package directory and using:

```julia
using Pkg
Pkg.develop(path="./SQPPackage")
using SQPPackage
```

## Quick Start

Here's a simple example of how to use the SQP solver:

```julia
using SQPPackage, ADNLPModels

# Define a simple optimization problem
function f(x)
    return x[1]^2 + 2*x[2]^2
end

function c(x)
    return [x[1] + 2*x[2]^2 - 10.0, 2*x[1] + x[2] - 9.0]
end

# Create an ADNLPModel
x0 = [1.2, 5.4]
nlp = ADNLPModel(f, x0; c = c, lcon = [0.0, 0.0], ucon = [0.0, 0.0])

# Set up solver settings
settings = Settings(
    max_iter = 100,
    tol = 1e-6,
    verbose = true,
    qp_solver = :osqp,
    use_globalization = true
)

# Solve the problem
stats = sqp_method(nlp, settings)

# Check the results
println("Solver status: ", stats.sqp_status)
println("Solution: ", stats.x)
println("Objective value: ", stats.obj_val)
```

## Solver Configuration

The solver behavior can be customized through the `Settings` struct:

### QP Solver Selection
- `:osqp` - OSQP solver (default)
- `:clarabel` - Clarabel solver
- `:ipopt` - Ipopt solver
- `:madnlp` - MadNLP solver

### Hessian Convexification Methods
- `:lm` - Levenberg-Marquardt regularization (default)
- `:project` - Eigenvalue projection
- `:mirror` - Eigenvalue mirroring
- `:gershgorin` - Gershgorin circle regularization
- `:none` - No convexification

### Globalization Options
- `use_globalization = true` - Enable filter-line-search
- `use_soc = true` - Enable Second-Order Corrections

## Test Problems

The package includes 16 test problems (P1-P16) for validation and testing:

```julia
# Create and solve a test problem
prob = create_P1()  # Simple quadratic problem
stats = sqp_method(prob, Settings(verbose=true))
```

## Benchmarking

The package includes comprehensive benchmarking capabilities:

```julia
# Compare different QP solvers
include("benchmarks/compare_qp_solvers.jl")

# Compare Hessian convexification methods
include("benchmarks/compare_hessian_convexification.jl")

# Compare globalization strategies
include("benchmarks/compare_globalization.jl")

# Run large-scale benchmarks
include("benchmarks/large_benchmark.jl")
```

## Performance Profiling

The package supports performance profiling using the methodology from Dolan & Moré (2002):

```julia
# Generate performance profiles
times = run_benchmark(solvers, problems)
profile = performance_profile(times, solver_names)
```

## Package Structure

```
SQPPackage/
├── src/
│   ├── SQPPackage.jl          # Main package file
│   ├── sqp_method.jl          # Core SQP implementation
│   ├── settings.jl            # Configuration settings
│   ├── stats.jl               # Statistics and status tracking
│   ├── qp_interface.jl        # QP solver interface
│   ├── convexify.jl           # Hessian convexification methods
│   ├── filter_linesearch.jl   # Filter-line-search globalization
│   ├── residuals.jl           # KKT residual computation
│   ├── benchmarking.jl        # Benchmarking utilities
│   └── ...
├── testproblems/
│   └── testproblems.jl        # Test problem definitions
├── benchmarks/
│   ├── compare_qp_solvers.jl
│   ├── compare_hessian_convexification.jl
│   ├── compare_globalization.jl
│   └── large_benchmark.jl
├── results/                   # Benchmark results (PDFs)
├── README.md
└── min_example.jl            # Minimal working example
```

## Algorithm Details

The SQP method implemented in this package follows these key algorithmic components:

1. **Quadratic Programming Subproblem**: At each iteration, solve a QP that linearizes the constraints and uses a quadratic model of the Lagrangian
2. **Hessian Convexification**: Ensure the Hessian is positive definite using various regularization techniques
3. **Filter-Line-Search**: Use a filter method to ensure global convergence while maintaining fast local convergence
4. **Second-Order Corrections**: Apply SOC steps to improve convergence when the trial step is rejected

## Dependencies

- Julia ≥ 1.6
- LinearAlgebra, SparseArrays, Printf (standard library)
- NLPModels.jl, ADNLPModels.jl
- OptimizationProblems.jl (for benchmarking)
- OSQP.jl, Clarabel.jl, QuadraticModels.jl
- Ipopt.jl, MadNLP.jl
- Parameters.jl

## Authors

Developed as part of the TU Braunschweig optimization course.

## License

This package is provided for educational purposes.

## References

- Dolan, Elizabeth D., and Jorge J. Moré. "Benchmarking optimization software with performance profiles." Mathematical programming 91 (2002): 201–213.