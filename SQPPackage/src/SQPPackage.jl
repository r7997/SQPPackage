module SQPPackage

# Laden notwendiger Pakete für lineare Algebra, Sparse-Matrizen, Formatierung usw.
using LinearAlgebra, SparseArrays, Printf #Plots
using NLPModels, ADNLPModels, QuadraticModels
using Parameters: @with_kw

# Import externer QP- und NLP-Solver
import Clarabel
import OSQP
import Ipopt
import MadNLP

# Einbinden der internen Moduldateien mit den jeweiligen Funktionalitäten
include("problem_types.jl")          # Definition von Problemtypen
include("settings.jl")               # Einstellungen und Parameter
include("stats.jl")                  # Statistiken und Statusinformationen
include("residuals.jl")              # Berechnung von Residuen
include("convexify.jl")              # Hessian-Konvexifizierung
include("qp_interface.jl")           # Schnittstelle zu QP-Solvern
include("filter_linesearch.jl")      # Filterbasierte Line-Search
include("sqp_method.jl")             # Haupt-SQP-Algorithmus
include("../testproblems/testproblems.jl")  # Testprobleme
include("ipopt_jump.jl")             # Nutzung von Ipopt über JuMP
include("../benchmarks/benchmarking.jl")           # Benchmarking-Funktionen

# Verwenden der enthaltenen Submodule
using .SettingsModule
using .StatsModule
using .ResidualsModule
using .ConvexifyModule
using .QPInterface
using .FilterLineSearchSQP
using .SQPMethod
using .SQPTestProblems
using .IpoptADNLP
using .Benchmarking

# Export der Hauptfunktionen des Solvers
export sqp_method, solve_with_ipopt_adnlp

# Export von Einstellungen und Konfiguration
export Settings

# Export von Statistiken und Statusdaten
export Stats, SQPStatus, kkt_point, max_iter, qp_failed, infeasible

# Export von Residuen-Funktionen
export get_primal_residual, get_dual_residual, compute_kkt_residuals

# Export von Hessian-Konvexifizierungsfunktionen
export convexify_hessian!, convexify_hessian

# Export der QP-Solver-Schnittstelle
export QPSolver, QPResult, create_qp_solver, solve_qp

# Export der Testprobleme
export create_P1, create_P2, create_P3, create_P4, create_P5, create_P6, create_P7, create_P8, create_P9, create_P10, create_P11, create_P12, create_P13, create_P14, create_P15, create_P16

# Export von Benchmarking-Werkzeugen
export performance_profile, run_benchmark, benchmark_solver

# Export von Filter- und Line-Search-Funktionen
export Filter, FilterPoint, constraint_violation, filter_line_search

end