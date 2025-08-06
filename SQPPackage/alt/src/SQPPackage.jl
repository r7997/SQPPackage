module SQPPackage

using LinearAlgebra, SparseArrays, Printf
using NLPModels, ADNLPModels, QuadraticModels
using Parameters: @with_kw
using Statistics 
using CSV, DataFrames
using Plots

# Externe Solver
import Clarabel
import OSQP
import Ipopt
import MadNLP

# Enum für SQP-Status
@enum SQPStatus begin
    kkt_point        # KKT-Punkt erreicht
    max_iter         # Maximale Iterationen erreicht
    qp_failed        # QP-Lösung fehlgeschlagen
    infeasible       # Problem unlösbar
    restoration_phase # Wiederherstellungsphase aktiv
end

# Stats-Struktur
mutable struct Stats
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    obj_val::Float64
    inf_pr::Float64
    inf_du::Float64
    inf_comp::Float64
    qp_status::SQPStatus
    sqp_status::SQPStatus
    iter::Int
    obj_evals::Int
    grad_evals::Int
    cons_evals::Int
    jac_evals::Int
    hess_evals::Int
    qp_solve_status::Symbol
    total_time::Float64
    iteration_data::Vector{Any}
    
    # Konstruktor
    function Stats(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}, 
                   obj_val::Float64, inf_pr::Float64, inf_du::Float64, inf_comp::Float64,
                   qp_status::SQPStatus, sqp_status::SQPStatus, iter::Int)
        new(x, y, z, obj_val, inf_pr, inf_du, inf_comp, qp_status, sqp_status, iter,
            0, 0, 0, 0, 0, :unknown, 0.0, [])
    end
end

# Iterationsdaten-Struktur
struct IterationData
    iter::Int
    objective::Float64
    inf_pr::Float64
    inf_du::Float64
    inf_comp::Float64
    step_norm::Float64
    alpha::Float64
    type::String
    ls::Int
end

# Einbinden der Moduldateien
include("problem_types.jl")
include("settings.jl")
include("residuals.jl")
include("convexify.jl")
include("qp_interface.jl")
include("filter_linesearch.jl")
include("sqp_method.jl")
include("test_problems.jl")
include("ipopt_jump.jl")
include("benchmarking.jl")

# Exporte
export sqp_method, solve_with_ipopt_adnlp, Settings, Stats, SQPStatus, 
       kkt_point, max_iter, qp_failed, infeasible, IterationData,
       get_primal_residual, get_dual_residual, compute_kkt_residuals,
       convexify_hessian!, convexify_hessian, QPSolver, QPResult, create_qp_solver, 
       solve_qp, create_P1, create_P2, create_P3, create_P4, create_P5, 
       create_P6, create_P7, create_P8, create_P9, create_P10, create_P11, 
       create_P12, create_P13, create_P14, create_P15, create_P16,
       performance_profile, run_benchmark, benchmark_solver,
       Filter, FilterPoint, constraint_violation, filter_line_search

end