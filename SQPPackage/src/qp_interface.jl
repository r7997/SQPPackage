module QPInterface

using LinearAlgebra, SparseArrays, Printf
import Clarabel
import OSQP
import QuadraticModels
import Ipopt
import MadNLP

using ..SettingsModule: Settings

# Exportiert die Hauptfunktionen und Datentypen dieses Moduls
export solve_qp, create_qp_solver, AbstractQPSolver, QPResult
export ClarabelSolver, OSQPSolver, QuadraticModelsSolver

# Abstrakte QP Solver Schnittstelle
abstract type AbstractQPSolver end

# Clarabel Solver-Typ
struct ClarabelSolver <: AbstractQPSolver
    solver::Any
    settings::Any
    problem_data::Dict{String, Any}
    n::Int
    m::Int
end

# OSQP Solver-Typ
struct OSQPSolver <: AbstractQPSolver
    solver::Any
    settings::Any
    problem_data::Dict{String, Any}
    n::Int
    m::Int
end

# QuadraticModels Solver-Typ
struct QuadraticModelsSolver <: AbstractQPSolver
    solver_type::Symbol
    qp_model::Any
    solver::Any
    n::Int
    m::Int
end

# QP Ergebnisstruktur
struct QPResult
    x::Vector{Float64}
    y::Vector{Float64}
    zL::Vector{Float64}
    zU::Vector{Float64}
    status::Symbol
    success::Bool
end

# Factory-Funktion für QP Solver (nach Typ in den Einstellungen)
function create_qp_solver(n::Int, m::Int, settings::Settings)
    solver_type = settings.qp_solver
    if solver_type == :clarabel
        return create_clarabel_solver(n, m, settings)  # Clarabel erstellen
    elseif solver_type == :osqp
        return create_osqp_solver(n, m, settings)      # OSQP erstellen
    elseif solver_type == :ipopt
        return create_quadratic_models_solver(n, m, :ipopt, settings)  # Ipopt
    elseif solver_type == :madnlp
        return create_quadratic_models_solver(n, m, :madnlp, settings) # MadNLP
    else
        error("Nicht unterstützter QP-Solver-Typ in den Einstellungen: $solver_type")  # Fehler
    end
end

# Clarabel Solver erstellen
function create_clarabel_solver(n::Int, m::Int, settings::Settings)
    try
        solver_settings = Clarabel.Settings(; settings.clarabel_settings...)
        # Beispielstruktur: dummy-Daten (werden später überschrieben)
        P = spzeros(n, n)
        q = zeros(n)
        A = spzeros(0, n)
        b = Float64[]
        cones = Clarabel.Cone[]  # Leerer Cone-Array für Initialisierung

        # Clarabel Solver mit positional arguments erstellen
        solver = Clarabel.Solver()
        Clarabel.setup!(solver, P, q, A, b, cones, solver_settings)

        problem_data = Dict{String, Any}()  # Leeres Dict
        return ClarabelSolver(solver, solver_settings, problem_data, n, m)
    catch e
        error("Fehler beim Erstellen Clarabel-Solver: $e")
    end
end

# OSQP Solver erstellen
function create_osqp_solver(n::Int, m::Int, settings::Settings)
    try
        solver = OSQP.Model()  # OSQP Modell
        osqp_settings = settings.osqp_settings
        # Problem-Daten initialisieren
        problem_data = Dict{String, Any}(
            "verbose" => osqp_settings["verbose"],
            "max_iter" => osqp_settings["max_iter"],
            "eps_abs" => osqp_settings["eps_abs"],
            "eps_rel" => osqp_settings["eps_rel"],
            "initialized" => false
        )
        return OSQPSolver(solver, nothing, problem_data, n, m)
    catch e
        error("Fehler beim Erstellen des OSQP-Solvers: $e")
    end
end

# QuadraticModels Solver erstellen
function create_quadratic_models_solver(n::Int, m::Int, solver_type::Symbol, settings::Settings)
    try
        qp_model = QuadraticModels.QuadraticModel(
            zeros(n), zeros(n,n), zeros(0,n), zeros(0), zeros(0), zeros(n), zeros(n), name="SQP_QP"
        )
        if solver_type == :ipopt
            solver = Ipopt.IpoptSolver(qp_model)
            solver.options[:hessian_constant] = true
            solver.options[:jacobian_constant] = true
            solver.options[:print_level] = 0
        elseif solver_type == :madnlp
            solver = MadNLP.MadNLPSolver(qp_model)
            solver.options[:hessian_constant] = true
            solver.options[:jacobian_constant] = true
            solver.options[:print_level] = 0
        end
        return QuadraticModelsSolver(solver_type, qp_model, solver, n, m)
    catch e
        @warn "Fehler beim Erstellen $solver_type: $e"
        error("Fehler beim Erstellen $solver_type: $e")
    end
end

# Transformation zu Clarabel Format
function transform_to_clarabel_format(H, g, J, lcon, ucon, lvar, uvar)
    n = length(g)
    m_con = length(lcon)
    P = sparse(H)  # Hessian in Sparse-Format
    q = g          # Gradient

    # Gleichungs- und Ungleichungs-Constraints initialisieren
    A_eq = spzeros(0, n)
    b_eq = Float64[]
    A_ineq = spzeros(0, n)
    b_ineq = Float64[]
    
    # Gleichheits- und Ungleichheits-Constraints
    for i in 1:m_con
        if lcon[i] == ucon[i]
            A_eq = [A_eq; J[i:i, :]]
            b_eq = [b_eq; lcon[i]]
        else
            if lcon[i] > -Inf
                A_ineq = [A_ineq; -J[i:i, :]]
                b_ineq = [b_ineq; -lcon[i]]
            end
            if ucon[i] < Inf
                A_ineq = [A_ineq; J[i:i, :]]
                b_ineq = [b_ineq; ucon[i]]
            end
        end
    end

    # Variablen-Grenzen in Ungleichung umwandeln
    for i in 1:n
        if lvar[i] > -Inf
            A_ineq = [A_ineq; -sparse(1, n, [1.0], 1, n)]
            b_ineq = [b_ineq; -lvar[i]]
        end
        if uvar[i] < Inf
            A_ineq = [A_ineq; sparse(1, n, [1.0], 1, n)]
            b_ineq = [b_ineq; uvar[i]]
        end
    end

    return P, q, A_eq, b_eq, A_ineq, b_ineq  # Rückgabe
end

# Transformation zu OSQP Format
function transform_to_osqp_format(H, g, J, lcon, ucon, lvar, uvar)
    n = length(g)
    m_con = length(lcon)
    P = sparse(H)
    q = g

    # Constraints kombinieren
    A_constraints = [J; sparse(I, n, n)]
    l_constraints = [lcon; lvar]
    u_constraints = [ucon; uvar]

    return P, q, A_constraints, l_constraints, u_constraints  # Rückgabe
end

# QP lösen Funktion (Dispatch basierend auf Solver-Typ)
function solve_qp(qp_solver::AbstractQPSolver, H, g, J, lcon, ucon, lvar, uvar, settings)
    if isa(qp_solver, ClarabelSolver)
        return solve_qp_clarabel(qp_solver, H, g, J, lcon, ucon, lvar, uvar, settings)
    elseif isa(qp_solver, OSQPSolver)
        return solve_qp_osqp(qp_solver, H, g, J, lcon, ucon, lvar, uvar, settings)
    elseif isa(qp_solver, QuadraticModelsSolver)
        return solve_qp_quadratic_models(qp_solver, H, g, J, lcon, ucon, lvar, uvar, settings)
    else
        error("Unbekannter QP-Solver Typ")
    end
end

# Clarabel QP Lösung
function solve_qp_clarabel(qp_solver::ClarabelSolver, H, g, J, lcon, ucon, lvar, uvar, settings)
    try
        P, q, A_eq, b_eq, A_ineq, b_ineq = transform_to_clarabel_format(H, g, J, lcon, ucon, lvar, uvar)
        
        # Clarabel Format erstellen
        A = [A_eq; A_ineq]
        b = [b_eq; b_ineq]
        
        # Cone-Definitionen
        cones = Clarabel.Cone[]
        if length(b_eq) > 0
            push!(cones, Clarabel.ZeroConeT(length(b_eq)))
        end
        if length(b_ineq) > 0
            push!(cones, Clarabel.NonnegativeConeT(length(b_ineq)))
        end
        
        # Solver neu konfigurieren für dieses Problem
        Clarabel.setup!(qp_solver.solver, P, q, A, b, cones, qp_solver.settings)
        Clarabel.solve!(qp_solver.solver)
        
        result = Clarabel.solution(qp_solver.solver)
        status_info = Clarabel.solver_status(qp_solver.solver)
        
        if status_info == Clarabel.SOLVED
            n, m = qp_solver.n, qp_solver.m
            y = length(result.z) >= m ? result.z[1:m] : zeros(m)
            zL = zeros(n)
            zU = zeros(n)
            return QPResult(result.x, y, zL, zU, :Solved, true)
        else
            return QPResult(zeros(qp_solver.n), zeros(qp_solver.m), zeros(qp_solver.n), zeros(qp_solver.n), :Failed, false)
        end
    catch e
        @warn "Clarabel-Fehler: $e"
        return QPResult(zeros(qp_solver.n), zeros(qp_solver.m), zeros(qp_solver.n), zeros(qp_solver.n), :Error, false)
    end
end

# OSQP QP Lösung
function solve_qp_osqp(qp_solver::OSQPSolver, H, g, J, lcon, ucon, lvar, uvar, settings)
    try
        P, q, A, l, u = transform_to_osqp_format(H, g, J, lcon, ucon, lvar, uvar)
        # Setup nur beim ersten Mal
        if !qp_solver.problem_data["initialized"]
            verbose = qp_solver.problem_data["verbose"]
            max_iter = qp_solver.problem_data["max_iter"]
            eps_abs = qp_solver.problem_data["eps_abs"]
            eps_rel = qp_solver.problem_data["eps_rel"]
            OSQP.setup!(qp_solver.solver;
                P=sparse(triu(P)),
                q=q,
                A=sparse(A),
                l=l,
                u=u,
                verbose=verbose,
                max_iter=max_iter,
                eps_abs=eps_abs,
                eps_rel=eps_rel
            )
            qp_solver.problem_data["initialized"] = true
        else
            # Update bei späteren Aufrufen
            P_upper = triu(sparse(P))
            OSQP.update!(qp_solver.solver;
                Px=nonzeros(P_upper),
                Ax=nonzeros(sparse(A)),
                q=q,
                l=l,
                u=u
            )
        end
        results = OSQP.solve!(qp_solver.solver)
        if results.info.status == :Solved
            y = results.y[1:length(lcon)]
            zL = max.(0, -results.y[length(lcon)+1:end])
            zU = max.(0, results.y[length(lcon)+1:end])
            return QPResult(results.x, y, zL, zU, :Solved, true)
        else
            return QPResult(zeros(qp_solver.n), zeros(qp_solver.m), zeros(qp_solver.n), zeros(qp_solver.n), :Failed, false)
        end
    catch e
        @warn "OSQP-Fehler: $e"
        return QPResult(zeros(qp_solver.n), zeros(qp_solver.m), zeros(qp_solver.n), zeros(qp_solver.n), :Error, false)
    end
end

# QuadraticModels Lösung
function solve_qp_quadratic_models(qp_solver::QuadraticModelsSolver, H, g, J, lcon, ucon, lvar, uvar, settings)
    try
        n, m = qp_solver.n, qp_solver.m
        qp_model = QuadraticModels.QuadraticModel(
            g, H, J, lcon, ucon, lvar, uvar, name="SQP_QP"
        )
        if qp_solver.solver_type == :ipopt
            solver = Ipopt.IpoptSolver(qp_model)
            result = Ipopt.solve!(solver, qp_model)
        elseif qp_solver.solver_type == :madnlp
            solver = MadNLP.MadNLPSolver(qp_model)
            result = MadNLP.solve!(solver, qp_model)
        else
            error("Unbekannter Solver-Typ für QuadraticModels")
        end

        if result.status == :first_order || result.status == :acceptable
            y = result.multipliers[1:m]
            zL = max.(0, -result.multipliers_L)
            zU = max.(0, result.multipliers_U)
            return QPResult(result.solution, y, zL, zU, :Solved, true)
        else
            return QPResult(zeros(n), zeros(m), zeros(n), zeros(n), :Failed, false)
        end
    catch e
        @warn "QuadraticModels-Fehler: $e"
        return QPResult(zeros(qp_solver.n), zeros(qp_solver.m), zeros(qp_solver.n), zeros(qp_solver.n), :Error, false)
    end
end

# Hilfsfunktion: Constraint-Multipliers extrahieren (Clarabel)
function extract_constraint_multipliers(z, num_eq, num_ineq)
    y = zeros(Float64, num_eq + 2 * num_ineq)
    return y
end

# Hilfsfunktion: Variable bounds Multipliers
function extract_bound_multipliers(z, total_constraints, num_eq)
    zL = zeros(Float64, total_constraints - num_eq)
    zU = zeros(Float64, total_constraints - num_eq)
    return zL, zU
end

end # module QPInterface