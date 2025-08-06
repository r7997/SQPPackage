module SQPMethod

using LinearAlgebra, SparseArrays, Printf
using NLPModels, ADNLPModels
using Parameters: @with_kw

# Externe Solver und interne Module importieren
import Clarabel
import OSQP
import QuadraticModels
import Ipopt
import MadNLP

import ..ProblemTypes: AbstractNLPModel
import ..SettingsModule: Settings
import ..StatsModule: Stats, SQPStatus, kkt_point, max_iter, qp_failed, infeasible
import ..ConvexifyModule: convexify_hessian!
import ..FilterLineSearchSQP: Filter, FilterPoint, constraint_violation, filter_line_search,
                             add_to_filter!, is_acceptable_to_filter
import ..ResidualsModule: compute_kkt_residuals

export sqp_method, AbstractQPSolver, create_qp_solver, solve_qp

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

# QP lösen Funktion
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
function create_clarabel_solver(n::Int, m::Int, settings::Settings)
    try
        solver_settings = Clarabel.Settings(; settings.clarabel_settings...)
        # Beispielstruktur: dummy-Daten (werden später überschrieben)
        P = spzeros(n, n)
        q = zeros(n)
        A = spzeros(0, n)
        b = Float64[]
        cones = Clarabel.ZeroConeT[]  # Leerer Cone für Initialisierung

        solver = Clarabel.Solver(P, q, A, b, cones; settings = solver_settings)

        problem_data = Dict{String, Any}()  # Leeres Dict
        return ClarabelSolver(solver, solver_settings, problem_data, n, m)
    catch e
        error("Fehler beim Erstellen Clarabel-Solver: $e")
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

# Druckfunktion: Iterations-Header
function print_iteration_header()
    @printf("================================================================================\n")
    @printf("%-4s %-12s %-10s %-10s %-10s %-10s %-6s %-4s %-2s
",
            "iter", "objective", "inf_pr", "inf_du", "inf_comp", "step_norm", "alpha", "type", "ls")
    @printf("--------------------------------------------------------------------------------\n")
end

# Druckfunktion: Iterations-Details
function print_iteration(k, f, inf_pr, inf_du, inf_comp, d_norm, alpha, step_type, ls)
    @printf("%-4d %-12.4e %-10.2e %-10.2e %-10.2e %-10.2e %-6.3f %-4s %2d
",
        k, f, inf_pr, inf_du, inf_comp, d_norm, alpha, step_type, ls)
end

# Abschluss-Status-Druck
function print_final_status(stats::Stats)
    println("="^80)
    println("Finaler Lösungssummary")
    println("-"^80)
    @printf("* Objective = %.4e
", stats.obj_val)
    @printf("* Primal feasibility = %.4e
", stats.inf_pr)
    @printf("* Dual feasibility = %.4e
", stats.inf_du)
    @printf("* Residual = %.4e
", stats.inf_comp)
    @printf("* Step norm = %.4e
", !isempty(stats.iteration_data) ? stats.iteration_data[end].step_norm : 0.0)
    @printf("* SQP Solver-Status = %s
", stats.sqp_status)
    @printf("* Letzter QP-Lösungsstatus = %s
", stats.qp_solve_status)
    println("-"^80)
    println("Performance Statistik")
    println("-"^80)
    @printf("* Objective evals = %d
", stats.obj_evals)
    @printf("* Grad evals = %d
", stats.grad_evals)
    @printf("* Constraint evals = %d
", stats.cons_evals)
    @printf("* Jacobian evals = %d
", stats.jac_evals)
    @printf("* Hessian evals = %d
", stats.hess_evals)
    @printf("* Totalzeit in SQP = %.3f Sekunden
", stats.total_time)
    println("-"^80)
end

# Second Order Correction (SOC) Resultat
struct SOCResult
    d_soc::Vector{Float64}
    success::Bool
    qp_status::Symbol
end

# SOC-Schritt berechnen
function compute_soc_step(nlp::AbstractNLPModel, x::Vector{Float64}, d::Vector{Float64},
                          y::Vector{Float64}, qp_solver::AbstractQPSolver, settings::Settings)
    n = nlp.meta.nvar
    m = nlp.meta.ncon
    try
        x_trial = x + d
        c_trial = cons(nlp, x_trial)
        A = jac(nlp, x)
        b_soc = -c_trial

        H = hess(nlp, x, y)
        H_reg = convexify_hessian!(copy(H), settings)  # Hessian verfeinern

        g = grad(nlp, x)

        lvar_soc = nlp.meta.lvar - x_trial
        uvar_soc = nlp.meta.uvar - x_trial
        lcon_soc = m > 0 ? b_soc : Float64[]
        ucon_soc = m > 0 ? b_soc : Float64[]

        result = solve_qp(qp_solver, H_reg, g, A, lcon_soc, ucon_soc, lvar_soc, uvar_soc, settings)

        if result.success
            return SOCResult(result.x, true, result.status)
        end
    catch e
        @warn "SOC-Update fehlgeschlagen: $e"
    end
    return SOCResult(zeros(n), false, :Error)  # Fehlerfall
end

# Haupt-SQP-Methode
function sqp_method(nlp::ADNLPModel, settings::Settings=Settings())
    n = nlp.meta.nvar
    m = nlp.meta.ncon
    x_k = copy(nlp.meta.x0)
    y_k = zeros(m)
    z_k = zeros(n)

    qp_solver = create_qp_solver(n, m, settings)  # QP Solver erstellen
    filter = Filter()
    # Statistiks initialisieren
    stats = Stats(
        copy(x_k), copy(y_k), copy(z_k),
        obj(nlp, x_k), 0.0, 0.0, 0.0,
        max_iter, max_iter, 0
    )
    stats.obj_evals = 1
    stats.cons_evals = m > 0 ? 1 : 0
    start_time = time()

    k = 0
    converged = false
    tol = settings.tol
    max_iterations = settings.max_iter
    soc_improvement_threshold = getfield(settings, :soc_improvement_threshold)::Float64

    # Iterationsausgabe, falls aktiviert
    settings.verbose && print_iteration_header()

    while k < max_iterations && !converged
        f_k = obj(nlp, x_k)  # Zielwert
        g_k = grad(nlp, x_k)   # Gradient
        c_k = m > 0 ? cons(nlp, x_k) : Float64[]  # Nebenbedingungen
        J_k = m > 0 ? jac(nlp, x_k) : zeros(0, n)  # Jacobi-Matrix
        H_k = hess(nlp, x_k, y_k)  # Hessian

        # Residuen berechnen
        primal_res, dual_res, comp_res = compute_kkt_residuals(nlp, x_k, y_k, z_k)

        # Abbruch bei Konvergenz
        if primal_res < tol && dual_res < tol && comp_res < tol
            stats.sqp_status = kkt_point
            converged = true
            break
        end

        # QP-Rahmen: Variablen- und Constraints-Grenzen
        lvar_qp = nlp.meta.lvar - x_k
        uvar_qp = nlp.meta.uvar - x_k
        lcon_qp = m > 0 ? nlp.meta.lcon - c_k : Float64[]
        ucon_qp = m > 0 ? nlp.meta.ucon - c_k : Float64[]

        H_k = convexify_hessian!(copy(H_k), settings)  # Hessian verfeinern

        # QP-Lösung
        qp_result = solve_qp(qp_solver, H_k, g_k, J_k, lcon_qp, ucon_qp, lvar_qp, uvar_qp, settings)
        stats.qp_solve_status = qp_result.status

        if !qp_result.success
            stats.sqp_status = qp_failed
            @warn "QP-Lösung fehlgeschlagen bei Iteration $k"
            break
        end

        # Schritt und Multiplikatoren
        d_k = qp_result.x
        y_qp = qp_result.y
        z_L_qp = qp_result.zL
        z_U_qp = qp_result.zU

        d_norm = norm(d_k, Inf)
        h_k = constraint_violation(nlp, x_k)

        # Line Search
        alpha, step_type, ls_count = filter_line_search(
            nlp, x_k, d_k, y_k, f_k, h_k, g_k, filter, settings
        )
        stats.obj_evals += ls_count
        stats.cons_evals += ls_count * (m > 0 ? 1 : 0)

        # SOC-Update, falls aktiviert und sinnvoll
        if settings.use_soc && alpha < 0.5
            soc_result = compute_soc_step(nlp, x_k, d_k, y_k, qp_solver, settings)
            if soc_result.success
                alpha_soc, step_type_soc, ls_count_soc = filter_line_search(
                    nlp, x_k, soc_result.d_soc, y_k, f_k, h_k, g_k, filter, settings
                )
                if alpha_soc > soc_improvement_threshold * alpha
                    alpha = alpha_soc
                    step_type = step_type_soc * "-soc"
                    d_k = soc_result.d_soc
                    stats.obj_evals += ls_count_soc
                    stats.cons_evals += ls_count_soc * (m > 0 ? 1 : 0)
                end
            end
        end

        # Iterationsdaten speichern
        push!(stats.iteration_data, IterationData(
            k, f_k, primal_res, dual_res, comp_res, d_norm, alpha, step_type, ls_count
        ))

        # Schritt durchführen
        x_k .+= alpha .* d_k
        if m > 0
            y_k .= (1 - alpha) .* y_k .+ alpha .* y_qp
        end
        if length(z_L_qp) > 0 && length(z_U_qp) > 0
            z_k .= (1 - alpha) .* z_k .+ alpha .* (z_U_qp - z_L_qp)
        end

        # Ausgabe
        if settings.verbose
            print_iteration(k, f_k, primal_res, dual_res, comp_res, d_norm, alpha, step_type, ls_count)
        end

        k += 1
    end

    # Endstatus setzen
    stats.x .= x_k
    stats.y .= y_k
    stats.z .= z_k
    stats.iter = k
    stats.obj_val = obj(nlp, x_k)
    stats.obj_evals += 1

    # Endresiduen
    primal_res, dual_res, comp_res = compute_kkt_residuals(nlp, x_k, y_k, z_k)
    stats.inf_pr = primal_res
    stats.inf_du = dual_res
    stats.inf_comp = comp_res
    stats.total_time = time() - start_time

    # Max. Iteration
    if !converged && k >= max_iterations
        stats.sqp_status = max_iter
    end

    # Abschlussausgabe
    settings.verbose && print_final_status(stats)

    return stats
end

end # Modul