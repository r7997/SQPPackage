module SQPMethod

using LinearAlgebra, SparseArrays, Printf
using NLPModels, ADNLPModels
using Parameters: @with_kw

# Interne Module importieren
import ..ProblemTypes: AbstractNLPModel
import ..SettingsModule: Settings
import ..StatsModule: Stats, SQPStatus, kkt_point, max_iter, qp_failed, infeasible
import ..ConvexifyModule: convexify_hessian!
import ..FilterLineSearchSQP: Filter, FilterPoint, constraint_violation, filter_line_search,
                             add_to_filter!, is_acceptable_to_filter
import ..ResidualsModule: compute_kkt_residuals
import ..QPInterface: create_qp_solver, solve_qp, AbstractQPSolver, QPResult
include("benchmarking.jl")
using .Benchmarking

export sqp_method

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

# Second Order Correction (SOC) Resultat
struct SOCResult
    d_soc::Vector{Float64}
    success::Bool
    qp_status::Symbol
end

# Druckfunktion: Iterations-Header
function print_iteration_header()
    @printf("================================================================================\n")
    @printf("%-4s %-12s %-10s %-10s %-10s %-10s %-6s %-4s %-2s\n",
            "iter", "objective", "inf_pr", "inf_du", "inf_comp", "step_norm", "alpha", "type", "ls")
    @printf("--------------------------------------------------------------------------------\n")
end

# Druckfunktion: Iterations-Details
function print_iteration(k, f, inf_pr, inf_du, inf_comp, d_norm, alpha, step_type, ls)
    @printf("%-4d %-12.4e %-10.2e %-10.2e %-10.2e %-10.2e %-6.3f %-4s %2d\n",
        k, f, inf_pr, inf_du, inf_comp, d_norm, alpha, step_type, ls)
end

# Abschluss-Status-Druck
function print_final_status(stats::Stats)
    println("="^80)
    println("Finaler Lösungssummary")
    println("-"^80)
    @printf("* Objective = %.4e\n", stats.obj_val)
    @printf("* Primal feasibility = %.4e\n", stats.inf_pr)
    @printf("* Dual feasibility = %.4e\n", stats.inf_du)
    @printf("* Residual = %.4e\n", stats.inf_comp)
    @printf("* Step norm = %.4e\n", !isempty(stats.iteration_data) ? stats.iteration_data[end].step_norm : 0.0)
    @printf("* SQP Solver-Status = %s\n", stats.sqp_status)
    @printf("* Letzter QP-Lösungsstatus = %s\n", stats.qp_solve_status)
    println("-"^80)
    println("Performance Statistik")
    println("-"^80)
    @printf("* Objective evals = %d\n", stats.obj_evals)
    @printf("* Grad evals = %d\n", stats.grad_evals)
    @printf("* Constraint evals = %d\n", stats.cons_evals)
    @printf("* Jacobian evals = %d\n", stats.jac_evals)
    @printf("* Hessian evals = %d\n", stats.hess_evals)
    @printf("* Totalzeit in SQP = %.3f Sekunden\n", stats.total_time)
    println("-"^80)
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

end # module SQPMethod