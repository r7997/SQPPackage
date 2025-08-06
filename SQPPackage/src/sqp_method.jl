"""
SQPMethod - Kernimplementierung der Sequential Quadratic Programming Methode

Dieses Modul enthält die Hauptschleife des SQP-Algorithmus mit:
- Filter-Liniensuche für globale Konvergenz
- Second-Order Corrections (SOC) für bessere lokale Konvergenz
- Multiple Hessian-Konvexifizierungsstrategien
- Robuste Fehlerbehandlung und Iteration-Tracking
"""
module SQPMethod

# Standard-Bibliotheken für numerische Algebra und Formatierung
using LinearAlgebra, SparseArrays, Printf
# NLP-Modellierung
using NLPModels, ADNLPModels
# Parameter-Management
using Parameters: @with_kw

# Interne Modulimporte - alle Kernkomponenten des SQP-Lösers
import ..ProblemTypes: AbstractNLPModel                    # Problemdefinitionen
import ..SettingsModule: Settings                          # Konfigurationsparameter
import ..StatsModule: Stats, SQPStatus, kkt_point, max_iter, qp_failed, infeasible  # Statistiken
import ..ConvexifyModule: convexify_hessian!               # Hessian-Regularisierung
import ..FilterLineSearchSQP: Filter, FilterPoint, constraint_violation, filter_line_search,
                             add_to_filter!, is_acceptable_to_filter  # Globalisierungsstrategie
import ..ResidualsModule: compute_kkt_residuals            # KKT-Residuenberechnung
import ..QPInterface: create_qp_solver, solve_qp, AbstractQPSolver, QPResult  # QP-Löser-Interface
#include("benchmarking.jl")
import ..Benchmarking

export sqp_method  # Hauptfunktion exportieren

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

# Druckt die Kopfzeile für die Iterationsausgabe des SQP-Algorithmus
function print_iteration_header()
    @printf("================================================================================\n")
    @printf("%-4s %-12s %-10s %-10s %-10s %-10s %-6s %-4s %-2s\n",
            "iter",      # Iterationsnummer
            "objective", # Zielfunktionswert
            "inf_pr",    # Primale Unzulässigkeit
            "inf_du",    # Duale Unzulässigkeit  
            "inf_comp",  # Komplementaritätsfehler
            "step_norm", # Norm des Suchschritts
            "alpha",     # Schrittweite
            "type",      # Schritttyp
            "ls")        # Liniensuchschritte
    @printf("--------------------------------------------------------------------------------\n")
end

# Druckt die Daten einer einzelnen Iteration in formatierter Form
function print_iteration(k, f, inf_pr, inf_du, inf_comp, d_norm, alpha, step_type, ls)
    @printf("%-4d %-12.4e %-10.2e %-10.2e %-10.2e %-10.2e %-6.3f %-4s %2d\n",
        k,         # Iterationsnummer
        f,         # Zielfunktionswert
        inf_pr,    # Primale Unzulässigkeit (Nebenbedingungsverletzung)
        inf_du,    # Duale Unzulässigkeit (Gradient der Lagrange-Funktion)
        inf_comp,  # Komplementaritätsfehler
        d_norm,    # Norm des berechneten Suchschritts
        alpha,     # Gewählte Schrittweite aus Liniensuche
        step_type, # Art des Schritts (z.B. "full", "soc")
        ls)        # Anzahl benötigte Liniensuchschritte
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

# Berechnet einen Second-Order Correction (SOC) Schritt
# SOC wird verwendet wenn der normale SQP-Schritt eine zu kleine Schrittweite liefert
# Ziel: Bessere Approximation der Nebenbedingungen durch quadratische Korrektur
function compute_soc_step(nlp::AbstractNLPModel, x::Vector{Float64}, d::Vector{Float64},
                          y::Vector{Float64}, qp_solver::AbstractQPSolver, settings::Settings)
    n = nlp.meta.nvar  # Anzahl Variablen
    m = nlp.meta.ncon  # Anzahl Nebenbedingungen
    try
        # Berechne Testpunkt nach dem ursprünglichen SQP-Schritt
        x_trial = x + d
        
        # Evaluiere Nebenbedingungen am Testpunkt - diese sollten durch SOC korrigiert werden
        c_trial = cons(nlp, x_trial)
        
        # Jacobi-Matrix der Nebenbedingungen (lineare Approximation)
        A = jac(nlp, x)
        
        # SOC-Ziel: Nebenbedingungsverletzung korrigieren
        b_soc = -c_trial
        
        # Hessian der Lagrange-Funktion für quadratisches Modell
        H = hess(nlp, x, y)
        H_reg = convexify_hessian!(copy(H), settings)  # Positive Definitheit sicherstellen
        
        # Gradient der Zielfunktion
        g = grad(nlp, x)
        
        # Variablenschranken für SOC-Schritt (relativ zum Testpunkt)
        lvar_soc = nlp.meta.lvar - x_trial
        uvar_soc = nlp.meta.uvar - x_trial
        
        # Nebenbedingungsschranken für SOC (Gleichungen: c_trial + A*d_soc = 0)
        lcon_soc = m > 0 ? b_soc : Float64[]
        ucon_soc = m > 0 ? b_soc : Float64[]
        
        # Löse das SOC-QP-Teilproblem
        result = solve_qp(qp_solver, H_reg, g, A, lcon_soc, ucon_soc, lvar_soc, uvar_soc, settings)

        if result.success
            return SOCResult(result.x, true, result.status)
        end
    catch e
        @warn "SOC-Update fehlgeschlagen: $e"
    end
    return SOCResult(zeros(n), false, :Error)  # Fehlerfall
end

# Haupt-SQP-Methode - Löst nichtlineare Optimierungsprobleme mittels Sequential Quadratic Programming
function sqp_method(nlp::ADNLPModel, settings::Settings=Settings())
    # Problemdimensions extrahieren
    n = nlp.meta.nvar  # Anzahl Entscheidungsvariablen
    m = nlp.meta.ncon  # Anzahl Nebenbedingungen
    
    # Startwerte initialisieren
    x_k = copy(nlp.meta.x0)  # Aktuelle primale Variablen (Startwert)
    y_k = zeros(m)           # Lagrange-Multiplikatoren für Nebenbedingungen
    z_k = zeros(n)           # Lagrange-Multiplikatoren für Variablenschranken

    # QP-Sublöser für quadratische Teilprobleme erstellen
    qp_solver = create_qp_solver(n, m, settings)
    
    # Filter für globale Konvergenz initialisieren
    filter = Filter()
    
    # Statistik-Objekt zur Verfolgung des Algorithmus-Fortschritts
    stats = Stats(
        copy(x_k), copy(y_k), copy(z_k),  # Aktuelle Iterationswerte
        obj(nlp, x_k), 0.0, 0.0, 0.0,    # Zielfunktionswert und Residuen
        max_iter, max_iter, 0             # Status-Initialisierung
    )
    stats.obj_evals = 1                  # Bereits eine Zielfunktionsauswertung
    stats.cons_evals = m > 0 ? 1 : 0     # Nebenbedingungsauswertungen (falls vorhanden)
    start_time = time()                  # Zeitmessung starten

    # Algorithmus-Kontrollvariablen
    k = 0                    # Iterationszähler
    converged = false        # Konvergenz-Flag
    tol = settings.tol       # Konvergenz-Toleranz
    max_iterations = settings.max_iter  # Maximale Iterationszahl
    
    # SOC-Parameter: Mindestverbesserung um SOC-Schritt zu akzeptieren
    soc_improvement_threshold = getfield(settings, :soc_improvement_threshold)::Float64

    # Drucke Iterationsüberschrift wenn Verbose-Modus aktiviert
    settings.verbose && print_iteration_header()

    # Hauptschleife des SQP-Algorithmus
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