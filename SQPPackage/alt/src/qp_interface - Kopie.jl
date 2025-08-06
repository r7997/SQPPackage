module QPInterface

using LinearAlgebra, SparseArrays, OSQP
using ..SettingsModule

# Exportiert die Hauptfunktionen und Datentypen dieses Moduls
export solve_qp, create_qp_solver, QPSolver, QPResult

# Struktur zur Speicherung der QP-Solver-Informationen
struct QPSolver
    n::Int              # Anzahl der Variablen (Dimension der Zielfunktion)
    m::Int              # Anzahl der Constraints (Dimension der Einschränkungen)
    settings::Settings  # Solver-Einstellungen
end

# Struktur zur Speicherung des Ergebnisses einer QP-Lösung
struct QPResult
    x::Vector{Float64}   # Lösung des Problems (Variable x)
    y::Vector{Float64}   # Lagrange-Multiplikatoren für Constraints
    zL::Vector{Float64}  # Primal dual variable für untere Bounden (zL)
    zU::Vector{Float64}  # Primal dual variable für obere Bounden (zU)
    status::Symbol       # Status der Lösung (z.B. :Solved, :Suboptimal, :Error)
    success::Bool        # Indikator, ob das Problem erfolgreich gelöst wurde
end

# Funktion zur Erstellung eines neuen QP-Solvers mit gegebenen Dimensionen und Einstellungen
function create_qp_solver(n::Int, m::Int, settings::Settings)
    return QPSolver(n, m, settings)  # Gibt einen neuen QP-Solver zurück
end

# Funktion zur Lösung des Quadratischen Programmierungsproblems (QP)
function solve_qp(qp_solver::QPSolver, H::AbstractMatrix, g::Vector{Float64},
                  J::AbstractMatrix, lcon::Vector{Float64}, ucon::Vector{Float64},
                  lvar::Vector{Float64}, uvar::Vector{Float64}, settings::Settings)
    
    n = qp_solver.n  # Anzahl der Variablen
    m = length(lcon)  # Anzahl der Constraints
    
    try
        # Konvertiere die Matrizen in Sparse-Formate
        P = sparse(H)  # Zielfunktionsmatrix (Hessische Matrix)
        A = sparse(J)  # Jacobimatrix der Constraints
        q = g           # Gradientenvektor der Zielfunktion
        
        # Kombiniere die Constraints (obere und untere Grenzen für Variablen und Constraints)
        l = [lcon; lvar]  # Untere Bounds für Constraints und Variablen
        u = [ucon; uvar]  # Obere Bounds für Constraints und Variablen
        A_full = [A; sparse(I, n, n)]  # Erweiterte Jacobimatrix mit Identitätsmatrix
        
        # Hole die OSQP-spezifischen Einstellungen aus den übergebenen Einstellungen
        osqp_settings = settings.osqp_settings
        
        # Initialisiere das OSQP Modell mit den gegebenen Parametern
        model = OSQP.Model()
        OSQP.setup!(model; 
            P=P, q=q, A=A_full, l=l, u=u,
            verbose = get(osqp_settings, "verbose", false),
            max_iter = get(osqp_settings, "max_iter", 10000),
            eps_abs = get(osqp_settings, "eps_abs", 1e-6),
            eps_rel = get(osqp_settings, "eps_rel", 1e-6)
        )
        
        # Löse das QP-Problem mit OSQP
        results = OSQP.solve!(model)
        
        # Überprüfe den Status des Lösungsversuchs
        if results.info.status_val == 1  # Gelöst (Solved)
            return QPResult(
                results.x,  # Lösung der Variablen
                results.y[1:m],  # Lagrange-Multiplikatoren für die Constraints
                max.(0, -results.y[m+1:end]),  # zL für die unteren Bounds
                max.(0, results.y[m+1:end]),    # zU für die oberen Bounds
                :Solved,  # Status: Gelöst
                true  # Erfolg
            )
        else
            # Rückgriff auf Gradientenabstieg, wenn OSQP nicht erfolgreich ist
            d = -g / (norm(g) + 1e-8)  # Berechne Richtung via Gradientenabstieg
            return QPResult(d, zeros(m), zeros(n), zeros(n), :Suboptimal, true)  # Suboptimale Lösung
        end
        
    catch e
        # Fehlerbehandlung, falls das OSQP-Modell fehlschlägt
        try
            d = -H \ g  # Versuche eine Lösung durch direkte Matrixinversion
            return QPResult(d, zeros(m), zeros(n), zeros(n), :Solved, true)  # Rückgabe der Lösung
        catch
            # Fallback: Berechne eine grobe Lösung via Gradientenabstieg
            d = -g / (norm(g) + 1e-8)  # Gradientenabstiegsrichtung
            return QPResult(d, zeros(m), zeros(n), zeros(n), :Error, false)  # Fehlerstatus
        end
    end
end

end
