module StatsModule

# Import
using Base.Enums

# Exports für den Modulkonsum
export SQPStatus, Stats, kkt_point, max_iter, qp_failed, infeasible, restoration_phase, unbounded

# Enum für den SQP-Status. Jeder Status steht für einen bestimmten Zustand während der Optimierung.
@enum SQPStatus begin
    kkt_point        # Wenn der KKT-Punkt erreicht wurde (Optimalitätsbedingungen)
    max_iter         # Wenn die maximale Anzahl an Iterationen erreicht wurde
    qp_failed        # Wenn das Quadratic Programming (QP) Problem fehlschlägt
    unbounded        # Wenn das Optimierungsproblem unbeschränkt ist
    infeasible       # Wenn das Optimierungsproblem unlösbar ist
    restoration_phase  # Wenn der Algorithmus in die Wiederherstellungsphase geht (nach einem Fehler)
end

# Struktur, die alle notwendigen Statistiken für die SQP-Methode speichert
mutable struct Stats
    x::Vector{Float64}              # Lösung des Optimierungsproblems (z. B. Entscheidungsvariablen)
    y::Vector{Float64}              # Dualvariablen (z. B. Lagrange-Multiplikatoren)
    z::Vector{Float64}              # Slack-Variablen (z. B. Violation of Constraints)
    obj_val::Float64                # Der Wert der Zielfunktion bei der Lösung
    inf_pr::Float64                 # Primal-Infeasibility (Verletzung der Primal Constraints)
    inf_du::Float64                 # Dual-Infeasibility (Verletzung der Dual Constraints)
    inf_comp::Float64               # Kompensation der Infeasibility, auch genannt "Feasibility Gap"
    qp_status::SQPStatus            # Status des QP-Problems (z. B. erfolgreich, fehlgeschlagen)
    sqp_status::SQPStatus           # Status der gesamten SQP-Methode (z. B. KKT-Punkt, max Iterationen)
    iter::Int                       # Die aktuelle Iterationsnummer
    obj_evals::Int                  # Anzahl der Auswertungen der Zielfunktion
    grad_evals::Int                 # Anzahl der Berechnungen des Gradienten
    cons_evals::Int                 # Anzahl der Berechnungen der Constraints
    jac_evals::Int                  # Anzahl der Berechnungen der Jacobi-Matrix
    hess_evals::Int                 # Anzahl der Berechnungen der Hessian-Matrix
    qp_solve_status::Symbol        # Der Status des QP-Lösers (z. B. :solved, :failed, etc.)
    total_time::Float64            # Gesamtzeit des Algorithmus
    x_prev::Vector{Float64}        # Die Werte von x in der vorherigen Iteration (für Convergence-Checks)
    iteration_data::Vector{Any}    # Weitere Daten für jede Iteration (z. B. Fortschritt, Zeit etc.)

    # Konstruktor, der es ermöglicht, eine Stats-Struktur mit den notwendigen Werten zu erstellen
    function Stats(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}, 
                   obj_val::Float64, inf_pr::Float64, inf_du::Float64, inf_comp::Float64,
                   qp_status::SQPStatus, sqp_status::SQPStatus, iter::Int)
        new(x, y, z, obj_val, inf_pr, inf_du, inf_comp, qp_status, sqp_status, iter,
            0, 0, 0, 0, 0, :unknown, 0.0,
            similar(x),         # Initialisierung von x_prev mit einer Kopie von x
            [])                 # Initialisierung von iteration_data als leeres Array
    end
end

end
