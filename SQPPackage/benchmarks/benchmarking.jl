
"""
Benchmarking - Performance-Analyse für Optimierungslöser

Dieses Modul implementiert Performance Profiles nach Dolan & Moré (2002)
zur objektiven Bewertung und zum Vergleich verschiedener Löser-Konfigurationen.

Hauptfunktionen:
- Performance Profile Erstellung
- Residuen-Berechnung für Zulässigkeit
- Statistische Auswertung von Löser-Performance
"""
module Benchmarking

using LinearAlgebra, Printf
using Plots
using ..SQPPackage: Stats, SQPStatus, kkt_point, constraint_violation

export performance_profile, get_primal_residual, get_dual_residual

# Berechnet das primale Residuum - Maß für Nebenbedingungsverletzung
# Gibt zurück: ||max(0, lcon - c(x)) + max(0, c(x) - ucon)||
function get_primal_residual(nlp, x)
    """
    Berechnet das primale Residuum ||c(x)|| basierend auf Nebenbedingungstyp
    """
    # Unrestringierte Probleme haben keine Nebenbedingungen
    if nlp.meta.ncon == 0
        return 0.0
    end
    
    # Evaluiere alle Nebenbedingungen am aktuellen Punkt
    c = cons(nlp, x)
    lcon, ucon = nlp.meta.lcon, nlp.meta.ucon  # Untere und obere Schranken
    
    residual = 0.0
    for i in 1:length(c)
        if lcon[i] == ucon[i]  # Gleichheitsnebenbedingung: c_i(x) = b_i
            residual += abs(c[i] - lcon[i])
        else  # Ungleichheitsnebenbedingung: lcon_i ≤ c_i(x) ≤ ucon_i
            if c[i] < lcon[i]      # Verletzung der unteren Schranke
                residual += abs(c[i] - lcon[i])
            elseif c[i] > ucon[i]  # Verletzung der oberen Schranke
                residual += abs(c[i] - ucon[i])
            end
            # Falls lcon_i ≤ c_i(x) ≤ ucon_i: Keine Verletzung, residual += 0
        end
    end
    
    return residual
end

# Berechnet das duale Residuum - Maß für Optimalitätsbedingungen
# Basiert auf Gradient der Lagrange-Funktion: ∇L(x,y,z) = ∇f(x) - ∇c(x)^T y - z
function get_dual_residual(nlp, x, y, zL, zU, g)
    """
    Berechnet das duale Residuum ||∇L(x,y,z)|| für KKT-Bedingungen
    """
    # Gradient der Lagrange-Funktion: ∇L = ∇f - ∇c^T * y - z
    if nlp.meta.ncon > 0
        J = jac(nlp, x)              # Jacobi-Matrix der Nebenbedingungen
        dual_residual = g - J' * y   # ∇f(x) - ∇c(x)^T * y
    else
        dual_residual = copy(g)      # Unrestringierte Probleme: nur ∇f(x)
    end
    
    # Berücksichtige Lagrange-Multiplikatoren für Variablenschranken
    if !isempty(zL)
        dual_residual -= zL          # Subtrahiere Multiplikatoren für untere Schranken
    end
    if !isempty(zU)
        dual_residual += zU          # Addiere Multiplikatoren für obere Schranken
    end
    
    # Infinity-Norm als Maß für größte Komponente des Gradienten
    return norm(dual_residual, Inf)
end

function performance_profile(times, names; title="Performance Profile", logscale=false)
    # Logarithmisch verteilte τ-Werte
    τ = exp.(range(log(1), log(100), length=100))

    # Erstelle Plot mit besserem Styling
    plt = Plots.plot(
        xlabel="Performance Ratio τ", 
        ylabel="Fraction of Problems Solved ρ(τ)", 
        title=title, 
        legend=:bottomright,
        grid=true,
        size=(800, 600),
        dpi=300
    )

    # Farben für verschiedene Solver
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray]

    for (i, solver) in enumerate(names)
        # Identifiziere erfolgreiche Läufe
        finite_mask = isfinite.(times[:, i])

        if !any(finite_mask)
            # Keine erfolgreichen Läufe
            Plots.plot!(plt, τ, zeros(length(τ)), 
                       label="$solver (keine Lösungen)", 
                       lw=3, 
                       linestyle=:dash,
                       color=colors[mod(i-1, length(colors)) + 1])
            continue
        end

        # Berechne Performance-Ratios
        r = fill(Inf, size(times, 1))
        for j in 1:size(times, 1)
            if any(isfinite.(times[j, :]))  # Mindestens ein Solver hat das Problem gelöst
                min_time = minimum(times[j, isfinite.(times[j, :])])
                if finite_mask[j]
                    r[j] = times[j, i] / min_time
                end
            end
        end

        # Berechne ρ(τ)
        ρ = [count(r .<= t) / size(times, 1) for t in τ]

        Plots.plot!(plt, τ, ρ, 
                   label=solver, 
                   lw=3,
                   color=colors[mod(i-1, length(colors)) + 1])
    end

    # Achsenkonfiguration
    Plots.ylims!(plt, 0, 1)
    if logscale
        Plots.plot!(plt, xscale=:log10)
        Plots.xlims!(plt, 1, 100)
    else
        Plots.xlims!(plt, 1, 20)
    end

    # Referenzlinien
    Plots.hline!(plt, [0.5, 0.8], color=:gray, linestyle=:dot, alpha=0.5, label="")

    return plt
end

end
