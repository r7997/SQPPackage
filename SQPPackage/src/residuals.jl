module ResidualsModule

using ADNLPModels
using NLPModels
using LinearAlgebra
using NLPModels

# Exportiert die Hauptfunktionen dieses Moduls
export get_primal_residual, get_dual_residual, get_total_complementarity_residual, 
       compute_kkt_residuals, constraint_violation

# Funktion zur Berechnung der Verletzung der Constraints (c) eines Modells
function constraint_violation(nlp::ADNLPModel, x::Vector{Float64})
    if nlp.meta.ncon == 0  # Falls keine Constraints vorhanden sind
        return 0.0
    end

    c = cons(nlp, x)  # Berechne die Constraints
    lcon = nlp.meta.lcon  # Untere Grenzen der Constraints
    ucon = nlp.meta.ucon  # Obere Grenzen der Constraints

    h = 0.0
    # Durchlaufe alle Constraints und berechne die Verstöße
    for i in 1:length(c)
        if isfinite(lcon[i]) && c[i] < lcon[i]
            h += (lcon[i] - c[i])^2  # Violation der unteren Grenze
        elseif isfinite(ucon[i]) && c[i] > ucon[i]
            h += (c[i] - ucon[i])^2  # Violation der oberen Grenze
        elseif isfinite(lcon[i]) && isfinite(ucon[i]) && lcon[i] == ucon[i]
            h += (c[i] - lcon[i])^2  # Violation eines festen Wertes
        end
    end
    return sqrt(h)  # Rückgabe der gesamten Verletzung
end

# Funktion zur Berechnung des primalen Residuums
function get_primal_residual(x, c, cL, cU, xL, xU)
    h_eq = 0.0
    # Berechne die Verletzung der Gleichheits-Constraints
    for i = 1:length(c)
        if abs(cL[i] - cU[i]) < 1e-12  # Für Gleichheits-Constraints
            h_eq = max(h_eq, abs(c[i] - cU[i]))
        end
    end

    h_ineq = 0.0
    # Berechne die Verletzung der Ungleichheits-Constraints
    for i = 1:length(c)
        if abs(cL[i] - cU[i]) >= 1e-12  # Für Ungleichheits-Constraints
            violation = max(0.0, c[i] - cU[i], cL[i] - c[i])
            h_ineq = max(h_ineq, violation)
        end
    end

    h_bounds = 0.0
    # Berechne die Verletzung der Variablen-Bounds
    for i = 1:length(x)
        if !isinf(xU[i])
            h_bounds = max(h_bounds, max(0.0, x[i] - xU[i]))
        end
        if !isinf(xL[i])
            h_bounds = max(h_bounds, max(0.0, xL[i] - x[i]))
        end
    end
    return max(h_eq, h_ineq, h_bounds)  # Rückgabe des maximalen Residuums
end

# Funktion zur Berechnung des Gesamtkomplementaritätsresiduums
function get_total_complementarity_residual(x, c, xL, xU, cL, cU, zL, zU, yL, yU)
    comp_residual = 0.0
    # Berechne das Komplementaritätsresiduum für die unteren Bounds
    for i = 1:length(x)
        if !isinf(xL[i]) && length(zL) >= i
            comp_residual = max(comp_residual, abs(zL[i] * (x[i] - xL[i])))
        end
    end
    # Berechne das Komplementaritätsresiduum für die oberen Bounds
    for i = 1:length(x)
        if !isinf(xU[i]) && length(zU) >= i
            comp_residual = max(comp_residual, abs(zU[i] * (xU[i] - x[i])))
        end
    end
    # Berechne das Komplementaritätsresiduum für die Constraints
    for i = 1:length(c)
        if !isinf(cL[i]) && length(yL) >= i
            comp_residual = max(comp_residual, abs(yL[i] * (c[i] - cL[i])))
        end
    end
    for i = 1:length(c)
        if !isinf(cU[i]) && length(yU) >= i
            comp_residual = max(comp_residual, abs(yU[i] * (cU[i] - c[i])))
        end
    end
    return comp_residual  # Rückgabe des gesamten Komplementaritätsresiduums
end

# Funktion zur Berechnung des dualen Residuums
function get_dual_residual(g, A, y, z)
    grad_lag = g + A' * y + z  # Berechne den Gradienten des Lagrange-Funktionals
    return norm(grad_lag, Inf)  # Rückgabe des maximalen dualen Residuums
end

# Funktion zur Berechnung der KKT-Restfehler
function compute_kkt_residuals(nlp::ADNLPModel, x, y, z)
    g = grad(nlp, x)  # Berechne den Gradienten der Zielfunktion
    c = nlp.meta.ncon > 0 ? cons(nlp, x) : Float64[]  # Berechne die Constraints
    A = nlp.meta.ncon > 0 ? jac(nlp, x) : zeros(0, length(x))  # Jacobimatrix der Constraints
    
    # Berechne und gebe die KKT-Restfehler zurück
    return (
        get_primal_residual(x, c, nlp.meta.lcon, nlp.meta.ucon, nlp.meta.lvar, nlp.meta.uvar),
        get_dual_residual(g, A, y, z),
        get_total_complementarity_residual(
            x, c, nlp.meta.lvar, nlp.meta.uvar, 
            nlp.meta.lcon, nlp.meta.ucon,
            z, z, y, y
        )
    )
end

end
