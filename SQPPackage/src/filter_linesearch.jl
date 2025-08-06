module FilterLineSearchSQP

using LinearAlgebra
using SparseArrays
using Printf
using ADNLPModels
using NLPModels

using ..SettingsModule
using ..StatsModule

# Export der wichtigsten Funktionen und Datentypen aus diesem Modul
export FilterPoint, Filter, constraint_violation, filter_line_search, 
       add_to_filter!, remove_dominated_points!, is_acceptable_to_filter

# --- Strukturen für den Filter ---

# FilterPoint speichert einen Punkt mit einem Wert der Constraint-Verletzung (h) und einem Funktionswert (f)
struct FilterPoint
    h::Float64  # Constraint-Verletzung
    f::Float64  # Funktionswert
end

# Filter speichert eine Liste von FilterPoints, um sie in der Filter-Line-Search zu verwenden
mutable struct Filter
    points::Vector{FilterPoint}  # Liste von Filterpunkten
    function Filter()
        new(FilterPoint[])  # Initialisierung der leeren Punkteliste
    end
end

# --- Hilfsfunktionen für Filter ---

# Füge einen neuen Punkt zum Filter hinzu und entferne dominierte Punkte
function add_to_filter!(filter::Filter, h::Float64, f::Float64)
    push!(filter.points, FilterPoint(h, f))  # Neuen Punkt hinzufügen
    remove_dominated_points!(filter)  # Entferne dominierte Punkte
end

# Entferne dominierte Punkte aus dem Filter
function remove_dominated_points!(filter::Filter)
    n = length(filter.points)  # Anzahl der Punkte im Filter
    if n <= 1  # Falls nur ein Punkt oder weniger vorhanden sind, nichts tun
        return
    end

    keep = trues(n)  # Array, das festlegt, welche Punkte behalten werden
    for i in 1:n
        for j in 1:n
            if i != j && keep[j]  # Vergleiche Punkt i mit Punkt j
                if filter.points[j].h <= filter.points[i].h &&  # Wenn Punkt j den Punkt i dominiert
                   filter.points[j].f <= filter.points[i].f
                    keep[i] = false  # Markiere Punkt i zum Entfernen
                    break
                end
            end
        end
    end

    # Filter aktualisieren, nur die behaltenen Punkte bleiben übrig
    filter.points = filter.points[keep]
end

# Überprüfe, ob ein neuer Punkt akzeptabel für den Filter ist
function is_acceptable_to_filter(filter::Filter, h::Float64, f::Float64, gh::Float64, gf::Float64)
    for point in filter.points  # Vergleiche mit allen Punkten im Filter
        if h >= point.h - gh && f >= point.f - gf
            return false  # Der Punkt ist nicht akzeptabel, da er dominiert wird
        end
    end
    return true  # Der Punkt ist akzeptabel
end

# --- Funktionen zur Problembewertung ---

# Berechne die Verletzung der Constraints für den Punkt x
function constraint_violation(nlp::ADNLPModel, x::Vector{Float64})
    if nlp.meta.ncon == 0  # Keine Constraints vorhanden
        return 0.0
    end

    c = cons(nlp, x)  # Hole die Constraints
    lcon = nlp.meta.lcon  # Untere Bounds der Constraints
    ucon = nlp.meta.ucon  # Obere Bounds der Constraints

    h = 0.0
    for i in 1:length(c)
        # Überprüfe und berechne die Verletzung der Constraints
        if isfinite(lcon[i]) && c[i] < lcon[i]
            h += (lcon[i] - c[i])^2  # Untere Grenze verletzt
        elseif isfinite(ucon[i]) && c[i] > ucon[i]
            h += (c[i] - ucon[i])^2  # Obere Grenze verletzt
        elseif isfinite(lcon[i]) && isfinite(ucon[i]) && lcon[i] == ucon[i]
            h += (c[i] - lcon[i])^2  # Gleichheits-Constraint verletzt
        end
    end

    return sqrt(h)  # Berechne die Gesamtheit der Verletzungen
end

# Berechne den KKT-Fehler (Karush-Kuhn-Tucker Fehler) für den aktuellen Punkt
function compute_kkt_error(nlp::ADNLPModel, x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64})
    g = grad(nlp, x)  # Berechne den Gradienten der Zielfunktion

    grad_lag = copy(g)  # Kopiere den Gradienten
    if nlp.meta.ncon > 0 && length(y) > 0  # Wenn Constraints vorhanden sind
        J = jac(nlp, x)  # Hole die Jacobian-Matrix der Constraints
        grad_lag += J' * y  # Addiere den Beitrag der Constraints
    end

    # Berechne die Primal- und Dualresiduen
    primal_res = constraint_violation(nlp, x)
    dual_res = norm(grad_lag)  # Dualresiduum (Gradient des Lagrangeansatzes)

    return max(primal_res, dual_res)  # Rückgabe des maximalen Fehlers
end

# --- Hilfsbedingungen für Line Search ---

# Überprüfe die Switching-Bedingung für den Filter
function check_switching_condition(grad_lag_norm::Float64, h::Float64, dlt::Float64, ga::Float64, a::Float64)
    threshold = ga * min(a, 1.0) * h^dlt  # Berechne den Schwellwert
    return grad_lag_norm <= threshold  # Überprüfe die Switching-Bedingung
end

# Überprüfe die Armijo-Bedingung für den Funktionswert
function check_armijo_condition(f_new::Float64, f_current::Float64, eta_f::Float64, a::Float64, grad_f_dot_d::Float64)
    return f_new <= f_current + eta_f * a * grad_f_dot_d  # Armijo-Bedingung
end

# Überprüfe, ob eine ausreichende Abnahme der Constraint-Verletzung vorliegt
function check_sufficient_decrease(h_new::Float64, h_current::Float64, gh::Float64)
    return h_new <= (1 - gh) * h_current  # Sufficient Decrease Bedingung
end

# Berechne den minimalen Alpha-Wert für die Line Search
function compute_alpha_min(h_current::Float64, gh::Float64, sh::Float64, f_current::Float64, gf::Float64, sf::Float64)
    if h_current > 0
        return gh / (sh * h_current)  # Wenn die aktuelle Verletzung der Constraints positiv ist
    else
        return gf / (sf * abs(f_current))  # Ansonsten berechne mit dem Funktionswert
    end
end

# --- Hauptfunktion: Filter Line Search ---

function filter_line_search(nlp::ADNLPModel, x::Vector{Float64}, d::Vector{Float64},
                            y::Vector{Float64}, f_current::Float64, h_current::Float64,
                            g_current::Vector{Float64}, filter::Filter, settings::Settings)
    # Initialisierung der Alpha-Werte und der maximalen Iterationen
    a = 1.0
    ls_iters = 0
    max_ls_iters = hasfield(typeof(settings), :max_ls_iters) ? settings.max_ls_iters : 20

    grad_f_dot_d = dot(g_current, d)  # Berechne das Skalarprodukt zwischen Gradienten und Richtungsvektor

    # Parameter aus Settings
    gh = hasfield(typeof(settings), :gh) ? settings.gh : 1e-5
    gf = hasfield(typeof(settings), :gf) ? settings.gf : 1e-5
    eta_f = hasfield(typeof(settings), :eta_f) ? settings.eta_f : 1e-4
    dlt = hasfield(typeof(settings), :dlt) ? settings.dlt : 1.0
    ga = hasfield(typeof(settings), :ga) ? settings.ga : 1e-5
    sh = hasfield(typeof(settings), :sh) ? settings.sh : 2.3
    sf = hasfield(typeof(settings), :sf) ? settings.sf : 2.3

    a_min = if hasfield(typeof(settings), :a_min_fixed) && settings.a_min_fixed !== nothing
        settings.a_min_fixed
    else
        compute_alpha_min(h_current, gh, sh, f_current, gf, sf)  # Berechne den minimalen Alpha-Wert
    end

    # Schleife für die Line Search
    while a >= a_min && ls_iters < max_ls_iters
        ls_iters += 1
        x_trial = x + a * d  # Berechne den neuen Trial-Punkt

        if any(.!isfinite.(x_trial))  # Überprüfe, ob der Punkt NaN oder Inf enthält
            settings.verbose && println("FEHLER: Numerische Instabilität in Liniensuche")
            return 0.0, "error", ls_iters  # Fehler: Unendliche Werte
        end

        try
            f_trial = obj(nlp, x_trial)  # Berechne den Funktionswert des Trial-Punkts
            h_trial = constraint_violation(nlp, x_trial)  # Berechne die Constraint-Verletzung des Trial-Punkts

            # Wenn der Trial-Punkt ungültig ist, reduziere Alpha und wiederhole
            if !isfinite(f_trial) || !isfinite(h_trial)
                settings.verbose && println("Trial-Punkt enthält NaN/Inf bei a = $a, reduziere Schrittweite")
                a *= 0.1
                continue
            end

            g_trial = grad(nlp, x_trial)  # Berechne den Gradienten des Trial-Punkts
            grad_lag_norm = norm(g_trial)  # Berechne den normierten Gradienten des Lagrangeansatzes

            is_f_type = check_switching_condition(grad_lag_norm, h_current, dlt, ga, a)  # Überprüfe den Typ des Punktes

            # Überprüfe die Armijo-Bedingung oder Suffiziente Abnahme der Constraint-Verletzung
            if is_f_type
                if check_armijo_condition(f_trial, f_current, eta_f, a, grad_f_dot_d)
                    return a, "f", ls_iters  # Wenn die Armijo-Bedingung erfüllt ist, zurückgeben
                end
            else
                if check_sufficient_decrease(h_trial, h_current, gh) ||
                   is_acceptable_to_filter(filter, h_trial, f_trial, gh, gf)
                    add_to_filter!(filter, h_current, f_current)  # Filter aktualisieren
                    return a, "h", ls_iters  # Wenn die Suffizienzbedingung erfüllt ist, zurückgeben
                end
            end

        catch e  # Fehlerbehandlung
            settings.verbose && println("Fehler bei a = ", a, ": ", e)
        end

        # Reduziere Alpha und wiederhole den Suchvorgang
        a *= 0.5
    end

    settings.verbose && println("WARNUNG: Keine akzeptable Schrittgröße gefunden")
    return a, "h", ls_iters  # Wenn keine Schrittgröße gefunden wurde
end

end # Ende des Moduls
