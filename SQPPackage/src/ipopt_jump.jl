module IpoptADNLP

# Importiere benötigte Module
using ADNLPModels   # Für die Definition von Nichtlinearer Programmierung (NLP) Problemen
using NLPModelsIpopt # Ermöglicht die direkte Verwendung von Ipopt mit NLPModels
using JuMP          # Modellierungssprache für Optimierungsprobleme in Julia
using Ipopt         # Optimierungssolver für nichtlineare Probleme
using Printf        # Für formatierte Ausgabe
using LinearAlgebra # Für lineare Algebra Operationen (z.B. Norm)

# Importiere spezifische Funktionen aus ADNLPModels (Objektfunktion und Nebenbedingungen)
import ADNLPModels: obj, cons

# Exportiere die Hauptfunktionen dieses Moduls
export solve_with_ipopt_adnlp, solve_with_ipopt_jump
function solve_with_ipopt_adnlp(nlp::ADNLPModel)
    # Konfiguriere Ipopt: Unterdrücke alle Ausgaben (Banner, Print-Level etc.)
    output = ipopt(nlp,
                   print_level=0,
                   sb="yes", # "skip_display_banner"
                   output_file="",
                   file_print_level=0,
                   print_options_documentation="no",
                   print_user_options="no")

    # Extrahiere Lösungskomponenten
    x_sol = output.solution # Optimale Variablenwerte
    status = output.status # Terminierungsstatus des Solvers

    # Extrahiere Lagrange-Multiplikatoren sicher (falls vorhanden)
    y_ipopt = get(() -> Float64[], output, :multipliers)   # Für Nebenbedingungen
    zL_ipopt = get(() -> Float64[], output, :multipliers_L) # Für untere Variablen-Grenzen
    zU_ipopt = get(() -> Float64[], output, :multipliers_U) # Für obere Variablen-Grenzen

    return x_sol, y_ipopt, zL_ipopt, zU_ipopt, status
end

function solve_with_ipopt_jump(nlp::ADNLPModel)
    model = Model(Ipopt.Optimizer) # Erstelle ein JuMP-Modell mit Ipopt als Solver

    # Unterdrücke alle Ipopt-Ausgaben über JuMP-Attribute
    set_silent(model)
    set_attribute(model, "print_level", 0)
    set_attribute(model, "sb", "yes")
    set_attribute(model, "print_options_documentation", "no")
    set_attribute(model, "print_user_options", "no")

    n = nlp.meta.nvar # Anzahl der Variablen
    m = nlp.meta.ncon # Anzahl der Nebenbedingungen

    @variable(model, x[i=1:n]) # Füge Variablen zum JuMP-Modell hinzu

    # Setze Variablen-Schranken
    for i in 1:n
        if isfinite(nlp.meta.lvar[i])
            set_lower_bound(x[i], nlp.meta.lvar[i])
        end
        if isfinite(nlp.meta.uvar[i])
            set_upper_bound(x[i], nlp.meta.uvar[i])
        end
    end

    # Setze Startwerte für die Variablen
    for i in 1:n
        if isfinite(nlp.meta.x0[i])
            set_start_value(x[i], nlp.meta.x0[i])
        end
    end

    # Füge die Zielfunktion hinzu (Minimierung oder Maximierung)
    if nlp.meta.minimize
        @objective(model, Min, obj(nlp, x))
    else
        @objective(model, Max, obj(nlp, x))
    end

    # Füge Nebenbedingungen hinzu, falls vorhanden
    if m > 0
        cons_expr = cons(nlp, x) # Holen der Nebenbedingungsausdrücke
        for i in 1:m
            lcon = nlp.meta.lcon[i]
            ucon = nlp.meta.ucon[i]

            if isfinite(lcon) && isfinite(ucon)
                if lcon == ucon # Gleichheitsnebenbedingung
                    @constraint(model, cons_expr[i] == lcon)
                else # Bereichsnebenbedingung
                    @constraint(model, lcon <= cons_expr[i] <= ucon)
                end
            elseif isfinite(lcon) # Nur untere Schranke
                @constraint(model, cons_expr[i] >= lcon)
            elseif isfinite(ucon) # Nur obere Schranke
                @constraint(model, cons_expr[i] <= ucon)
            end
        end
    end

    optimize!(model) # Löse das Optimierungsproblem

    # Extrahiere die Lösung und den Status
    status = termination_status(model)
    x_sol = value.(x)

    # Extrahiere Lagrange-Multiplikatoren für Nebenbedingungen
    y_ipopt = Float64[]
    if m > 0
        try
            # Versuche, Dualwerte von benannten Constraints zu bekommen
            y_ipopt = [dual(constraint_by_name(model, "c$i")) for i in 1:m]
        catch
            y_ipopt = zeros(m) # Falls nicht verfügbar, Nullen verwenden
        end
    end

    # Extrahiere Lagrange-Multiplikatoren für Variablen-Schranken
    zL_ipopt = zeros(n)
    zU_ipopt = zeros(n)
    try
        for i in 1:n
            if has_lower_bound(x[i])
                zL_ipopt[i] = dual(LowerBoundRef(x[i])) # Dual für untere Schranke
            end
            if has_upper_bound(x[i])
                zU_ipopt[i] = dual(UpperBoundRef(x[i])) # Dual für obere Schranke
            end
        end
    catch
        # Fehler beim Extrahieren der Multiplikatoren, Nullen verwenden
    end

    return x_sol, y_ipopt, zL_ipopt, zU_ipopt, status
end

# Hilfsfunktion zum sicheren Abrufen von Feldern, die möglicherweise nicht existieren
get(default::Function, obj, field::Symbol) = hasfield(typeof(obj), field) ? getfield(obj, field) : default()

function compare_solutions(name, x_sqp, x_ipopt,
                           y_sqp=nothing, y_ipopt=nothing,
                           z_sqp=nothing, z_ipopt=nothing)
    println("\n=== Vergleich für Problem $name ===")

    # Vergleich der Lösungvektoren (Variablen)
    println("Lösungsvektoren:")
    println("SQP:   ", round.(x_sqp, digits=6))
    println("Ipopt: ", round.(x_ipopt, digits=6))
    @printf("Differenz (L2 Norm): %.6e\n", norm(x_sqp - x_ipopt))

    # Vergleich der Nebenbedingungs-Multiplikatoren
    if !isnothing(y_sqp) && !isnothing(y_ipopt) && !isempty(y_sqp) && !isempty(y_ipopt)
        println("\nNebenbedingungen-Multiplikatoren:")
        println("SQP:   ", round.(y_sqp, digits=6))
        println("Ipopt: ", round.(y_ipopt, digits=6))
        @printf("Differenz (L2 Norm): %.6e\n", norm(y_sqp - y_ipopt))
    end

    # Vergleich der Schranken-Multiplikatoren
    if !isnothing(z_sqp) && !isnothing(z_ipopt) && !isempty(z_sqp) && !isempty(z_ipopt)
        println("\nSchranken-Multiplikatoren:")
        println("SQP:   ", round.(z_sqp, digits=6))
        println("Ipopt: ", round.(z_ipopt, digits=6))
        @printf("Differenz (L2 Norm): %.6e\n", norm(z_sqp - z_ipopt))
    end
end

end # Ende des Moduls