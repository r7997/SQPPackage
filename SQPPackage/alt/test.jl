#!/usr/bin/env julia
"""
Minimal Example for SQPPackage.jl

Demonstriert die grundlegende Verwendung des SQP-Lösers mit Testproblemen aus src/test_problems.jl
"""

# Pfad zum src-Verzeichnis hinzufügen
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

include(joinpath(@__DIR__, "testproblems", "test.jl"))
using .SQPTestProblems  # ← beachte den Punkt für lokales Modul
using ..testproblems  # Verwende das Testprobleme-Modul
using SQPPackage
using Plots, LinearAlgebra, Printf

println("="^60)
println("SQPPackage.jl - Konvergenzverläufe für Testprobleme")
println("="^60)

# Liste aller Testproblem-Funktionen
PROBLEM_FUNCTIONS = [
    create_P1, create_P2, create_P3, create_P4, create_P5, create_P6,
    create_P7, create_P8, create_P9, create_P10, create_P11, create_P12,
    create_P13, create_P14, create_P15, create_P16
]

# Namen für die Probleme
PROBLEM_NAMES = [
    "P1: Einfaches quadratisches Problem",
    "P2: Quadratisches Problem mit Gleichheitsnebenbedingungen",
    "P3: Nichtlineares Problem mit Gleichheitsnebenbedingung",
    "P4: Komplexeres nichtlineares Problem",
    "P5: Exponentielles Problem mit Ungleichungsnebenbedingungen",
    "P6: Nichtlineares Problem mit Kreisnebenbedingung",
    "P7: Einfaches nichtlineares Problem mit Ungleichungsnebenbedingung",
    "P8: Komplexes Problem mit mehreren Nebenbedingungen",
    "P9: Nichtlineares Problem in einer Variablen",
    "P10: Quadratisches Problem mit Kreisnebenbedingung",
    "P11: Nichtlineares Problem mit zwei Gleichungsnebenbedingungen",
    "P12: Quadratisches Problem mit Ungleichungsnebenbedingungen",
    "P13: Rosenbrock-Funktion mit Nebenbedingungen",
    "P14: Lineares Problem mit mehreren Variablen und Nebenbedingungen",
    "P15: Einfaches quadratisches Problem mit Ungleichungsnebenbedingung",
    "P16: Quartic-Funktion mit Box-Constraints"
]

# Allgemeine Einstellungen für alle Probleme
BASE_SETTINGS = Settings(
    verbose = false,
    max_iter = 50,
    tol = 1e-6,
    qp_solver = :osqp,  # Nur OSQP verwenden
    use_globalization = true,
    hessian_convexification = :lm
)

function solve_and_plot(prob_name::String, prob_func::Function, settings::Settings)
    println("\n" * "="^60)
    println(prob_name)
    println("="^60)
    
    # Problem erstellen
    nlp = prob_func()
    
    # Problem lösen und Statistiken sammeln
    stats = sqp_method(nlp, settings)
    
    # Ergebnisse anzeigen
    println("\nLösung:")
    println("x = ", round.(stats.x, digits=6))
    @printf("f(x) = %.6f\n", stats.obj_val)
    println("Status: ", stats.sqp_status)
    println("Iterationen: ", length(stats.iteration_data))
    @printf("Gesamtzeit: %.4f s\n", stats.total_time)
    
    # Konvergenzverlauf plotten (falls Daten vorhanden)
    if !isempty(stats.iteration_data)
        plt = plot(layout=(3,1), size=(800,900), title=prob_name)
        
        # 1. Zielfunktion
        iters = [data.iter for data in stats.iteration_data]
        objectives = [data.objective for data in stats.iteration_data]
        plot!(plt[1], iters, objectives, 
              xlabel="Iteration", ylabel="f(x)", 
              label="Zielfunktion", lw=2, marker=:circle)
        
        # 2. Residuen
        primal_res = [data.inf_pr for data in stats.iteration_data]
        dual_res = [data.inf_du for data in stats.iteration_data]
        plot!(plt[2], iters, primal_res, 
              xlabel="Iteration", ylabel="Residuum (log)",
              label="Primales Residuum", 
              yscale=:log10, lw=2, color=:blue)
        plot!(plt[2], iters, dual_res, 
              label="Duales Residuum", 
              yscale=:log10, lw=2, color=:red, linestyle=:dash)
        
        # 3. Schrittweite und Alpha
        step_norms = [data.step_norm for data in stats.iteration_data]
        alphas = [data.alpha for data in stats.iteration_data]
        plot!(plt[3], iters, step_norms, 
              xlabel="Iteration", ylabel="Schrittweite",
              label="||d||", lw=2, color=:green)
        plot!(twinx(plt[3]), iters, alphas,
              ylabel="Alpha", label="Schrittlänge α",
              lw=2, color=:purple, linestyle=:dashdot)
        
        # Dateiname erstellen (ohne Sonderzeichen)
        safe_name = replace(prob_name, r"[:\s]" => "_")
        filename = "convergence_$safe_name.pdf"
        savefig(plt, filename)
        println("\nKonvergenzplot gespeichert als '$filename'")
    else
        println("\nKeine Iterationsdaten für Plot verfügbar")
    end
end

# Hauptfunktion
function main()
    # Erstelle results-Verzeichnis falls nicht vorhanden
    isdir("results") || mkdir("results")
    
    # Löse jedes Problem und erstelle Plot
    for (i, prob_func) in enumerate(PROBLEM_FUNCTIONS)
        name = PROBLEM_NAMES[i]
        try
            println("\n\n" * "="^60)
            @printf("Verarbeite Problem %d/%d: %s\n", i, length(PROBLEM_FUNCTIONS), name)
            solve_and_plot(name, prob_func, BASE_SETTINGS)
        catch e
            println("\nFehler bei Problem $name:")
            showerror(stdout, e)
            println("\nStacktrace:")
            stacktrace(catch_backtrace())
            println("\n" * "-"^60)
        end
    end
    
    println("\n" * "="^60)
    @printf("Alle %d Probleme verarbeitet!\n", length(PROBLEM_FUNCTIONS))
    println("="^60)
end

# Skript ausführen
main()