#!/usr/bin/env julia

"""
Großer Benchmark: SQP vs. etablierte Solver (Ipopt, MadNLP)
Verwendet optimale Konfiguration aus vorherigen Benchmarks
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using SQPPackage
include("run_benchmark_template.jl")

# Versuche konkurrierende Solver zu laden
try
    using NLPModelsIpopt
    global IPOPT_AVAILABLE = true
catch
    global IPOPT_AVAILABLE = false
    println("Warning: Ipopt nicht verfügbar")
end

try
    using MadNLP
    global MADNLP_AVAILABLE = true
catch
    global MADNLP_AVAILABLE = false
    println("Warning: MadNLP nicht verfügbar")
end

println("="^60)
println("Großer Benchmark: SQP vs. etablierte Solver")
println("="^60)

# Solver-Konfigurationen mit optimalen Einstellungen
solver_configs = Dict{String, Function}()

# Unsere beste SQP-Konfiguration (basierend auf vorherigen Benchmarks)
solver_configs["SQP_Optimal"] = nlp -> sqp_method(nlp, Settings(
    qp_solver=:osqp,
    hessian_convexification=:lm,
    use_globalization=true,
    use_soc=true,
    tol=1e-6,
    max_iter=1000,
    verbose=false,
    soc_improvement_threshold=0.5,
    γh=1e-5,
    γf=1e-5,
    γα=0.05,
    sh=1.1,
    sf=2.3,
    ηf=1e-4,
    osqp_settings=Dict(
        "verbose" => false,
        "max_iter" => 1000,
        "eps_abs" => 1e-6,
        "eps_rel" => 1e-6
    )
))

# SQP ohne SOC zum Vergleich
solver_configs["SQP_without_SOC"] = nlp -> sqp_method(nlp, Settings(
    qp_solver=:osqp,
    hessian_convexification=:lm,
    use_globalization=true,
    use_soc=false,
    tol=1e-6,
    max_iter=1000,
    verbose=false
))

# Ipopt (falls verfügbar)
if IPOPT_AVAILABLE
    solver_configs["Ipopt"] = nlp -> ipopt(nlp, 
        print_level=0,
        tol=1e-6,
        max_iter=1000,
        max_cpu_time=300.0
    )
end

# MadNLP (falls verfügbar)
if MADNLP_AVAILABLE
    solver_configs["MadNLP"] = nlp -> madnlp(nlp,
        print_level=MadNLP.SILENT,
        tol=1e-6,
        max_iter=1000,
        max_cpu_time=300.0
    )
end

# Problem-Filter für große Probleme (250+ Variablen)
function large_problem_filter()
    if OPTIMIZATION_PROBLEMS_AVAILABLE
        return filter_problems_by_size(250, 1000, 0)  # 250-1000 Variablen
    else
        return Symbol[]  # Fallback auf Testprobleme
    end
end

# Problem-Filter für mittlere Probleme (als Fallback)
function medium_problem_filter()
    return filter_problems_by_size(50, 250, 0)  # 50-250 Variablen
end

println("Großer Benchmark wird gestartet...")

# Teste verschiedene Problemgrößen
problem_filters = [
    ("Large_Problems", large_problem_filter, false),
    ("Medium_Problems", medium_problem_filter, false),
    ("Test_Problems", () -> [:P1, :P2, :P3, :P4, :P5, :P6, :P7, :P8, :P9, :P10, :P11, :P12, :P13, :P14, :P15, :P16], true)
]

for (problem_set_name, filter_func, use_test_probs) in problem_filters
    println("\n" * "="^50)
    println("Teste auf: $problem_set_name")
    println("="^50)

    # Überspringe OptimizationProblems wenn nicht verfügbar
    if !use_test_probs && !OPTIMIZATION_PROBLEMS_AVAILABLE
        println("OptimizationProblems.jl nicht verfügbar, überspringe $problem_set_name...")
        continue
    end

    # Hole Probleme
    if use_test_probs
        test_problems = filter_func()
    else
        test_problems = filter_func()
        if isempty(test_problems)
            println("Keine Probleme in $problem_set_name gefunden, überspringe...")
            continue
        end
    end

    # Passe maximale Problemanzahl an
    max_probs = use_test_probs ? length(test_problems) : min(100, length(test_problems))

    for metric in [:total_time, :function_evals]
        println("\nMetrik: $metric")

        results, τ_values, profiles = run_benchmark_template(
            solver_configs, 
            filter_func,
            metric=metric,
            save_prefix="large_benchmark_$problem_set_name",
            results_dir=joinpath(@__DIR__, "..", "results"),
            use_test_problems=use_test_probs,
            max_problems=max_probs,
            timeout=600.0  # 10 Minuten Timeout für große Probleme
        )

        println("✓ Großer Benchmark ($problem_set_name, $metric) abgeschlossen")
    end
end

println("\n" * "="^60)
println("✓ Großer Benchmark abgeschlossen!")
println("Results gespeichert in: results/")
println("")
println("Benchmark-Übersicht:")
println("  ✓ QP-Solver Vergleich")
println("  ✓ Hessian-Konvexifizierung Vergleich") 
println("  ✓ Globalisierungs-Strategien Vergleich")
println("  ✓ Toleranz-Einstellungen Vergleich")
println("  ✓ Großer Benchmark vs. etablierte Solver")
println("="^60)