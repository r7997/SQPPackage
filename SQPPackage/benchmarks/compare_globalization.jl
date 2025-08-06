#!/usr/bin/env julia

"""
Globalisierungs-Strategien Vergleichsbenchmark
Vergleicht verschiedene Kombinationen von Liniensuche und SOC
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using SQPPackage
include("run_benchmark_template.jl")

println("="^60)
println("Globalisierungs-Strategien Vergleichsbenchmark")
println("="^60)

# Globalisierungs-Konfigurationen
solver_configs = Dict{String, Function}()

# Verschiedene Kombinationen von Globalisierung und SOC
strategies = [
    ("No_Globalization", false, false),
    ("LineSearch_Only", true, false),
    ("SOC_Only", false, true),
    ("LineSearch_SOC", true, true)
]

for (name, use_glob, use_soc) in strategies
    solver_configs["SQP-$name"] = nlp -> sqp_method(nlp, Settings(
        use_globalization=use_glob,
        use_soc=use_soc,
        qp_solver=:osqp,  # Bester QP-Solver
        hessian_convexification=:lm,  # Beste Konvexifizierung
        verbose=false,
        max_iter=100,
        tol=1e-6,
        # Spezialisierte SOC-Parameter
        soc_improvement_threshold=0.5,
        # Filter-Parameter für Liniensuche
        γh=1e-5,
        γf=1e-5,
        γα=0.05,
        sh=1.1,
        sf=2.3,
        ηf=1e-4
    ))
end

# Problem-Filter
problem_filter() = filter_problems_by_size(3, 50, 1)  # Mit Nebenbedingungen

println("Globalisierungs-Strategien Vergleich wird gestartet...")

# Teste auf beiden Problemsets
for use_test_problems in [true, false]
    problem_set_name = use_test_problems ? "Testprobleme" : "OptimizationProblems"
    println("\n" * "="^40)
    println("Teste auf: $problem_set_name")
    println("="^40)

    if !use_test_problems && !SQPPackage.Benchmarking.OPTIMIZATION_PROBLEMS_AVAILABLE
        println("OptimizationProblems.jl nicht verfügbar, überspringe...")
        continue
    end

    for metric in [:total_time, :function_evals]
        println("\nMetrik: $metric")

        results, τ_values, profiles = run_benchmark_template(
            solver_configs,
            problem_filter,
            metric=metric,
            save_prefix="globalization_strategies",
            results_dir=joinpath(@__DIR__, "..", "results"),
            use_test_problems=use_test_problems,
            max_problems=30
        )

        println("✓ Globalisierung Benchmark ($metric) abgeschlossen")
    end
end

println("\n" * "="^60)
println("✓ Alle Globalisierungs-Strategien Benchmarks abgeschlossen!")
println("Results gespeichert in: results/")
println("="^60)