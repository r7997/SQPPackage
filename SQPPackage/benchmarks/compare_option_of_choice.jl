
#!/usr/bin/env julia

"""
Toleranz-Einstellungen Vergleichsbenchmark
Vergleicht verschiedene Toleranz-Einstellungen für die SQP-Methode
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using SQPPackage
include("run_benchmark_template.jl")

println("="^60)
println("Toleranz-Einstellungen Vergleichsbenchmark")
println("="^60)

# Toleranz-Konfigurationen
solver_configs = Dict{String, Function}()

# Verschiedene Toleranz-Einstellungen
tolerance_options = [
    ("Strict_1e-8", 1e-8),
    ("Standard_1e-6", 1e-6),
    ("Relaxed_1e-4", 1e-4),
    ("Loose_1e-3", 1e-3)
]

for (name, tol_val) in tolerance_options
    solver_configs["SQP-$name"] = nlp -> sqp_method(nlp, Settings(
        tol=tol_val,
        qp_solver=:osqp,  # Bester QP-Solver
        hessian_convexification=:lm,  # Beste Konvexifizierung
        use_globalization=true,  # Beste Globalisierung
        use_soc=true,
        verbose=false,
        max_iter=100,
        # OSQP-Toleranz entsprechend anpassen
        osqp_settings=Dict(
            "verbose" => false,
            "max_iter" => 1000,
            "eps_abs" => tol_val,
            "eps_rel" => tol_val
        )
    ))
end

# Problem-Filter
problem_filter() = filter_problems_by_size(3, 50, 0)

println("Toleranz-Einstellungen Vergleich wird gestartet...")

# Teste auf beiden Problemsets
for use_test_problems in [true, false]
    problem_set_name = use_test_problems ? "Testprobleme" : "OptimizationProblems"
    println("\n" * "="^40)
    println("Teste auf: $problem_set_name")
    println("="^40)
    
    if !use_test_problems && !OPTIMIZATION_PROBLEMS_AVAILABLE
        println("OptimizationProblems.jl nicht verfügbar, überspringe...")
        continue
    end
    
    for metric in [:total_time, :function_evals]
        println("\nMetrik: $metric")
        
        results, τ_values, profiles = run_benchmark_template(
            solver_configs, 
            problem_filter,
            metric=metric,
            save_prefix="tolerance_comparison",
            results_dir=joinpath(@__DIR__, "..", "results"),
            use_test_problems=use_test_problems,
            max_problems=30
        )
        
        println("✓ Toleranz Benchmark ($metric) abgeschlossen")
    end
end

println("\n" * "="^60)
println("✓ Alle Toleranz-Einstellungen Benchmarks abgeschlossen!")
println("Results gespeichert in: results/")
println("="^60)
