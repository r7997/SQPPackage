
#!/usr/bin/env julia

"""
Hessian-Konvexifizierung Vergleichsbenchmark
Vergleicht verschiedene Regularisierungsstrategien
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using SQPPackage
include("run_benchmark_template.jl")

println("="^60)
println("Hessian-Konvexifizierung Vergleichsbenchmark")
println("="^60)

# Hessian-Konvexifizierungs-Konfigurationen
solver_configs = Dict{String, Function}()

# Verschiedene Konvexifizierungsmethoden
convex_methods = [:none, :lm, :project, :mirror, :gershgorin]

for method in convex_methods
    method_name = string(method)
    solver_configs["SQP-$method_name"] = nlp -> sqp_method(nlp, Settings(
        hessian_convexification=method,
        qp_solver=:osqp,  # Verwende besten QP-Solver aus vorherigem Benchmark
        verbose=false,
        max_iter=100,
        tol=1e-6,
        use_globalization=true,
        regularization=1e-8,
        use_regularization=(method != :none)
    ))
end

# Problem-Filter
problem_filter() = filter_problems_by_size(3, 50, 0)  # Auch unrestringierte Probleme

println("Hessian-Konvexifizierung Vergleich wird gestartet...")

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
            save_prefix="hessian_convexification",
            results_dir=joinpath(@__DIR__, "..", "results"),
            use_test_problems=use_test_problems,
            max_problems=30
        )
        
        println("✓ Hessian-Konvexifizierung Benchmark ($metric) abgeschlossen")
    end
end

println("\n" * "="^60)
println("✓ Alle Hessian-Konvexifizierung Benchmarks abgeschlossen!")
println("Results gespeichert in: results/")
println("="^60)
