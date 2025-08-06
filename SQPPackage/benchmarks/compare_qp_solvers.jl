
#!/usr/bin/env julia

"""
QP-Solver Vergleichsbenchmark
Vergleicht OSQP, Clarabel, Ipopt und MadNLP als QP-Sublöser
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using SQPPackage
include("run_benchmark_template.jl")

println("="^60)
println("QP-Solver Vergleichsbenchmark")
println("="^60)

# QP-Solver Konfigurationen
solver_configs = Dict{String, Function}()

# SQP mit verschiedenen QP-Solvern
solver_configs["SQP-OSQP"] = nlp -> sqp_method(nlp, Settings(
    qp_solver=:osqp, 
    verbose=false,
    max_iter=100,
    tol=1e-6,
    hessian_convexification=:lm,
    use_globalization=true
))

solver_configs["SQP-Clarabel"] = nlp -> sqp_method(nlp, Settings(
    qp_solver=:clarabel, 
    verbose=false,
    max_iter=100,
    tol=1e-6,
    hessian_convexification=:lm,
    use_globalization=true
))

# Nur hinzufügen wenn verfügbar
try
    solver_configs["SQP-Ipopt"] = nlp -> sqp_method(nlp, Settings(
        qp_solver=:ipopt, 
        verbose=false,
        max_iter=100,
        tol=1e-6,
        hessian_convexification=:lm,
        use_globalization=true
    ))
catch
    println("Ipopt QP-Interface nicht verfügbar")
end

try
    solver_configs["SQP-MadNLP"] = nlp -> sqp_method(nlp, Settings(
        qp_solver=:madnlp, 
        verbose=false,
        max_iter=100,
        tol=1e-6,
        hessian_convexification=:lm,
        use_globalization=true
    ))
catch
    println("MadNLP QP-Interface nicht verfügbar")
end

# Problem-Filter für OptimizationProblems.jl
problem_filter() = filter_problems_by_size(3, 50, 1)  # 3-50 Variablen, mindestens 1 Nebenbedingung

println("QP-Solver Vergleich wird gestartet...")

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
    
    # Benchmark für beide Metriken
    for metric in [:total_time, :function_evals]
        println("\nMetrik: $metric")
        
        results, τ_values, profiles = run_benchmark_template(
            solver_configs, 
            problem_filter,
            metric=metric,
            save_prefix="qp_solvers_comparison",
            results_dir=joinpath(@__DIR__, "..", "results"),
            use_test_problems=use_test_problems,
            max_problems=30
        )
        
        println("✓ QP-Solver Benchmark ($metric) abgeschlossen")
    end
end

println("\n" * "="^60)
println("✓ Alle QP-Solver Benchmarks abgeschlossen!")
println("Results gespeichert in: results/")
println("="^60)
