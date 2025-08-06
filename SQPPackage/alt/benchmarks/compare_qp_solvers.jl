using SQPPackage
using OptimizationProblems

# QP-Solver Konfigurationen
solver_configs = Dict{String, Function}(
    "SQP-Ipopt" => nlp -> sqp_method(nlp, Settings(qp_solver=:ipopt, verbose=false)),
    "SQP-OSQP" => nlp -> sqp_method(nlp, Settings(qp_solver=:osqp, verbose=false)),
    "SQP-Clarabel" => nlp -> sqp_method(nlp, Settings(qp_solver=:clarabel, verbose=false))
)

# Testprobleme filtern
problem_filter() = filter_problems_by_size(3, 50, 0)

println("QP-Solver Vergleich wird gestartet...")

# Benchmark für beide Metriken
for metric in [:total_time, :function_evals]
    results, τ_values, profiles = run_benchmark_template(
        solver_configs, 
        problem_filter,
        metric=metric,
        save_prefix="qp_solvers_comparison",
        results_dir="../results"
    )
    
    println("✓ QP-Solver Benchmark ($metric) abgeschlossen")
end

println("✓ Alle QP-Solver Benchmarks abgeschlossen!")