using SQPPackage

# Hessian-Konvexifizierungs-Konfigurationen
solver_configs = Dict{String, Function}(
    "No-Convexification" => nlp -> sqp_method(nlp, Settings(convexify_method=:none, verbose=false)),
    "Eigenvalue-Project" => nlp -> sqp_method(nlp, Settings(convexify_method=:project, verbose=false)),
    "Eigenvalue-Mirror" => nlp -> sqp_method(nlp, Settings(convexify_method=:mirror, verbose=false)),
    "Regularization" => nlp -> sqp_method(nlp, Settings(convexify_method=:regularize, verbose=false))
)

problem_filter() = filter_problems_by_size(3, 50, 0)

println("Hessian-Konvexifizierung Vergleich wird gestartet...")

for metric in [:total_time, :function_evals]
    results, τ_values, profiles = run_benchmark_template(
        solver_configs, 
        problem_filter,
        metric=metric,
        save_prefix="hessian_convexification",
        results_dir="../results"
    )
    
    println("✓ Hessian-Konvexifizierung Benchmark ($metric) abgeschlossen")
end

println("✓ Alle Hessian-Konvexifizierung Benchmarks abgeschlossen!")