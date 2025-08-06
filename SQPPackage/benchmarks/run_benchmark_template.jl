
#!/usr/bin/env julia

"""
Benchmark Template für systematische Solver-Vergleiche
Basiert auf Performance Profiles nach Dolan & Moré (2002)
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using SQPPackage
using SQPPackage.Benchmarking: performance_profile
using Plots
using Printf
using Statistics
using LinearAlgebra

# Versuche OptimizationProblems zu laden
OPTIMIZATION_PROBLEMS_AVAILABLE = false
try
    using OptimizationProblems
    global OPTIMIZATION_PROBLEMS_AVAILABLE = true
catch
    global OPTIMIZATION_PROBLEMS_AVAILABLE = false
    println("Warning: OptimizationProblems.jl nicht verfügbar")
end

# Versuche konkurrierende Solver zu laden
IPOPT_AVAILABLE = false
MADNLP_AVAILABLE = false
try
    using NLPModelsIpopt
    global IPOPT_AVAILABLE = true
catch
    println("Warning: Ipopt nicht verfügbar")
end

try
    using MadNLP
    global MADNLP_AVAILABLE = true
catch
    println("Warning: MadNLP nicht verfügbar")
end

export run_benchmark_template, filter_problems_by_size, get_problem

"""
Filtert Probleme nach Größe aus OptimizationProblems.jl
"""
function filter_problems_by_size(min_vars::Int, max_vars::Int, min_cons::Int=0)
    if !OPTIMIZATION_PROBLEMS_AVAILABLE
        return Symbol[]
    end
    
    try
        meta = OptimizationProblems.meta
        if min_cons > 0
            filtered = meta[(min_vars .<= meta.nvar .<= max_vars) .& (meta.ncon .>= min_cons), [:name]]
        else
            filtered = meta[(min_vars .<= meta.nvar .<= max_vars), [:name]]
        end
        return Symbol.(filtered.name)
    catch e
        println("Fehler beim Filtern der Probleme: $e")
        return Symbol[]
    end
end

"""
Erstellt ein Problem basierend auf Namen und Typ
"""
function get_problem(name::Symbol, use_test_problems::Bool=false)
    if use_test_problems
        # Verwende lokale Testprobleme
        include(joinpath(@__DIR__, "..", "testproblems", "testproblems.jl"))
        return eval(Symbol("create_", name))()
    else
        # Verwende OptimizationProblems.jl
        if !OPTIMIZATION_PROBLEMS_AVAILABLE
            error("OptimizationProblems.jl nicht verfügbar")
        end
        return getfield(OptimizationProblems.ADNLPProblems, name)()
    end
end

"""
Hauptfunktion für Benchmark-Template
"""
function run_benchmark_template(
    solver_configs::Dict{String, Function},
    problem_filter::Function;
    metric::Symbol=:total_time,
    save_prefix::String="benchmark",
    results_dir::String="results",
    use_test_problems::Bool=false,
    max_problems::Int=50,
    timeout::Float64=300.0  # 5 Minuten Timeout pro Problem
)
    
    println("="^60)
    println("Benchmark: $save_prefix")
    println("Metrik: $metric")
    println("="^60)
    
    # Erstelle Results-Verzeichnis
    if !isdir(results_dir)
        mkpath(results_dir)
    end
    
    # Hole Probleme
    if use_test_problems
        test_problems = [:P1, :P2, :P3, :P4, :P5, :P6, :P7, :P8, :P9, :P10, :P11, :P12, :P13, :P14, :P15, :P16]
    else
        test_problems = problem_filter()
        if length(test_problems) > max_problems
            test_problems = test_problems[1:max_problems]
        end
    end
    
    println("Teste $(length(test_problems)) Probleme mit $(length(solver_configs)) Solvern")
    
    # Initialisiere Ergebnis-Matrizen
    n_problems = length(test_problems)
    n_solvers = length(solver_configs)
    times = Matrix{Float64}(undef, n_problems, n_solvers)
    function_evals = Matrix{Float64}(undef, n_problems, n_solvers)
    
    solver_names = collect(keys(solver_configs))
    results = Dict{String, Dict{Symbol, Any}}()
    
    # Teste jeden Solver
    for (solver_idx, (solver_name, solver_func)) in enumerate(solver_configs)
        println("\nTeste Solver: $solver_name")
        println("-"^40)
        
        results[solver_name] = Dict{Symbol, Any}()
        
        for (prob_idx, problem_name) in enumerate(test_problems)
            print("  Problem $problem_name: ")
            
            try
                # Erstelle Problem
                nlp = get_problem(problem_name, use_test_problems)
                
                # Zeitmessung
                start_time = time()
                
                # Führe Solver aus mit Timeout
                local stats
                try
                    stats = solver_func(nlp)
                    elapsed = time() - start_time
                    
                    # Überprüfe Erfolg
                    success = false
                    if hasfield(typeof(stats), :sqp_status)
                        success = (stats.sqp_status == kkt_point)
                    elseif hasfield(typeof(stats), :status)
                        success = (stats.status in [:first_order, :acceptable])
                    end
                    
                    # Speichere Metriken
                    if success && elapsed < timeout
                        times[prob_idx, solver_idx] = elapsed
                        
                        # Funktionsauswertungen (falls verfügbar)
                        if hasfield(typeof(stats), :neval_obj)
                            total_evals = stats.neval_obj + stats.neval_cons + stats.neval_grad + stats.neval_jac + stats.neval_hess
                            function_evals[prob_idx, solver_idx] = Float64(total_evals)
                        else
                            function_evals[prob_idx, solver_idx] = elapsed  # Fallback zu Zeit
                        end
                        
                        println("SUCCESS ($(round(elapsed, digits=3))s)")
                    else
                        times[prob_idx, solver_idx] = Inf
                        function_evals[prob_idx, solver_idx] = Inf
                        println("FAILED/TIMEOUT")
                    end
                    
                catch timeout_error
                    times[prob_idx, solver_idx] = Inf
                    function_evals[prob_idx, solver_idx] = Inf
                    println("TIMEOUT/ERROR")
                end
                
            catch e
                println("ERROR: $e")
                times[prob_idx, solver_idx] = Inf
                function_evals[prob_idx, solver_idx] = Inf
            end
        end
    end
    
    # Wähle Metrik für Performance Profile
    metric_data = (metric == :total_time) ? times : function_evals
    metric_name = (metric == :total_time) ? "Gesamtzeit" : "Funktionsauswertungen"
    
    # Erstelle Performance Profile
    τ_values = exp.(range(log(1), log(100), length=100))
    plt = performance_profile(metric_data, solver_names, 
                            title="Performance Profile: $save_prefix ($metric_name)",
                            logscale=true)
    
    # Speichere Plot
    problem_set_name = use_test_problems ? "testproblems" : "optimization_problems"
    filename = joinpath(results_dir, "$(save_prefix)_$(metric)_$(problem_set_name).pdf")
    savefig(plt, filename)
    println("\nPerformance Profile gespeichert: $filename")
    
    # Drucke Zusammenfassung
    println("\n" * "="^60)
    println("BENCHMARK ZUSAMMENFASSUNG")
    println("="^60)
    
    for (solver_name, _) in solver_configs
        solver_idx = findfirst(x -> x == solver_name, solver_names)
        solved_count = sum(isfinite.(metric_data[:, solver_idx]))
        total_problems = length(test_problems)
        success_rate = (solved_count / total_problems) * 100
        
        if solved_count > 0
            successful_values = metric_data[isfinite.(metric_data[:, solver_idx]), solver_idx]
            avg_metric = mean(successful_values)
            median_metric = median(successful_values)
            
            println("\n$solver_name:")
            println("  Gelöste Probleme: $solved_count / $total_problems ($(round(success_rate, digits=1))%)")
            println("  Durchschnitt $metric_name: $(round(avg_metric, digits=3))")
            println("  Median $metric_name: $(round(median_metric, digits=3))")
            println("  Min $metric_name: $(round(minimum(successful_values), digits=3))")
            println("  Max $metric_name: $(round(maximum(successful_values), digits=3))")
        else
            println("\n$solver_name:")
            println("  Gelöste Probleme: $solved_count / $total_problems (0.0%)")
        end
    end
    
    println("\n" * "="^60)
    
    return results, τ_values, plt
end
