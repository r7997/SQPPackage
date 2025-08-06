#!/usr/bin/env julia

"""
Large-scale benchmark comparing SQP configurations (with/without SOC) with Ipopt.
"""

# Add the parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using SQPPackage, SQPPackage.Benchmarking
using Plots
using Printf
using Statistics

# Try to load comparison solvers
IPOPT_AVAILABLE = false
try
    using NLPModelsIpopt
    global IPOPT_AVAILABLE = true
catch
    global IPOPT_AVAILABLE = false
    println("Warning: Ipopt not available")
end

# Try to load OptimizationProblems if available
OPTIMIZATION_PROBLEMS_AVAILABLE = false
try
    using OptimizationProblems
    global OPTIMIZATION_PROBLEMS_AVAILABLE = true
catch
    global OPTIMIZATION_PROBLEMS_AVAILABLE = false
    println("Warning: OptimizationProblems not available")
end

println("="^60)
println("Large-Scale Benchmark: SQP (with/without SOC) vs Ipopt")
println("="^60)

function get_problem(problem_name, use_test_problems)
    if use_test_problems
        return eval(Symbol("create_", problem_name))()
    else
        return getfield(OptimizationProblems.ADNLPProblems, problem_name)()
    end
end

function filter_optimization_problems(min_vars, max_vars, has_constraints=false)
    try
        if !OPTIMIZATION_PROBLEMS_AVAILABLE
            return Symbol[]
        end
        meta = OptimizationProblems.meta
        if has_constraints
            filtered = meta[(min_vars .<= meta.nvar .<= max_vars) .& (meta.ncon .> 0), [:name]]
        else
            filtered = meta[(min_vars .<= meta.nvar .<= max_vars), [:name]]
        end
        return Symbol.(filtered.name)
    catch
        return Symbol[]
    end
end

function performance_profile(times, names; title="Performance Profile", logscale=false)
    τ = 10 .^ range(0, 2, length=100)
    plt = Plots.plot(xlabel="τ", ylabel="ρ(τ)", title=title, legend=:bottomright)
    
    for (i, solver) in enumerate(names)
        r = times[:,i] ./ minimum(times, dims=2)
        r[isnan.(r)] .= Inf
        ρ = [count(r .<= t) / size(times,1) for t in τ]
        Plots.plot!(plt, τ, ρ, label=solver, lw=2)
    end
    
    logscale && Plots.plot!(plt, xscale=:log10)
    return plt
end

function large_benchmark_comparison()
    println("="^60)
    println("Large-Scale Benchmark: SQP (with/without SOC) vs Ipopt")
    println("="^60)

    # Use larger problems from OptimizationProblems.jl
    test_problem_names = filter_optimization_problems(250, 1000, true)
    if length(test_problem_names) > 50
        test_problem_names = test_problem_names[1:50]
    end

    println("Testing on $(length(test_problem_names)) problems")

    # Define solvers to compare
    solvers = Vector{Tuple{String, Union{Settings, Nothing}}}()
    
    push!(solvers, ("SQP_with_SOC", Settings(
        qp_solver=:osqp,
        hessian_convexification=:lm,
        use_globalization=true,
        use_soc=true,
        tol=1e-6,
        max_iter=1000,
        verbose=false
    )))
    
    push!(solvers, ("SQP_without_SOC", Settings(
        qp_solver=:osqp,
        hessian_convexification=:lm,
        use_globalization=true,
        use_soc=false,
        tol=1e-6,
        max_iter=1000,
        verbose=false
    )))
    
    if IPOPT_AVAILABLE
        push!(solvers, ("Ipopt", nothing))
    end

    # Storage for results
    results = Dict{String, Dict{Symbol, Dict{Symbol, Any}}}()
    for (solver_name, _) in solvers
        results[solver_name] = Dict{Symbol, Dict{Symbol, Any}}()
    end

    # Storage for timing results
    n_problems = length(test_problem_names)
    n_solvers = length(solvers)
    times = Matrix{Float64}(undef, n_problems, n_solvers)

    # Test each solver
    for (solver_idx, (solver_name, settings)) in enumerate(solvers)
        println("\nTesting solver: $solver_name")
        println("-"^40)

        for (prob_idx, problem_name) in enumerate(test_problem_names)
            print("  Problem $problem_name: ")

            try
                # Create problem
                nlp = get_problem(problem_name, false)

                start_time = time()
                
                if startswith(solver_name, "SQP")
                    stats = sqp_method(nlp, settings)
                    elapsed = time() - start_time
                    success = (stats.sqp_status == kkt_point)
                elseif solver_name == "Ipopt" && IPOPT_AVAILABLE
                    stats = ipopt(nlp, print_level=0, tol=1e-6, max_iter=1000)
                    elapsed = time() - start_time
                    success = (stats.status in [:first_order, :acceptable])
                else
                    elapsed = Inf
                    success = false
                end

                # Store results
                results[solver_name][problem_name] = Dict(
                    :time => elapsed,
                    :success => success
                )

                # Store time for performance profile
                times[prob_idx, solver_idx] = success ? elapsed : Inf

                if success
                    println("SUCCESS ($(round(elapsed, digits=3))s)")
                else
                    println("FAILED")
                end

            catch e
                println("ERROR: $e")
                results[solver_name][problem_name] = Dict(
                    :time => Inf,
                    :success => false
                )
                times[prob_idx, solver_idx] = Inf
            end
        end
    end

    # Create performance profile
    solver_names = [name for (name, _) in solvers]
    plt = performance_profile(times, solver_names, 
                            title="SQP SOC Comparison",
                            logscale=true)

    # Save results
    results_dir = joinpath(@__DIR__, "..", "results")
    if !isdir(results_dir)
        mkpath(results_dir)
    end

    savefig(plt, joinpath(results_dir, "sqp_soc_comparison.pdf"))
    println("\nPerformance profile saved to results/sqp_soc_comparison.pdf")

    # Print summary statistics
    println("\n" * "="^60)
    println("FINAL BENCHMARK SUMMARY")
    println("="^60)

    for (solver_name, _) in solvers
        solved_count = 0
        total_time = 0.0
        successful_times = Float64[]

        for (problem_name, result) in results[solver_name]
            if result[:success]
                solved_count += 1
                total_time += result[:time]
                push!(successful_times, result[:time])
            end
        end

        total_problems = length(test_problem_names)
        success_rate = (solved_count / total_problems) * 100
        avg_time = solved_count > 0 ? total_time / solved_count : 0.0

        println("\n$solver_name:")
        println("  Problems solved: $solved_count / $total_problems ($(round(success_rate, digits=1))%)")
        println("  Average time: $(round(avg_time, digits=3))s")
        println("  Total time: $(round(total_time, digits=3))s")

        if !isempty(successful_times)
            println("  Median time: $(round(median(successful_times), digits=3))s")
            println("  Min time: $(round(minimum(successful_times), digits=3))s")
            println("  Max time: $(round(maximum(successful_times), digits=3))s")
        end
    end

    println("\n" * "="^60)
    println("Benchmark Completed!")
    println("="^60)
end

# Run the benchmark
large_benchmark_comparison()