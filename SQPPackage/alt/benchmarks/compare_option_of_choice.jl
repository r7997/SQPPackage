#!/usr/bin/env julia

"""
Benchmark script to compare a custom algorithmic option of choice.
This example compares different tolerance settings.
"""

# Add the parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using SQPPackage
using Plots
using Printf
using OptimizationProblems

println("="^60)
println("Tolerance Settings Comparison Benchmark")
println("="^60)

function run_tolerance_comparison()
    # Test both problem sets
    problem_sets = [
        ("TestProblems", true),
        ("OptimizationProblems", false)
    ]

    # Different tolerance settings to compare
    tolerance_options = [
        ("Strict_1e-8", 1e-8),
        ("Standard_1e-6", 1e-6),
        ("Relaxed_1e-4", 1e-4),
        ("Loose_1e-3", 1e-3)
    ]

    for (set_name, use_test_problems) in problem_sets
        println("\nTesting on $set_name...")

        # Get problems based on set type
        if use_test_problems
            test_problems = [:P1, :P2, :P3, :P4, :P5, :P6, :P7, :P8]
        else
            try
                meta = OptimizationProblems.meta
                filtered = meta[(3 .<= meta.nvar .<= 50) .& (meta.ncon .> 0), [:name]]
                test_problems = Symbol.(filtered.name[1:min(20, length(filtered.name))])
            catch
                println("OptimizationProblems not available or error occurred, skipping...")
                continue
            end
        end

        # Storage for timing results
        n_problems = length(test_problems)
        n_options = length(tolerance_options)
        times = Matrix{Float64}(undef, n_problems, n_options)

        println("Testing $(n_problems) problems with $(n_options) tolerance settings...")

        for (i, problem_name) in enumerate(test_problems)
            print("Problem $problem_name: ")

            # Create problem
            try
                if use_test_problems
                    nlp = eval(Symbol("create_", problem_name))()
                else
                    nlp = getfield(OptimizationProblems.ADNLPProblems, problem_name)()
                end

                for (j, (name, tol_val)) in enumerate(tolerance_options)
                    try
                        # Configure settings with specific tolerance
                        settings = Settings(
                            tol=tol_val,
                            verbose=false,
                            max_iter=100,
                            qp_solver=:osqp,
                            hessian_convexification=:lm,
                            use_globalization=true
                        )

                        # Time the solver
                        start_time = time()
                        stats = sqp_method(nlp, settings)
                        elapsed = time() - start_time

                        # Store time if successful, otherwise infinity
                        if stats.sqp_status == kkt_point
                            times[i, j] = elapsed
                        else
                            times[i, j] = Inf
                        end

                        print("$(name): $(round(elapsed, digits=3))s ")
                    catch e
                        times[i, j] = Inf
                        print("$(name): FAIL ")
                    end
                end
                println()
            catch e
                println("ERROR creating problem")
                times[i, :] .= Inf
            end
        end

        # Create performance profile
        option_names = [opt[1] for opt in tolerance_options]
        plt = performance_profile(times, option_names, 
                                title="Tolerance Settings Comparison - $set_name",
                                logscale=true)

        # Create results directory if it doesn't exist
        results_dir = joinpath(@__DIR__, "..", "results")
        if !isdir(results_dir)
            mkpath(results_dir)
        end

        # Save plot
        filename = joinpath(results_dir, "tolerance_comparison_$(lowercase(set_name)).pdf")
        savefig(plt, filename)
        println("Results saved to: $filename")

        # Print summary statistics
        println("\nSummary for $set_name:")
        for (j, (name, tol_val)) in enumerate(tolerance_options)
            solved = sum(isfinite.(times[:, j]))
            total = size(times, 1)
            if solved > 0
                avg_time = mean(times[isfinite.(times[:, j]), j])
                println("  $name (tol=$tol_val): $solved/$total problems solved, avg time: $(round(avg_time, digits=3))s")
            else
                println("  $name (tol=$tol_val): $solved/$total problems solved")
            end
        end
    end
end

# Run the comparison
run_tolerance_comparison()

println("\n" * "="^60)
println("Tolerance Settings Comparison Completed!")
println("="^60)