module Benchmarking

using LinearAlgebra, Printf
using Plots
using ..SQPPackage: Stats, SQPStatus, kkt_point, constraint_violation

export performance_profile

function performance_profile(times, names; title="Performance Profile", logscale=false)
    τ = exp.(range(log(1), stop=log(100), length=100))
    plt = plot(
        xlabel="Performance Ratio τ", 
        ylabel="Fraction of Problems Solved ρ(τ)", 
        title=title, 
        legend=:bottomright,
        grid=true,
        size=(800, 600),
        dpi=300
    )
    
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray]
    
    for (i, solver) in enumerate(names)
        finite_mask = isfinite.(times[:, i])
        
        if !any(finite_mask)
            plot!(plt, τ, zeros(length(τ)), 
                   label="$solver (no solutions)", 
                   lw=3, linestyle=:dash,
                   color=colors[mod(i-1, length(colors)) + 1])
            continue
        end
        
        r = fill(Inf, size(times, 1))
        for j in 1:size(times, 1)
            if any(isfinite.(times[j, :]))
                min_time = minimum(times[j, isfinite.(times[j, :])])
                if finite_mask[j]
                    r[j] = times[j, i] / min_time
                end
            end
        end
        
        ρ = [count(r .<= t) / size(times, 1) for t in τ]
        plot!(plt, τ, ρ, 
               label=solver, 
               lw=3,
               color=colors[mod(i-1, length(colors)) + 1])
    end
    
    ylims!(plt, 0, 1)
    if logscale
        plot!(plt, xscale=:log10)
        xlims!(plt, 1, 100)
    else
        xlims!(plt, 1, 20)
    end
    
    hline!(plt, [0.5, 0.8], color=:gray, linestyle=:dot, alpha=0.5, label="")
    return plt
end

end