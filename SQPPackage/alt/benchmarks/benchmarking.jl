module Benchmarking
using LinearAlgebra, Printf
import Plots
using SQPPackage.StatsModule, SQPPackage.ResidualsModule
using ADNLPModels

export get_primal_residual, get_dual_residual, performance_profile
@@ -39,23 +38,69 @@ function get_dual_residual(nlp::ADNLPModel, x::Vector, y::Vector, zL::Vector, zU
end

function performance_profile(times, names; title="Performance Profile", logscale=false)
    # Ersetze logspace durch eine korrekte geometrische Sequenz
    τ = 10 .^ range(0, 2, length=100)  # Werte von 10^0=1 bis 10^2=100
    # Logarithmisch verteilte τ-Werte
    τ = exp.(range(log(1), log(100), length=100))

    plt = Plots.plot(xlabel="τ", ylabel="ρ(τ)", title=title, legend=:bottomright)
    # Erstelle Plot mit besserem Styling
    plt = Plots.plot(
        xlabel="Performance Ratio τ", 
        ylabel="Fraction of Problems Solved ρ(τ)", 
        title=title, 
        legend=:bottomright,
        grid=true,
        size=(800, 600),
        dpi=300
    )

    # Farben für verschiedene Solver
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray]

    for (i, solver) in enumerate(names)
        r = times[:,i] ./ minimum(times, dims=2)
        # Setze NaN/Inf-Werte auf eine hohe Zahl
        r[isnan.(r)] = Inf
        ρ = [count(r .<= t) / size(times,1) for t in τ]
        Plots.plot!(plt, τ, ρ, label=solver, lw=2)
        # Identifiziere erfolgreiche Läufe
        finite_mask = isfinite.(times[:, i])

        if !any(finite_mask)
            # Keine erfolgreichen Läufe
            Plots.plot!(plt, τ, zeros(length(τ)), 
                       label="$solver (keine Lösungen)", 
                       lw=3, 
                       linestyle=:dash,
                       color=colors[mod(i-1, length(colors)) + 1])
            continue
        end

        # Berechne Performance-Ratios
        r = fill(Inf, size(times, 1))
        for j in 1:size(times, 1)
            if any(isfinite.(times[j, :]))  # Mindestens ein Solver hat das Problem gelöst
                min_time = minimum(times[j, isfinite.(times[j, :])])
                if finite_mask[j]
                    r[j] = times[j, i] / min_time
                end
            end
        end

        # Berechne ρ(τ)
        ρ = [count(r .<= t) / size(times, 1) for t in τ]

        Plots.plot!(plt, τ, ρ, 
                   label=solver, 
                   lw=3,
                   color=colors[mod(i-1, length(colors)) + 1])
    end

    # Achsenkonfiguration
    Plots.ylims!(plt, 0, 1)
    if logscale
        Plots.plot!(plt, xscale=:log10)
        Plots.xlims!(plt, 1, 100)
    else
        Plots.xlims!(plt, 1, 20)
    end

    # Referenzlinien
    Plots.hline!(plt, [0.5, 0.8], color=:gray, linestyle=:dot, alpha=0.5, label="")

    return plt
end
end