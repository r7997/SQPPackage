module SettingsModule

using Parameters: @with_kw
export Settings

@with_kw mutable struct Settings
    max_iter::Int = 200
    tol::Float64 = 1e-8
    verbose::Bool = false
    regularization::Float64 = 1e-8
    min_eigenvalue::Float64 = 1e-8
    use_regularization::Bool = true
    hessian_convexification::Symbol = :lm
    epsilon::Float64 = 1e-8
    use_globalization::Bool = true
    use_advanced_termination::Bool = false
    use_soc::Bool = false
    soc_improvement_threshold::Float64 = 0.5
    hmax_factor::Float64 = 1.0
    hmin_factor::Float64 = 1e-4
    γh::Float64 = 1e-5
    γf::Float64 = 1e-5
    δ::Float64 = 1.0
    γα::Float64 = 0.05
    sh::Float64 = 1.1
    sf::Float64 = 2.3
    ηf::Float64 = 1e-4
    qp_solver::Symbol = :osqp
    osqp_settings::Dict = Dict(  
        "verbose" => false,
        "max_iter" => 10000,
        "eps_abs" => 1e-6,
        "eps_rel" => 1e-6
    )
    clarabel_settings::Dict = Dict(:verbose => false)
    benchmark::Bool = false
    benchmark_file::String = "benchmark_results.csv"
end

end