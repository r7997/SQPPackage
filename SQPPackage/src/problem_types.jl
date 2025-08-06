module ProblemTypes

# Exportiere die abstrakten Typen und Strukturen, damit sie außerhalb dieses Moduls sichtbar sind.
export AbstractNLPModel, OptimizationProblem


abstract type AbstractNLPModel end

mutable struct OptimizationProblem <: AbstractNLPModel
    meta::NamedTuple # Metadaten des Optimierungsproblems
    obj_func::Function # Zielfunktion f(x)
    grad_func::Function # Gradient der Zielfunktion ∇f(x)
    hess_func::Function # Hesse-Matrix der Lagrange-Funktion ∇²ₓₓL(x, λ)
    cons_func::Function # Nebenbedingungen c(x)
    jac_func::Function # Jacobi-Matrix der Nebenbedingungen ∇c(x)
end

end # Ende des Moduls ProblemTypes