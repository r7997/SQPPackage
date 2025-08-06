#!/usr/bin/env julia
"""
Minimal Example for SQPPackage.jl

Demonstriert die grundlegende Verwendung des SQP-Lösers und erstellt einen Konvergenzplot.
"""

# Aktuelles Verzeichnis zum Ladepfad hinzufügen und Pakete laden
push!(LOAD_PATH, ".")
using SQPPackage
using Plots

println("="^60)
println("SQPPackage.jl - Minimal Example")
println("="^60)

# Modul für alle SQP-Testprobleme P1 bis P16 mit ADNLPModels
module SQPTestProblems


using ADNLPModels
using LinearAlgebra

export create_P1, create_P2, create_P3, create_P4, create_P5, create_P6,
       create_P7, create_P8, create_P9, create_P10, create_P11, create_P12,
       create_P13, create_P14, create_P15, create_P16

# Hilfsfunktionen (könnten auch ausgelagert werden)
f_P1(x) = x[1]^2 + 2x[2]^2
c_P1(x) = Float64[]
function create_P1()
    x0 = [1.0, 1.0]
    return ADNLPModel(f_P1, x0; c = c_P1)
end

f_P2(x) = x[1]^2 + 2x[2]^2
c_P2(x) = [x[1] + 2x[2]^2 - 10.0,
          2x[1] + x[2] - 9.0]
function create_P2()
    x0 = [1.2, 5.4]
    return ADNLPModel(f_P2, x0; 
        c = c_P2, 
        lcon = [0.0, 0.0],  # Beide Gleichheitsnebenbedingungen
        ucon = [0.0, 0.0]
    )
end

f_P3(x) = 2x[1]^4 + 4x[1]^2 - x[1]*x[2] + 6x[2]^2
c_P3(x) = [2x[1] - x[2] + 4.0]
function create_P3()
    x0 = [4.0, 4.0]
    return ADNLPModel(f_P3, x0; c = c_P3, lcon = [0.0], ucon = [0.0])
end

f_P4(x) = x[1]*x[4]*(x[1] + x[2] + x[3]) + x[3]
c_P4(x) = [x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 - 40.0,
          x[1] + x[2] + x[3] + x[4] - 25.0]
function create_P4()
    x0 = [3.0, 3.0, 3.0, 3.0]

    # Zielfunktion
    #=
    min x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3] -> min x[3]+h(x) mit h(x) = x[1] * x[4] * (x[1] + x[2] + x[3])

    Der Solver erwartet die Nebenbedingung in form  g(x) <= 0 haben und nicht g(x) >= 0.
    
    =#
    h(x) = x[1] * x[4] * (x[1] + x[2] + x[3])
    f(x) = h(x) + x[3]

    # Nebenbedingungen: alle in einer Funktion
    c(x) = [
        sum(x.^2) - 40,          # x₁² + x₂² + x₃² + x₄² = 40
        25 - prod(x)             # x₁x₂x₃x₄ ≥ 25 → 25 - x₁x₂x₃x₄ ≤ 0   
    ]

    # Variable Grenzen
    lvar = [1.0, 1.0, 1.0, 1.0]     #  xᵢ ≥ 1
    uvar = [5.0, 5.0, 5.0, 5.0]     # xᵢ ≤ 5
    
    # NB Grenzen 
    lcon = [0.0, -Inf]              
    ucon = [0.0, 0.0]               

    return ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)
end

f_P5(x) = exp(x[1]) * (4x[1]^2 + 2x[2] + 4x[1]*x[2] + 2x[1] + 1)
c_P5(x) = [x[1] + x[2] - x[1]*x[2] - 3/2,
          x[1]*x[2] + 10.0]
function create_P5()
    x0 = [2.0, -4.0]
    f(x) = exp(x[1]) * (4*x[1]^2 + 2*x[2] + 4*x[1]*x[2] + 2*x[1] + 1)

    c(x) = [
        3/2-(x[1] + x[2] - x[1]*x[2]),  # ≥ 1.5
        -10-x[1]*x[2]                 # ≥ -10
    ]

    # Variable Grenzen
    lvar = [-5.0, -5.0]    # -5 ≤ x₁ ≤ 3, -5 ≤ x₂ ≤ 3
    uvar = [3.0, 3.0]
    
    # Constraint Grenzen (b constraints ≤ 0)
    lcon = [-Inf, -Inf]     #  c(x) ≤ 0
    ucon = [0.0, 0.0]
    
    
    # Create ADNLPModel
    nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)
end

f_P6(x) = exp((x[1] - 2)^2) + (x[2] - 2)^2
c_P6(x) = [x[1]^2 + x[2]^2 - 9.0]
function create_P6()
    x0 = [7.0, 7.0]
    return ADNLPModel(f_P6, x0; c = c_P6, lcon = [-Inf], ucon = [0.0],
                      lvar = [5.0, 5.0], uvar = [10.0, 10.0])
end

f_P7(x) = -(x[1]^2 + x[2]^2)
c_P7(x) = [-x[1] + x[2]^2]
function create_P7()
    x0 = [0.1, 0.1]
    return ADNLPModel(f_P7, x0; c = c_P7, lcon = [0.0], ucon = [Inf],
                      lvar = [0.0, 0.0], uvar = [10.0, 10.0])
end

f_P8(x) = (x[1]-3)^2 + (x[2]-2)^2 + (x[3]-1)^2
c_P8(x) = [x[1] + x[2] + x[3] - 6.0,
          x[1]^2 - x[2],
          x[3]^2 - x[1]*x[2],
          x[1] - 2x[2] + x[3],
          x[1]^2 + x[2]^2 + x[3]^2 - 10.0,
          x[1]*x[3] - x[2] + 1.0]
function create_P8()
    x0 = [0.0, 0.0, 0.0]
    return ADNLPModel(f_P8, x0; c = c_P8,
                      lcon = [0.0, 0.0, 0.0, -1.0, -Inf, 0.0],
                      ucon = [0.0, 0.0, 0.0, 2.0, 0.0, Inf],
                      lvar = [-10.0, -10.0, -10.0], uvar = [10.0, 10.0, 10.0])
end

f_P9(x) = 0.05*x[1]^2 + log(cosh(x[1]))
c_P9(x) = Float64[]
function create_P9()
    x0 = [10.0]
    return ADNLPModel(f_P9, x0)
end

f_P10(x) = 2*(x[1]^2 + x[2]^2 - 1) - x[1]
c_P10(x) = [x[1]^2 + x[2]^2 - 1.0]
function create_P10()
    x0 = [cos(0.1*pi), sin(0.1*pi)]
    return ADNLPModel(f_P10, x0; c = c_P10, lcon = [0.0], ucon = [0.0],
                      lvar = [-10.0, -10.0], uvar = [10.0, 10.0])
end


function create_P11()
    f(x) = -x[1]
    c(x) = [x[2] - x[1]^3 - x[3]^2,
           x[1]^2 - x[2] - x[4]^2]
    x0 = [0.5, 0.5, 0.5, 0.5]
    return ADNLPModel(f, x0; c = c, lcon = [0.0, 0.0], ucon = [0.0, 0.0])
end

function create_P12()
    f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
    c(x) = [x[1] - 2x[2] + 1,
           -0.25*x[1]^2 - x[2]^2 + 1]
    x0 = [2.0, 0.0]
    return ADNLPModel(f, x0; c = c, lcon = [0.0, 0.0], ucon = [0.0, Inf])
end

function create_P13()
    f(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2
    c(x) = [x[1] - x[2]^2,
           x[1]^2 + x[2]]
    x0 = [-1.0, 1.0]
    return ADNLPModel(f, x0; c = c, lcon = [0.0, 0.0], ucon = [Inf, Inf],
                      lvar = [-2.0, -Inf], uvar = [0.5, 1.0])
end

function create_P14()
    f(x) = x[1] + 2x[2] + 4x[5] + exp(x[1]*x[4])
    c(x) = [x[1] + 2x[2] + 5x[5] - 6,
           x[1] + x[2] + x[3] - 3,
           x[4] + x[5] + x[6] - 2,
           x[1] + x[4] - 1,
           x[2] + x[5] - 2,
           x[3] + x[6] - 2]
    x0 = ones(6)
    return ADNLPModel(f, x0;
        c = c,
        lcon = zeros(6),
        ucon = zeros(6),
        lvar = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        uvar = [1.0, Inf, Inf, Inf, Inf, Inf]
    )
end

function create_P15()
    f(x) = 0.5*((x[1] - 1)^2 + x[2]^2)
    c(x) = [-x[1] + 2x[2]^2]
    x0 = [0.5, 0.5]
    return ADNLPModel(f, x0; c = c, lcon = [0.0], ucon = [0.0])
end

function create_P16()
    f(x) = 0.5*((x[1] - 2)^4 + (x[2] - 2)^4)
    x0 = [0.0, 0.0]
    return ADNLPModel(f, x0;
        lvar = [-1.0, -1.0],
        uvar = [1.0, 1.0])
end

end # module

