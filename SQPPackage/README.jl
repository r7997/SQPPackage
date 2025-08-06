# SQPPackage.jl

Ein Julia-Paket zur Lösung nichtlinearer Optimierungsprobleme mit der Sequential Quadratic Programming (SQP) Methode.

## Überblick

Dieses Paket implementiert eine Filter-Line-Search SQP-Methode für nichtlineare Optimierungsprobleme der Form:

```
min  f(x)
s.t. cℓ ≤ c(x) ≤ cu
     xℓ ≤ x ≤ xu
```

## Hauptmerkmale

- **Mehrere QP-Löser**: Unterstützt OSQP, Clarabel, Ipopt und MadNLP
- **Hessian-Konvexifizierung**: Verschiedene Strategien (Levenberg-Marquardt, Projektion, Spiegelung)
- **Globalisierungsstrategie**: Filter-Line-Search mit Second-Order Corrections
- **Robuste Implementierung**: Behandelt schlecht konditionierte Probleme
- **Benchmarking-Tools**: Integrierte Leistungsanalyse

## Installation

```julia
using Pkg
Pkg.develop(path="./SQPPackage")
using SQPPackage
```

## Schnellstart

```julia
using SQPPackage, ADNLPModels

# Zielfunktion und Nebenbedingungen definieren
f(x) = x[1]^2 + 2*x[2]^2
c(x) = [x[1] + 2*x[2]^2 - 10.0, 2*x[1] + x[2] - 9.0]

# Problem erstellen
x0 = [1.2, 5.4]
nlp = ADNLPModel(f, x0; c = c, lcon = [0.0, 0.0], ucon = [0.0, 0.0])

# Solver-Einstellungen
settings = Settings(
    max_iter = 100,
    tol = 1e-6,
    verbose = true,
    qp_solver = :osqp
)

# Problem lösen
stats = sqp_method(nlp, settings)
```

## Konfiguration

### QP-Löser
- `:osqp` - OSQP (Standard)
- `:clarabel` - Clarabel
- `:ipopt` - Ipopt
- `:madnlp` - MadNLP

### Hessian-Konvexifizierung
- `:lm` - Levenberg-Marquardt (Standard)
- `:project` - Eigenwert-Projektion
- `:mirror` - Eigenwert-Spiegelung
- `:gershgorin` - Gershgorin-Kreis-Regularisierung

## Testprobleme

Das Paket enthält 16 Testprobleme (P1-P16):

```julia
prob = create_P1()  # Einfaches quadratisches Problem
stats = sqp_method(prob, Settings(verbose=true))
```

## Benchmarking

```julia
# Vergleich verschiedener QP-Löser
include("benchmarks/compare_qp_solvers.jl")

# Vergleich von Hessian-Konvexifizierungsmethoden
include("benchmarks/compare_hessian_convexification.jl")
```

## Algorithmus

Die SQP-Methode besteht aus:

1. **Quadratisches Teilproblem**: Linearisierung der Nebenbedingungen mit quadratischem Lagrange-Modell
2. **Hessian-Regularisierung**: Sicherstellung positiver Definitheit
3. **Filter-Line-Search**: Globale Konvergenz bei schneller lokaler Konvergenz
4. **Second-Order Corrections**: Verbesserte Konvergenz bei abgelehnten Schritten

## Abhängigkeiten 
- Julia ≥ 1.6
- NLPModels.jl, ADNLPModels.jl
- OSQP.jl, Clarabel.jl, Ipopt.jl, MadNLP.jl
- OptimizationProblems.jl (für Benchmarking)
(für genaurere Informationen siehe ULM Diagramm)

Entwickelt im Rahmen des Optimierungskurses der TU Braunschweig.