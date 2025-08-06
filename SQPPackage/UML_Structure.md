
# UML Diagramm - SQPPackage.jl Struktur

## Klassendiagramm

### Hauptmodule

```
┌─────────────────┐
│   SQPPackage    │
├─────────────────┤
│ + sqp_method()  │
│ + Settings      │
│ + Stats         │
│ + SQPStatus     │
└─────────────────┘
```

### Core Classes/Structs

```
┌─────────────────────┐       ┌─────────────────────┐
│      Settings       │       │       Stats         │
├─────────────────────┤       ├─────────────────────┤
│ - max_iter: Int     │       │ - x: Vector{Float64}│
│ - tol: Float64      │       │ - y: Vector{Float64}│
│ - verbose: Bool     │       │ - z: Vector{Float64}│
│ - qp_solver: Symbol │       │ - obj_val: Float64  │
│ - use_soc: Bool     │       │ - inf_pr: Float64   │
│ - hessian_convex..  │       │ - inf_du: Float64   │
│ - osqp_settings     │       │ - sqp_status: Enum  │
├─────────────────────┤       │ - iter: Int         │
│ + Settings()        │       │ - total_time: Float │
└─────────────────────┘       └─────────────────────┘
```

### Enums

```
┌─────────────────────┐
│     SQPStatus       │
├─────────────────────┤
│ + kkt_point         │
│ + max_iter          │
│ + qp_failed         │
│ + infeasible        │
│ + restoration_phase │
└─────────────────────┘
```

### Module Struktur

```
SQPPackage
├── SettingsModule
│   └── Settings
├── StatsModule  
│   ├── Stats
│   └── SQPStatus
├── SQPMethod
│   ├── sqp_method()
│   ├── IterationData
│   └── SOCResult
├── ConvexifyModule
│   └── convexify_hessian!()
├── QPInterface
│   ├── AbstractQPSolver
│   ├── QPResult
│   ├── create_qp_solver()
│   └── solve_qp()
├── FilterLineSearchSQP
│   ├── Filter
│   ├── FilterPoint
│   └── filter_line_search()
├── ResidualsModule
│   └── compute_kkt_residuals()
├── ProblemTypes
│   ├── AbstractNLPModel
│   └── OptimizationProblem
└── Benchmarking
    └── performance_profile()
```

## Sequenzdiagramm - SQP Methode

```
User -> SQPPackage: sqp_method(nlp, settings)
SQPPackage -> QPInterface: create_qp_solver()
SQPPackage -> Filter: Filter()

loop [bis Konvergenz]
    SQPPackage -> NLPModel: obj(x), grad(x), cons(x), jac(x), hess(x)
    SQPPackage -> ResidualsModule: compute_kkt_residuals()
    SQPPackage -> ConvexifyModule: convexify_hessian!()
    SQPPackage -> QPInterface: solve_qp()
    
    alt [QP erfolgreich]
        SQPPackage -> FilterLineSearchSQP: filter_line_search()
        
        opt [use_soc aktiviert]
            SQPPackage -> SQPMethod: compute_soc_step()
            SQPPackage -> FilterLineSearchSQP: filter_line_search()
        end
        
        SQPPackage -> SQPPackage: update x, y, z
    else [QP fehlgeschlagen]
        SQPPackage -> Stats: sqp_status = qp_failed
    end
end

SQPPackage -> Stats: finalize_stats()
SQPPackage -> User: return Stats
```

## Abhängigkeitsdiagramm

```
External Dependencies:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ NLPModels   │  │ ADNLPModels │  │ LinearAlgebra│
└─────────────┘  └─────────────┘  └─────────────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
            ┌─────────────────┐
            │   SQPPackage    │
            └─────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
┌─────────┐       ┌─────────┐       ┌─────────┐
│  OSQP   │       │Clarabel │       │ Ipopt   │
└─────────┘       └─────────┘       └─────────┘
```

## Verwendungsbeispiel

```julia
# 1. Problem definieren
nlp = ADNLPModel(f, x0; c=c, lcon=lcon, ucon=ucon)

# 2. Einstellungen konfigurieren  
settings = Settings(
    max_iter=100,
    tol=1e-6,
    qp_solver=:osqp,
    use_soc=true
)

# 3. Solver ausführen
stats = sqp_method(nlp, settings)

# 4. Ergebnisse auswerten
println("Lösung: ", stats.x)
println("Status: ", stats.sqp_status)
```

## QP Solver Interfaces

```
┌─────────────────────┐
│ AbstractQPSolver    │
├─────────────────────┤
│ + solve()           │
└─────────────────────┘
          │
    ┌─────┼─────┬─────────────┬─────────────┐
    │     │     │             │             │
┌─────┐ ┌────┐ ┌──────────┐ ┌──────────┐ ┌────────┐
│OSQP │ │Clar│ │  Ipopt   │ │ MadNLP   │ │Custom  │
│     │ │abel│ │          │ │          │ │        │
└─────┘ └────┘ └──────────┘ └──────────┘ └────────┘
```

## Konvexifizierungsstrategien

```
┌─────────────────────────┐
│ convexify_hessian!()    │
├─────────────────────────┤
│ + :lm                   │
│ + :project              │
│ + :mirror               │
│ + :gershgorin          │
│ + :none                 │
└─────────────────────────┘
```
