push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# Dein Modul laden
using SQPPackage
using SQPPackage.SettingsModule
using SQPPackage.SQPMethod

using ADNLPModels
using LinearAlgebra

println("="^60)
println("Eigener SQP-Löser - Beispiele")
println("Modul erfolgreich geladen!")
println("="^60)
# Beispiel 1: Einfaches quadratisches Problem mit Ihren Settings
println("\n--- Beispiel 1: Einfaches Quadratisches Problem ---")
println("Problem: min x₁² + 2x₂²")

function f1(x)
    return x[1]^2 + 2*x[2]^2
end

x0 = [1.0, 1.0]
nlp1 = ADNLPModel(f1, x0)

# Ihre Settings-Struktur verwenden
settings1 = Settings(
    verbose=true, 
    max_iter=50, 
    tol=1e-6,
    qp_solver=:osqp,
    use_soc=true,
    hessian_convexification=:lm,
    use_globalization=true,
    regularization=1e-8
)

println("Verwendete Einstellungen:")
println("  - Max. Iterationen: ", settings1.max_iter)
println("  - Toleranz: ", settings1.tol)
println("  - QP-Löser: ", settings1.qp_solver)
println("  - SOC aktiviert: ", settings1.use_soc)
println("  - Hessian-Konvexifizierung: ", settings1.hessian_convexification)

# Mit Ihrer eigenen SQP-Methode lösen
try
    stats1 = sqp_method(nlp1, settings1)
    
    println("Lösung: x = ", round.(stats1.x, digits=6))
    println("Zielfunktionswert: f(x) = ", round(stats1.obj_val, digits=6))
    println("Status: ", stats1.sqp_status)
    println("Iterationen: ", stats1.iter)
catch e
    println("Fehler beim Lösen von Problem 1: ", e)
end

# Beispiel 2: Beschränktes Optimierungsproblem
println("\n--- Beispiel 2: Beschränktes Optimierungsproblem ---")
println("Problem: min x₁² + 2x₂²")
println("Unter den Nebenbedingungen: x₁ + 2x₂² = 10")
println("                           2x₁ + x₂ = 9")

function f2(x)
    return x[1]^2 + 2*x[2]^2
end

function c2(x)
    return [x[1] + 2*x[2]^2 - 10.0, 2*x[1] + x[2] - 9.0]
end

x0 = [1.2, 5.4]
nlp2 = ADNLPModel(f2, x0; c=c2, lcon=[0.0, 0.0], ucon=[0.0, 0.0])

# Erweiterte Einstellungen mit Filter-Parametern
settings2 = Settings(
    verbose=true,
    max_iter=100,
    tol=1e-6,
    qp_solver=:osqp,
    use_soc=true,
    soc_improvement_threshold=0.5,
    use_globalization=true,
    hessian_convexification=:lm,
    # Filter-Parameter
    γh=1e-5,
    γf=1e-5,
    γα=0.05,
    sh=1.1,
    sf=2.3,
    ηf=1e-4
)

try
    stats2 = sqp_method(nlp2, settings2)
    
    println("Lösung: x = ", round.(stats2.x, digits=6))
    println("Zielfunktionswert: f(x) = ", round(stats2.obj_val, digits=6))
    println("Status: ", stats2.sqp_status)
    
    # Nebenbedingungen überprüfen
    c_val = c2(stats2.x)
    println("Nebenbedingungsverletzungen: ", round.(c_val, digits=8))
    println("KKT-Residuen:")
    println("  Primale Zulässigkeit: ", stats2.inf_pr)
    println("  Duale Zulässigkeit: ", stats2.inf_du)
    println("  Komplementarität: ", stats2.inf_comp)
catch e
    println("Fehler beim Lösen von Problem 2: ", e)
end

# Beispiel 3: Verschiedene Hessian-Konvexifizierungsmethoden testen
println("\n--- Beispiel 3: Hessian-Konvexifizierungsmethoden ---")

# Rosenbrock-Funktion (bekannt für schlecht konditionierte Hessian)
function f3(x)
    return (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
end

x0 = [0.0, 0.0]
nlp3 = ADNLPModel(f3, x0)

konvexifizierungsmethoden = [:none, :lm, :project, :mirror, :gershgorin]

for methode in konvexifizierungsmethoden
    println("\nTeste Hessian-Konvexifizierung: ", methode)
    
    settings3 = Settings(
        verbose=false,
        max_iter=100,
        tol=1e-6,
        qp_solver=:osqp,
        hessian_convexification=methode,
        regularization=1e-6,
        use_regularization=true
    )
    
    try
        stats3 = sqp_method(nlp3, settings3)
        
        println("  Status: ", stats3.sqp_status)
        println("  Lösung: ", round.(stats3.x, digits=4))
        println("  Zielfunktion: ", round(stats3.obj_val, digits=6))
        println("  Iterationen: ", stats3.iter)
        println("  Zeit: ", round(stats3.total_time, digits=4), " s")
    catch e
        println("  Fehler: ", e)
    end
end

# Beispiel 4: OSQP-Einstellungen anpassen
println("\n--- Beispiel 4: OSQP-Einstellungen anpassen ---")

# Verschiedene OSQP-Konfigurationen testen
osqp_configs = [
    ("Standard", Dict("verbose" => false, "max_iter" => 1000, "eps_abs" => 1e-6)),
    ("Hohe Präzision", Dict("verbose" => false, "max_iter" => 5000, "eps_abs" => 1e-9, "eps_rel" => 1e-9)),
    ("Schnell", Dict("verbose" => false, "max_iter" => 500, "eps_abs" => 1e-4, "eps_rel" => 1e-4))
]

for (name, osqp_dict) in osqp_configs
    println("\nOSQP-Konfiguration: ", name)
    
    settings4 = Settings(
        verbose=false,
        max_iter=50,
        tol=1e-6,
        qp_solver=:osqp,
        osqp_settings=osqp_dict,
        use_globalization=true
    )
    
    try
        stats4 = sqp_method(nlp2, settings4)
        
        println("  Status: ", stats4.sqp_status)
        println("  Iterationen: ", stats4.iter)
        println("  QP-Status: ", stats4.qp_solve_status)
        println("  Zeit: ", round(stats4.total_time, digits=4), " s")
    catch e
        println("  Fehler: ", e)
    end
end

# Beispiel 5: Benchmark-Modus (falls implementiert)
println("\n--- Beispiel 5: Benchmark-Modus ---")

settings_benchmark = Settings(
    verbose=false,
    max_iter=100,
    tol=1e-6,
    benchmark=true,
    benchmark_file="mein_benchmark.csv"
)

println("Benchmark-Einstellungen:")
println("  - Benchmark aktiviert: ", settings_benchmark.benchmark)
println("  - Benchmark-Datei: ", settings_benchmark.benchmark_file)

try
    stats_bench = sqp_method(nlp1, settings_benchmark)
    println("Benchmark-Lauf abgeschlossen!")
    println("Status: ", stats_bench.sqp_status)
    
    if isfile(settings_benchmark.benchmark_file)
        println("Benchmark-Datei erstellt: ", settings_benchmark.benchmark_file)
    end
catch e
    println("Benchmark-Fehler: ", e)
end

println("\n" * "="^60)
println("Eigener SQP-Löser Beispiele abgeschlossen!")
println("Getestete Funktionen:")
println("  ✓ Grundlegende SQP-Optimierung")
println("  ✓ Beschränkte Probleme mit Filter-Liniensuche")
println("  ✓ Verschiedene Hessian-Konvexifizierungsmethoden")
println("  ✓ OSQP-Einstellungen Anpassung")
println("  ✓ Benchmark-Modus")
println("  ✓ Second Order Correction (SOC)")
println("="^60)