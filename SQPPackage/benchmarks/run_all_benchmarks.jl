
#!/usr/bin/env julia

"""
Master-Script zum Ausführen aller Benchmarks
Führt systematisch alle Benchmark-Studien durch
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using SQPPackage

println("="^80)
println("VOLLSTÄNDIGE BENCHMARK-SUITE FÜR SQPPACKAGE.JL")
println("="^80)
println("Basierend auf Performance Profiles nach Dolan & Moré (2002)")
println("")
println("Diese Suite führt folgende Benchmarks durch:")
println("  1. QP-Solver Vergleich")
println("  2. Hessian-Konvexifizierung Vergleich")
println("  3. Globalisierungs-Strategien Vergleich")
println("  4. Toleranz-Einstellungen Vergleich")
println("  5. Großer Benchmark vs. etablierte Solver")
println("="^80)

# Erstelle Results-Verzeichnis
results_dir = joinpath(@__DIR__, "..", "results")
if !isdir(results_dir)
    mkpath(results_dir)
    println("Results-Verzeichnis erstellt: $results_dir")
end

# Liste aller Benchmark-Scripts
benchmark_scripts = [
    ("QP-Solver Vergleich", "compare_qp_solvers.jl"),
    ("Hessian-Konvexifizierung", "compare_hessian_convexification.jl"),
    ("Globalisierungs-Strategien", "compare_globalization.jl"),
    ("Toleranz-Einstellungen", "compare_option_of_choice.jl"),
    ("Großer Benchmark", "large_benchmark.jl")
]

total_start_time = time()

for (i, (name, script)) in enumerate(benchmark_scripts)
    println("\n" * "="^60)
    println("BENCHMARK $i/$(length(benchmark_scripts)): $name")
    println("="^60)
    println("Ausführung: $script")
    
    script_start_time = time()
    
    try
        # Führe Benchmark-Script aus
        include(script)
        
        script_elapsed = time() - script_start_time
        println("\n✓ $name abgeschlossen in $(round(script_elapsed, digits=1)) Sekunden")
        
    catch e
        script_elapsed = time() - script_start_time
        println("\n✗ FEHLER in $name nach $(round(script_elapsed, digits=1)) Sekunden:")
        println("  $e")
        println("  Fahre mit nächstem Benchmark fort...")
    end
end

total_elapsed = time() - total_start_time

println("\n" * "="^80)
println("BENCHMARK-SUITE ABGESCHLOSSEN")
println("="^80)
println("Gesamtzeit: $(round(total_elapsed/60, digits=1)) Minuten")
println("")
println("Generierte Ergebnisse in: $results_dir")
println("")

# Zeige alle generierten PDF-Dateien
pdf_files = filter(f -> endswith(f, ".pdf"), readdir(results_dir))
if !isempty(pdf_files)
    println("Generierte Performance Profiles:")
    for pdf in sort(pdf_files)
        println("  📊 $pdf")
    end
else
    println("⚠️  Keine PDF-Dateien gefunden!")
end

println("")
println("Benchmark-Suite Details:")
println("  ✓ QP-Solver: OSQP, Clarabel, Ipopt, MadNLP")
println("  ✓ Hessian-Regularisierung: none, LM, project, mirror, gershgorin")
println("  ✓ Globalisierung: mit/ohne Liniensuche, mit/ohne SOC")
println("  ✓ Toleranzen: 1e-8, 1e-6, 1e-4, 1e-3")
println("  ✓ Vergleich mit etablierten Solvern")
println("  ✓ Testprobleme und OptimizationProblems.jl")
println("  ✓ Metriken: Gesamtzeit und Funktionsauswertungen")
println("="^80)
