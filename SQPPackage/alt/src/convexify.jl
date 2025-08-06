module ConvexifyModule

# Importiere notwendige Module und Exportiere 
using LinearAlgebra   # Für lineare Algebra Operationen wie Eigenwertzerlegung
using SparseArrays    # Für die Arbeit mit dünn besetzten Matrizen
import ..SettingsModule: Settings # Importiere den Settings-Typ aus einem übergeordneten Modul

export convexify_hessian!, convexify_hessian

function convexify_hessian!(H::Union{Matrix{Float64}, LinearAlgebra.Symmetric{Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}}}, settings::Settings)
    # Konvertiere Sparse-Matrix in Dense-Matrix, falls notwendig
    H_dense = isa(H, SparseArrays.SparseMatrixCSC) || isa(H, LinearAlgebra.Symmetric{Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}}) ? Matrix(H) : H
    n = size(H_dense, 1) # Ermittle die Dimension der Matrix
    ϵ = settings.epsilon # Numerische Toleranz aus den Einstellungen

    # Prüfe auf NaN oder Inf Werte in der Matrix
    if any(x -> !isfinite(x), H_dense)
        println("WARNUNG: Hessian enthält NaN oder Inf - wird durch Identitätsmatrix ersetzt")
        H_dense = Matrix(I, n, n) * ϵ # Ersetze durch skalierte Einheitsmatrix
        return H_dense
    end

    strategy = settings.hessian_convexification # Hole die gewählte Konvexifizierungsstrategie

    if strategy == :none
        return H_dense 
    elseif strategy == :lm
        # Levenberg-Marquardt-Ansatz: Addiere Wert zur Diagonale, falls kleinster Eigenwert negativ
        Lambda_min = minimum(eigvals(LinearAlgebra.Symmetric(H_dense))) # Berechne den kleinsten Eigenwert
        if Lambda_min < 0
            add_val = abs(Lambda_min) + ϵ # Wert, der addiert werden muss
            for i in 1:n; H_dense[i, i] += add_val; end # Addiere den Wert zur Diagonale
        end
    elseif strategy == :project
        # Projektion: Negative Eigenwerte werden auf epsilon projiziert
        E = eigen(LinearAlgebra.Symmetric(H_dense)) # Führe Eigenwertzerlegung durch
        Lambda_proj = max.(E.values, ϵ) # Ersetze negative Eigenwerte durch epsilon
        H_dense .= E.vectors * LinearAlgebra.Diagonal(Lambda_proj) * E.vectors' # Rekonstruiere die Matrix
    elseif strategy == :mirror
        # Spiegelung: Negative Eigenwerte werden gespiegelt und auf epsilon begrenzt
        E = eigen(LinearAlgebra.Symmetric(H_dense)) # Führe Eigenwertzerlegung durch
        Lambda_mirror = max.(abs.(E.values), ϵ) # Nimm den Betrag der Eigenwerte, min. epsilon
        H_dense .= E.vectors * LinearAlgebra.Diagonal(Lambda_mirror) * E.vectors' # Rekonstruiere die Matrix
    elseif strategy == :gershgorin
        # Gershgorin-Kreise: Diagonale wird angepasst, falls die Gershgorin-Untergrenze negativ ist
        Lambda_min = Inf # Initialisiere den minimalen Gershgorin-Wert
        for i in 1:n
            r_i = sum(abs, H_dense[i, :]) - abs(H_dense[i, i]) # Berechne den Radius des Gershgorin-Kreises
            Lambda_i = H_dense[i, i] - r_i # Berechne die untere Grenze des Gershgorin-Kreises
            Lambda_min = min(Lambda_min, Lambda_i) # Finde die kleinste untere Grenze
        end
        if Lambda_min < 0
            add_val = abs(Lambda_min) + ϵ # Wert, der addiert werden muss
            for i in 1:n; H_dense[i, i] += add_val; end # Addiere den Wert zur Diagonale
        end
    else
        error("Unbekannte Strategie: $strategy") # Fehlermeldung bei unbekannter Strategie
    end

    return H_dense # Gib die modifizierte Matrix zurück
end


function convexify_hessian(H::Union{Matrix{Float64}, LinearAlgebra.Symmetric{Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}}}, settings::Settings)
    H_copy = Matrix(H) # Erstelle eine dichte Kopie der Matrix
    return convexify_hessian!(H_copy, settings) # Rufe die in-place Funktion auf der Kopie auf
end

end # Ende des Moduls