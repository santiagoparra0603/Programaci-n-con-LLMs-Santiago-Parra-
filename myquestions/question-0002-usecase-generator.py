import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def segmentar_rutas(X: np.ndarray, random_state: int = 42) -> dict:
    """
    Función de referencia que debe resolver el problema de las rutas.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mejor_k = None
    mejor_score = -np.inf
    mejor_etiquetas = None

    # Probar k de 2 a 8 inclusive
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        etiquetas = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, etiquetas)
        
        # En caso de empate, se queda con el primero (el k más pequeño)
        if score > mejor_score:
            mejor_score = score
            mejor_k = k
            mejor_etiquetas = etiquetas

    # Construir resumen estadístico con el X original
    df_resumen = pd.DataFrame(X)
    df_resumen.columns = [f"feature_{i}" for i in range(X.shape[1])]
    df_resumen["cluster"] = mejor_etiquetas
    resumen = df_resumen.groupby("cluster").mean()
    resumen.index.name = None

    return {
        "mejor_k": mejor_k,
        "mejor_score": float(mejor_score),
        "etiquetas": mejor_etiquetas,
        "resumen": resumen
    }

def generar_caso_de_uso_segmentar_rutas():
    """
    Generador que devuelve (input_dict, expected_output) para el evaluador.
    """
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    # Parámetros aleatorios
    n_samples   = int(rng.integers(150, 501))
    n_features  = int(rng.integers(3, 7))
    n_centers   = int(rng.integers(2, 6))
    cluster_std = float(rng.uniform(0.5, 2.5))
    random_state_val = int(rng.integers(0, 999))

    from sklearn.datasets import make_blobs
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state_val
    )

    # Añadir ruido
    noise = rng.normal(0, rng.uniform(0.1, 0.5), size=X.shape)
    X = X + noise

    # --- ESTRUCTURA CORRECTA DE RETORNO ---
    
    # 1. El primer elemento DEBE ser un diccionario con los argumentos de la función
    input_args = {
        "X": X.copy(),
        "random_state": random_state_val
    }

    # 2. El segundo elemento es el resultado esperado
    output_esperado = segmentar_rutas(X.copy(), random_state=random_state_val)

    # Se retornan por separado para que el evaluador pueda hacer: inp, exp = func()
    return input_args, output_esperado

# ── Demo local ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        inp, out = generar_caso_de_uso_segmentar_rutas()
        print("✅ Formato de retorno correcto para el Archivo 0002")
        print(f"Tipo de input: {type(inp)}")
        print(f"Mejor K en este caso: {out['mejor_k']}")
    except Exception as e:
        print(f"❌ Error en el formato: {e}")
