import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def segmentar_rutas(X: np.ndarray, random_state: int = 42) -> dict:
    """
    Función de referencia para generar el resultado esperado.
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

# CAMBIO DE NOMBRE AQUÍ: de generar_caso_de_uso_preparar_datos -> generar_caso_de_uso_segmentar_rutas
def generar_caso_de_uso_segmentar_rutas():
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    # Parámetros aleatorios para diversidad de tests
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

    # Añadir ruido leve
    noise = rng.normal(0, rng.uniform(0.1, 0.5), size=X.shape)
    X = X + noise

    # ── INPUT ──────────────────────────────────────────────────────────────────
    input_caso = {
        "X": X.copy(),
        "random_state": random_state_val
    }

    # ── OUTPUT esperado ────────────────────────────────────────────────────────
    output_caso = segmentar_rutas(X.copy(), random_state=random_state_val)

    # Retorno en formato compatible con evaluadores (diccionario con input y expected)
    return {
        "input": [X.copy(), random_state_val],
        "expected": output_caso
    }

# ── Demo local ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    resultado = generar_caso_de_uso_segmentar_rutas()
    print(f"Mejor K encontrado: {resultado['expected']['mejor_k']}")
    print(f"Mejor Score: {resultado['expected']['mejor_score']:.4f}")
    print("\nResumen Estadístico (Primeras filas):")
    print(resultado['expected']['resumen'].head())
