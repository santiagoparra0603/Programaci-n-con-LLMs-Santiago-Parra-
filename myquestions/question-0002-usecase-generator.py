import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def segmentar_rutas(X: np.ndarray, random_state: int = 42) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mejor_k = None
    mejor_score = -np.inf
    mejor_etiquetas = None

    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        etiquetas = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, etiquetas)
        if score > mejor_score:
            mejor_score = score
            mejor_k = k
            mejor_etiquetas = etiquetas

    df_resumen = pd.DataFrame(X)
    df_resumen.columns = [f"feature_{i}" for i in range(X.shape[1])]
    df_resumen["cluster"] = mejor_etiquetas
    resumen = df_resumen.groupby("cluster").mean()
    resumen.index.name = None

    return {
        "mejor_k":     mejor_k,
        "mejor_score": float(mejor_score),
        "etiquetas":   mejor_etiquetas,
        "resumen":     resumen
    }


def generar_caso_de_uso_preparar_datos():
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    # Parámetros aleatorios
    n_samples   = int(rng.integers(150, 501))
    n_features  = int(rng.integers(3, 7))
    n_centers   = int(rng.integers(2, 6))
    cluster_std = float(rng.uniform(0.5, 2.5))
    random_state = int(rng.integers(0, 999))

    from sklearn.datasets import make_blobs
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state
    )

    # Añadir ruido gaussiano leve para que no sea perfectamente separable
    noise = rng.normal(0, rng.uniform(0.1, 0.5), size=X.shape)
    X = X + noise

    # ── INPUT ──────────────────────────────────────────────────────────────────
    input_caso = {
        "X":            X.copy(),
        "random_state": random_state
    }

    # ── OUTPUT esperado ────────────────────────────────────────────────────────
    output_caso = segmentar_rutas(X.copy(), random_state=random_state)

    return input_caso, output_caso


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for i in range(3):
        inp, out = generar_caso_de_uso_preparar_datos()
        X_in = inp["X"]
        print(f"\n{'='*60}")
        print(f"Caso {i+1}")
        print(f"  n_samples    : {X_in.shape[0]}")
        print(f"  n_features   : {X_in.shape[1]}")
        print(f"  random_state : {inp['random_state']}")
        print(f"  mejor_k      : {out['mejor_k']}")
        print(f"  mejor_score  : {out['mejor_score']:.4f}")
        print(f"  etiquetas[:8]: {out['etiquetas'][:8]}")
        print(f"  resumen:\n{out['resumen'].round(3)}")
