import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def pipeline_pca_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    alpha:   float = 1.0
) -> dict:

    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc  = scaler.transform(X_test_imp)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)

    model = Ridge(alpha=alpha)
    model.fit(X_train_pca, y_train)
    predicciones = model.predict(X_test_pca)

    rmse = float(np.sqrt(mean_squared_error(y_test, predicciones)))
    r2   = float(r2_score(y_test, predicciones))

    return {
        "n_componentes": int(pca.n_components_),
        "rmse":          rmse,
        "r2":            r2,
        "predicciones":  predicciones
    }


def generar_caso_de_uso_preparar_datos():
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    # Parámetros aleatorios del dataset
    n_total      = int(rng.integers(200, 601))
    n_features   = int(rng.integers(4, 11))        # entre 4 y 10 features
    n_informative = int(rng.integers(2, n_features))
    noise        = float(rng.uniform(5.0, 50.0))
    alpha        = float(rng.choice([0.1, 0.5, 1.0, 2.0, 5.0, 10.0]))
    random_state = int(rng.integers(0, 999))
    test_ratio   = float(rng.uniform(0.15, 0.35))

    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=n_total,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )

    # Split manual sin sklearn para no añadir dependencias extra
    n_test  = max(20, int(n_total * test_ratio))
    n_train = n_total - n_test
    idx     = rng.permutation(n_total)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    y_train, y_test = y[train_idx].copy(), y[test_idx].copy()

    # Inyectar nulos en X_train en 1 o 2 columnas aleatorias
    n_null_cols = int(rng.integers(1, 3))
    null_cols   = rng.choice(n_features, size=n_null_cols, replace=False)
    for col in null_cols:
        n_nulls  = int(rng.integers(5, max(6, n_train // 8)))
        null_idx = rng.choice(n_train, size=n_nulls, replace=False)
        X_train[null_idx, col] = np.nan

    # Ocasionalmente inyectar nulos en X_test también
    if rng.random() > 0.5:
        col_test  = int(rng.integers(0, n_features))
        null_idx2 = rng.choice(n_test, size=int(rng.integers(1, max(2, n_test // 10))), replace=False)
        X_test[null_idx2, col_test] = np.nan

    # ── INPUT ──────────────────────────────────────────────────────────────────
    input_caso = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test":  X_test,
        "y_test":  y_test,
        "alpha":   alpha
    }

    # ── OUTPUT esperado ────────────────────────────────────────────────────────
    output_caso = pipeline_pca_ridge(
        X_train.copy(), y_train.copy(),
        X_test.copy(),  y_test.copy(),
        alpha=alpha
    )

    return input_caso, output_caso


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for i in range(3):
        inp, out = generar_caso_de_uso_preparar_datos()
        print(f"\n{'='*60}")
        print(f"Caso {i+1}")
        print(f"  X_train shape  : {inp['X_train'].shape}")
        print(f"  X_test  shape  : {inp['X_test'].shape}")
        print(f"  nulos X_train  : {np.isnan(inp['X_train']).sum()}")
        print(f"  nulos X_test   : {np.isnan(inp['X_test']).sum()}")
        print(f"  alpha          : {inp['alpha']}")
        print(f"  n_componentes  : {out['n_componentes']}")
        print(f"  rmse           : {out['rmse']:.4f}")
        print(f"  r2             : {out['r2']:.4f}")
        print(f"  predicciones[:5]: {out['predicciones'][:5].round(3)}")
