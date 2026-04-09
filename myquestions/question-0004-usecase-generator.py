import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

def clasificar_congestion(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5
) -> dict:
    """
    Función de referencia para clasificación con clases desbalanceadas.
    """
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    clases = np.unique(y)
    pesos = compute_class_weight(class_weight='balanced', classes=clases, y=y)
    peso_map = dict(zip(clases.astype(int), pesos))

    skf = StratifiedKFold(n_splits=n_splits)
    precisiones, recalls, f1s, aucs = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Pipeline: Imputación y Escalado
        imputer = SimpleImputer(strategy='mean')
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # Mapeo de pesos por muestra
        pesos_muestra = np.where(y_tr == 1, peso_map[1], peso_map[0])

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
        clf.fit(X_tr, y_tr, sample_weight=pesos_muestra)

        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]

        precisiones.append(precision_score(y_te, y_pred, zero_division=0))
        recalls.append(recall_score(y_te, y_pred, zero_division=0))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_te, y_proba))

    return {
        "precision_media": float(np.mean(precisiones)),
        "recall_medio":    float(np.mean(recalls)),
        "f1_medio":        float(np.mean(f1s)),
        "roc_auc_medio":   float(np.mean(aucs)),
        "pesos_clase":     {int(k): float(v) for k, v in peso_map.items()}
    }

# CAMBIO DE NOMBRE: de generar_caso_de_uso_preparar_datos -> generar_caso_de_uso_clasificar_congestion
def generar_caso_de_uso_clasificar_congestion():
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    n_samples      = int(rng.integers(400, 1001))
    n_features     = int(rng.integers(4, 9))
    n_informative  = int(rng.integers(2, n_features))
    imbalance      = float(rng.uniform(0.06, 0.15)) 
    noise          = float(rng.uniform(0.0, 0.05))
    n_splits_val   = int(rng.choice([3, 4, 5]))
    random_state_val = int(rng.integers(0, 999))

    feature_names = [f"feature_{i}" for i in range(n_features)]
    target_col = "congestion"

    from sklearn.datasets import make_classification
    X_raw, y_raw = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 1),
        n_clusters_per_class=1,
        weights=[1 - imbalance, imbalance],
        flip_y=noise,
        random_state=random_state_val
    )

    df = pd.DataFrame(X_raw, columns=feature_names)

    # Inyectar nulos
    n_null_cols = int(rng.integers(1, min(4, n_features)))
    null_cols = rng.choice(feature_names, size=n_null_cols, replace=False)
    for col in null_cols:
        n_nulls = int(rng.integers(5, max(6, n_samples // 10)))
        null_idx = rng.choice(n_samples, size=n_nulls, replace=False)
        df.loc[null_idx, col] = np.nan

    df[target_col] = y_raw

    # ── INPUT ──────────────────────────────────────────────────────────────────
    input_args = {
        "df": df.copy(),
        "target_col": target_col,
        "n_splits": n_splits_val
    }

    # ── OUTPUT esperado ────────────────────────────────────────────────────────
    output_caso = clasificar_congestion(df.copy(), target_col, n_splits=n_splits_val)

    # Formato de retorno para el evaluador
    return {
        "input": [df.copy(), target_col, n_splits_val],
        "expected": output_caso
    }

# ── Demo local ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    resultado = generar_caso_de_uso_clasificar_congestion()
    print(f"Pesos de clase calculados: {resultado['expected']['pesos_clase']}")
    print(f"F1 Medio: {resultado['expected']['f1_medio']:.4f}")
    print(f"ROC-AUC Medio: {resultado['expected']['roc_auc_medio']:.4f}")
