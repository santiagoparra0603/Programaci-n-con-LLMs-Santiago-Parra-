import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def predecir_desercion(df: pd.DataFrame, target_col: str) -> dict:
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    feature_names = df.drop(columns=[target_col]).columns.tolist()
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    skf = StratifiedKFold(n_splits=5)
    scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='f1')

    clf.fit(X_scaled, y)
    importancias = dict(zip(feature_names, clf.feature_importances_))

    return {
        "f1_medio": float(np.mean(scores)),
        "f1_std": float(np.std(scores)),
        "importancias": importancias
    }


def generar_caso_de_uso_preparar_datos():
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    n_samples = int(rng.integers(200, 601))
    imbalance   = rng.uniform(0.10, 0.35)          # proporción de clase positiva

    # Parámetros aleatorios del dataset sintético
    n_informative = int(rng.integers(2, 5))
    noise_level   = rng.uniform(0.0, 0.3)
    random_state  = int(rng.integers(0, 999))

    from sklearn.datasets import make_classification
    X_raw, y_raw = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=n_informative,
        n_redundant=6 - n_informative - 1,
        n_clusters_per_class=1,
        weights=[1 - imbalance, imbalance],
        flip_y=noise_level,
        random_state=random_state
    )

    feature_names = [
        'videos_vistos', 'quizzes_respondidos', 'dias_activos',
        'foros_participados', 'promedio_nota', 'horas_semanales'
    ]
    df = pd.DataFrame(X_raw, columns=feature_names)

    # Inyectar nulos aleatorios en 1 o 2 columnas
    n_null_cols = int(rng.integers(1, 3))
    null_cols   = rng.choice(feature_names, size=n_null_cols, replace=False)
    for col in null_cols:
        null_idx = rng.choice(df.index, size=int(rng.integers(5, 40)), replace=False)
        df.loc[null_idx, col] = np.nan

    target_col = 'completo_curso'
    df[target_col] = y_raw

    # ── INPUT ──────────────────────────────────────────────────────────────────
    input_caso = {
        "df":         df.copy(),
        "target_col": target_col
    }

    # ── OUTPUT esperado (ejecutando la función real) ────────────────────────────
    output_caso = predecir_desercion(df.copy(), target_col)

    return input_caso, output_caso


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for i in range(3):
        inp, out = generar_caso_de_uso_preparar_datos()
        df_in = inp["df"]
        print(f"\n{'='*55}")
        print(f"Caso {i+1}")
        print(f"  n_samples       : {len(df_in)}")
        print(f"  nulos por col   : {df_in.isnull().sum().to_dict()}")
        print(f"  balance clases  : {df_in['completo_curso'].value_counts().to_dict()}")
        print(f"  f1_medio        : {out['f1_medio']:.4f}")
        print(f"  f1_std          : {out['f1_std']:.4f}")
        print(f"  importancias    :")
        for feat, imp in out['importancias'].items():
            print(f"    {feat:<28} {imp:.4f}")
