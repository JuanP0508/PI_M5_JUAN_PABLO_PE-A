import pandas as pd
from cargar_datos import CargarDatos
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = CargarDatos()

# ── Features y target ──────────────────────────────────────────────────
X = df.drop(columns=["Pago_atiempo", "puntaje", "fecha_prestamo"])
y = df["Pago_atiempo"]

# ── Identificar tipos ──────────────────────────────────────────────────
num_features = X.select_dtypes("number").columns.tolist()
num_features = [c for c in num_features if c not in ["tipo_credito"]]
cat_features = ["tipo_laboral", "tipo_credito"]
ord_features = ["tendencia_ingresos"]

# Quitar ordinales de categóricas si quedaron ahí
cat_features = [c for c in cat_features if c in X.columns]

print("Numéricas:", num_features)
print("Categóricas:", cat_features)
print("Ordinales:", ord_features)

# ── Pipelines por tipo ─────────────────────────────────────────────────
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=[['Decreciente', 'Estable', 'Creciente']],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

# ── ColumnTransformer ──────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features),
    ('ord', ord_transformer, ord_features)
])

# ── Funciones reutilizables ────────────────────────────────────────────
def build_model(modelo):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', modelo)
    ])

def get_splits(test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)

def summarize_classification(y_test, y_pred, nombre="Modelo"):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(f"\n{'='*50}\n  {nombre}\n{'='*50}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_transformado = preprocessor.fit_transform(X_train)
    print("Preprocessor — shape:", X_transformado.shape)
