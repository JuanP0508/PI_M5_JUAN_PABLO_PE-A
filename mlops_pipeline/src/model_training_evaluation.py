import sys
import os
import joblib
sys.path.insert(0, os.path.dirname(__file__))

from cargar_datos import CargarDatos
from ft_engineering import build_model, get_splits, summarize_classification

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

df = CargarDatos()
X_train, X_test, y_train, y_test = get_splits()

modelos = {
    "Decision Tree":     build_model(DecisionTreeClassifier(random_state=42, class_weight='balanced')),
    "Random Forest":     build_model(RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
    "Logistic Regression": build_model(LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
}

mejor_nombre = None
mejor_f1 = 0
mejor_modelo = None

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    summarize_classification(y_test, y_pred, nombre)
    f1 = f1_score(y_test, y_pred)
    if f1 > mejor_f1:
        mejor_f1 = f1
        mejor_nombre = nombre
        mejor_modelo = modelo

# Guardar el mejor modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(BASE_DIR, "mejor_modelo.pkl")
joblib.dump(mejor_modelo, ruta_modelo)
print(f"\nMejor modelo: {mejor_nombre} (F1: {mejor_f1:.4f})")
print(f"Modelo guardado en: {ruta_modelo}")