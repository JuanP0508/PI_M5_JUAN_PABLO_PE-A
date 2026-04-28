import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from cargar_datos import CargarDatos
from ft_engineering import build_model, get_splits, summarize_classification

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

df = CargarDatos()
X_train, X_test, y_train, y_test = get_splits()

dt = build_model(DecisionTreeClassifier(random_state=42, class_weight='balanced'))
dt.fit(X_train, y_train)
summarize_classification(y_test, dt.predict(X_test), "Decision Tree")

rf = build_model(RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
rf.fit(X_train, y_train)
summarize_classification(y_test, rf.predict(X_test), "Random Forest")

lr = build_model(LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
lr.fit(X_train, y_train)
summarize_classification(y_test, lr.predict(X_test), "Logistic Regression")