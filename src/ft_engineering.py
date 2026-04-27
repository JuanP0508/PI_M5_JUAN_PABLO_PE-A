# lirerias 

import pandas as pd
from cargar_datos import CargarDatos
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

df= CargarDatos()
print(df.head)
print(df.columns)
print(df.describe())


## pasos del pipeline

## identificar features y targets 

X = df.drop("Pago_atiempo", axis=1) #features
y = df["Pago_atiempo"] #target

#paso 2 identificar variables numericas y categoricas
num_features= X.select_dtypes("number").columns # variables numericas
cat_features= X.select_dtypes("object").columns # variables categoricas

print("features_numericas:", num_features)
print("features_numericas:", cat_features)

## paso #3 crear pipelines para cada feature numerica 
num_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean'))

    ]
)

# paso 4 combinar los transformadores con columntransformer
preproccesor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore').cat_features)
    ]
)


