# Bibliotecas e Conjunto de Dados
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_census = pd.read_csv('dataframes\census.csv', encoding='ISO-8859-1')

# Divisão das variáveis
X_census = df_census.iloc[:, 0:14].values
y_census = df_census.iloc[:, 14].values

# Tratamento de Dados
categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13]
label_encoders = {}

for column in categorical_columns:
    label_encoder = LabelEncoder()
    label_encoder.fit(X_census[:, column])
    X_census[:, column] = label_encoder.fit_transform(X_census[:, column])
    label_encoders[column] = label_encoder

one_hot_encoder_census = ColumnTransformer(transformers=[(
    'OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')

X_census = one_hot_encoder_census.fit_transform(X_census).toarray()

# Escalonamento
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

# Divisão de Teste e Treinamento
X_census_training_set, X_census_test_set, y_census_training_set, y_census_test_set = train_test_split(
    X_census, y_census, test_size=0.15, random_state=0)

# Salvando no formato pkl
with open('dataframes\pickle\census.pkl', mode='wb') as f:
    pickle.dump([X_census_training_set, y_census_training_set,
                X_census_test_set, y_census_test_set], f)