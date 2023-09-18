# Bibliotecas e Conjunto de Dados
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

from sklearn.preprocessing import LabelEncoder

df_weather = pd.read_csv('dataframes\weather_play.csv', encoding='ISO-8859-1')

# Divisão das variáveis
X_weather = df_weather.iloc[:, 1:5].values
y_weather = df_weather.iloc[:, 5].values

# Tratamento de dados
categorical_columns = [0, 1, 2, 3]
label_encoders = {}

for column in categorical_columns:
    label_encoder = LabelEncoder()
    label_encoder.fit(X_weather[:, column])
    X_weather[:, column] = label_encoder.fit_transform(X_weather[:, column])
    label_encoders[column] = label_encoder

# Salvando no formato pkl
with open('dataframes\pickle\weather_play.pkl', mode='wb') as f:
    pickle.dump([X_weather, y_weather], f)