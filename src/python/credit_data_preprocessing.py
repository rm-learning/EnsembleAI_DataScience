# Bibliotecas e Conjunto de Dados
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_credit = pd.read_csv('dataframes\credit_data.csv', encoding='ISO-8859-1')

# Tratamento de Dados
df_credit.mean()
df_credit['age'].mean()
df_credit['age'][df_credit['age'] > 0].mean()

df_credit.loc[df_credit['age'] < 0, 'age'] = 40.92
df_credit.loc[df_credit['age'] < 0]  # Visualizar mudanças

df_credit.isnull().sum()
df_credit.loc[pd.isnull(df_credit['age'])]
df_credit['age'].fillna(df_credit['age'].mean(), inplace=True)
df_credit.loc[pd.isnull(df_credit['age'])]  # Visualizar mudanças

# Divisão das variáveis
X_credit = df_credit.iloc[:, 1:4].values
y_credit = df_credit.iloc[:, 4].values

# Escalonamento
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

# Teste e Treinamento
X_credit_training_set, X_credit_test_set, y_credit_training_set, y_credit_test_set = \
    train_test_split(X_credit, y_credit, test_size=0.25, random_state=0)

# Salvando no formato pkl
with open('dataframes\pickle\credit_data.pkl', mode='wb') as f:
    pickle.dump([X_credit_training_set, y_credit_training_set,
                X_credit_test_set, y_credit_test_set], f)