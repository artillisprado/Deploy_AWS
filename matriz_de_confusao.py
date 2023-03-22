import dataframe
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = dataframe.Dados.dataframe
matriz_confusao = confusion_matrix
def matrizDeConfusao():
    
    X  = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    #Treinando o modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
    matriz = matriz_confusao(y, y_test)