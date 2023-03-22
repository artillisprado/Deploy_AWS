from random import Random
import matplotlib
import matplotlib.pyplot as plt
import dataframe
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import confusion_matrix

def random_forest_treinada():
    df = dataframe.Dados.dataframe
    x  = df.drop('HeartDisease', axis=1)#'AgeCategory',
    y = df['HeartDisease']
    # st.write(X)
    #Treinando o modelo
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)
    logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('RF',RandomForestClassifier(random_state=42))])
    logreg_pipeline.fit(x, y)
    # predictionsLR = logreg_pipeline.predict(x_test)
    return logreg_pipeline

def calculate_score_logistic_regression( x, y, x_test, y_test):
    logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('LR',LogisticRegression(random_state=42))])
    logreg_pipeline.fit(x, y)
    predictionsLR = logreg_pipeline.predict(x_test)
    lgrmc = confusion_matrix(y_test, predictionsLR)
    logreg_cv = cross_val_score(logreg_pipeline, x, y, cv=10, scoring='f1')
    if matrix:
        matrix(lgrmc, 'Regressão de Logística', logreg_cv)

def calculate_score_random_forest( x, y, x_test, y_test):
    rf_pipeline = Pipeline(steps = [('scale',StandardScaler()),('RF',RandomForestClassifier(random_state=42))])
    rf_pipeline.fit(x, y)
    predictionsRF = rf_pipeline.predict(x_test)
    rfcm = confusion_matrix(y_test, predictionsRF)
    rf_cv = cross_val_score(rf_pipeline, x, y, cv=10, scoring='f1')
    if matrix:
        matrix(rfcm, 'Random Forest', rf_cv)

def matrix(classifier, classifierName, score):
    score = score.mean()
    st.subheader('Matriz de Confusão ' + classifierName + ':')
    
    fig = px.imshow(classifier, text_auto=True, aspect="auto", color_continuous_scale='greens',
                labels=dict(x="Valores previstos ", y="Valores reais", color="Número de casos"),
                x=['Predição negativa', 'Predição positiva'],
                y=['Negativo', 'Positivo']
            )

    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig)

def rf_feat_importance( m, df):
        return pd.DataFrame({'Feature' : df.columns, 'Importance' : m.feature_importances_}).sort_values('Importance', ascending=False)

def feature_importance(df, y):
    #classificador
    colors = ["lightgray","lightgray","#0f4c81"]
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    
    df_feature = df[['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth','DiffWalking', 'Sex', 'Diabetic', 'PhysicalActivity', 'SleepTime','Asthma', 'KidneyDisease','SkinCancer']]

    rf_pipeline = Pipeline(steps = [('scale',StandardScaler()),('RF',RandomForestClassifier(random_state=42))])
    rf_pipeline.fit(df_feature, y)
    #Obtendo a Feature importance do Random Forest
    fi_random_florest = rf_feat_importance(rf_pipeline['RF'], df_feature)

    fig = px.bar(fi_random_florest, y='Importance', x='Feature', text_auto='.2s',
                title="Feature Importance", labels={ 'Importance': 'Importância' })
    fig.update_yaxes(showline=True, showgrid=False)
    fig.update_xaxes(showline=True, showgrid=False)
    st.plotly_chart(fig)

def mineracao():

    df = dataframe.Dados.dataframe

    x = df.drop('HeartDisease', axis=1)#'AgeCategory',
    y = df['HeartDisease']
    # st.write(X)
    #Treinando o modelo
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)

    #Balanceando os dados com Smote 
    oversample = SMOTE()
    X_train_resh, y_train_resh = oversample.fit_resample(X_train, y_train.ravel())


    feature_importance(df, y)
    calculate_score_random_forest(X_train, y_train, X_test, y_test)
    calculate_score_logistic_regression(X_train, y_train, X_test, y_test)

    st.markdown('___')
    st.markdown('# Conclusão')
    st.markdown('### Trabalhando com a predição de doenças a quantidade de falsos positivos em comparação a quantidade de falsos negativos é algo muito importante, porque é mais seguro o sujeito suspeitar que possua a doença e ir fazer os exames do que ele receber o diagnóstico negativo, realmente ter e não ir ao médico confirmar isso.')
    st.markdown('### Considerando que a diferença de falsos positivos entre os dois algoritmos não é tão significativa, porém, há uma grande diferença na quantidade de positivos verdadeiros, o algoritmo que foi visto como mais seguro é o Random Florest.')
