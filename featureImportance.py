
import streamlit as st
import dataframe
import pandas as pd
import categoricos_numericos
import numpy as np
from sklearn import model_selection, dummy, metrics, utils, linear_model, ensemble, feature_selection

class feature:
    df = dataframe.Dados.dataframe
    caracteristicas = df.drop('HeartDisease', axis=1)
    previsor = df['HeartDisease']

    f_regression = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=25)
    fit = f_regression.fit(caracteristicas, previsor)
    features = fit.transform(caracteristicas)
    cols = fit.get_support(indices=True)
    fregression_Dataframe = df.iloc[:,cols]

    chi2 = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=25)
    fit = chi2.fit(caracteristicas, previsor)
    features = fit.transform(caracteristicas)
    cols = fit.get_support(indices=True)
    chi2_Dataframe = df.iloc[:,cols]

    model = linear_model.LogisticRegression(max_iter=30000)
    rfe = feature_selection.RFE(model, n_features_to_select=25)
    fit = rfe.fit(caracteristicas, previsor)
    cols = fit.get_support(indices=True)
    linearModel_Dataframe = df.iloc[:,cols]


def featureImportance():
    st.markdown('# Análise de Feature Importance dos dados')
    st.markdown('### Feature Importance resumidamente trata da análise de quais relacionamentos entre colunas influenciam no resultado.')
    st.markdown('### Através da Feature Importance prentendemos encontrar o conjunto de atributos que nos tragam os resultados mais precisos.')
    st.markdown('### Iremos analisar 3 algoritmos através de funções da biblioteca sklearn do python.')
    st.markdown('- f_regression')


    #DADOS
    df = categoricos_numericos.returnDataFrame()
    caracteristicas = df.drop('HeartDisease', axis=1)
    previsor = df['HeartDisease']

    #F_REGRESSION
    st.markdown('# O "f_regression" encontra os melhores features através de regressão linear.')
    f_regression = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=10)
    fit = f_regression.fit(caracteristicas, previsor)
    features = fit.transform(caracteristicas)
    cols = fit.get_support(indices=True)
    st.dataframe(df.iloc[:,cols])

    #CHI2
    st.markdown('# O "CHI2" encontra os melhores features através de regressão linear.')
    chi2 = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=10)
    fit = chi2.fit(caracteristicas, previsor)
    features = fit.transform(caracteristicas)
    cols = fit.get_support(indices=True)
    st.dataframe(df.iloc[:,cols])

    #RFE
    st.markdown('# O "RFE" ou Recursive Feature Elimination trabalha encontrando e removendo features.')
    model = linear_model.LogisticRegression(max_iter=30000)
    rfe = feature_selection.RFE(model, n_features_to_select=10)
    fit = rfe.fit(caracteristicas, previsor)
    cols = fit.get_support(indices=True)
    st.dataframe(df.iloc[:,cols])