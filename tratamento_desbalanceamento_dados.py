import streamlit as st
import dataframe
import sklearn
import pandas as pd
import numpy as np
from sklearn import model_selection, dummy, metrics, utils, linear_model, ensemble
def desbalanceamento_dados():
    st.title("A Partir dos dados abaixo podemos notar que há um desbalanceamento dos dados")
    df = dataframe.Dados.dataframe_sem_tratamento
    df = dataframe.returnDataFrame(df)
    st.dataframe(df.HeartDisease.value_counts())
    #testando com DummyClassifier
    y = df.HeartDisease
    X = df.drop('HeartDisease', axis=1)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=27)
    # DummyClassifier to predict only target 0
    st.subheader(' - Vamos usar o DummyClassifier para visualizar o desbalanceamento em uma predição "burra"')
    dummy_classifier = dummy.DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    dummy_pred = dummy_classifier.predict(X_test)
    # checking unique labels
    st.subheader('# Unique predicted labels: {}'.format(np.unique(dummy_pred)))
    # checking accuracy
    st.subheader('# Test score: {}'.format(metrics.accuracy_score(y_test, dummy_pred)))
    st.subheader("# Esta alta taxa implica que os dados estão desbalanceados, logo precisamos resolver isso!")
    st.subheader("#Para isto, vamos usar a técnina de Classe minoritária sobreamostra")
    X = pd.concat([X_train, y_train], axis=1)
    sem_doencas_do_coracao = X[X.HeartDisease==0]
    com_doencas_do_coracao = X[X.HeartDisease==1]
    com_doencas_expandido = utils.resample(com_doencas_do_coracao,
                          replace=True, # sample with replacement
                          n_samples=len(sem_doencas_do_coracao), # match number in majority class
                          random_state=27) # reproducible results
    combinacao_expandido_com_sem_doencas = pd.concat([sem_doencas_do_coracao, com_doencas_expandido])
    st.dataframe(combinacao_expandido_com_sem_doencas.HeartDisease.value_counts())
    # st.dataframe(combinacao_expandido_com_sem_doencas)
    #TESTANDO A REGRESSAO LOGISTICA NOS DADOS
    subamostragem = linear_model.LogisticRegression(solver='liblinear').fit(X_train, y_train)
    # subamostragem = ensemble.RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
    subamostragem_pred = subamostragem.predict(X_test)
    st.subheader('''Notas:
     - F1 score simplesmente mede a porcentagem de previsões corretas que um modelo de aprendizado de máquina fez
     - O recall é intuitivamente a habilidade do classificador em encontrar todas as amostras positivas. O melhor valor é 1 e o pior valor é 0.
    ''')
    #AQUI ESTÁ DANDO ERRO PORQUE TEMPOS QUE USAR SOMENTE AS COLUNAS NUMÉRICAS
    st.title('Regressão Logística')
    st.subheader('Precisão dos dados: {}'.format(metrics.accuracy_score(y_test, subamostragem_pred)))

    st.subheader('f1 score: {}'.format(metrics.accuracy_score(y_test, subamostragem_pred)))
    st.subheader('recall store: {}'.format(metrics.recall_score(y_test, subamostragem_pred)))
    st.subheader("Note que o nosso recall score está muito baixo usando a regressão logística, então vamos testar a RandomForest")
    st.title("RandomForest")
    subamostragem = ensemble.RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
    subamostragem_pred = subamostragem.predict(X_test)
    st.subheader('Precisão dos dados: {}'.format(metrics.accuracy_score(y_test, subamostragem_pred)))
    st.subheader('f1 score: {}'.format(metrics.accuracy_score(y_test, subamostragem_pred)))
    st.subheader('recall store: {}'.format(metrics.recall_score(y_test, subamostragem_pred)))
    st.subheader('Já em randomForest, o f1 e a precisão diminuiram um pouco, mas o recall aumentou um pouco')
    st.subheader("Referência: https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18#:~:text=Imbalanced%20classes%20are%20a%20common%20problem%20in%20machine%20learning%20classification,spam%20filtering%2C%20and%20fraud%20detection")
    

