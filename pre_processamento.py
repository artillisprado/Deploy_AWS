import dataframe
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,cross_val_score

def bar_plotly(df,titulo):
    fig1 = px.bar(df, x="HeartDisease", y="qtd_HeartDisease", color="HeartDisease", title=titulo, labels={ 'qtd_HeartDisease': 'Quantidade de casos', 'HeartDisease': 'Ocorrência de doenças cardíacas', 'no_HeartDisease': 'Não', 'yes_HeartDisease': 'Sim' },text_auto=True,color_discrete_map={'some_group': 'red','some_other_group': 'green'})
    fig1.update_yaxes(showline=True, showgrid=False)
    fig1.update_xaxes(categoryorder='category ascending',showline=True, showgrid=False)
    return st.plotly_chart(fig1,use_container_width=True)

def pre_processamento():
    st.title('Pré-Processamento')

    df = dataframe.Dados.dataframe

    st.markdown(f'### Número de Linhas e Colunas {df.shape}')

    st.markdown("### Número de linhas duplicadas")

    duplicado_rows = df[df.duplicated()] 
    st.write("Número de linhas duplicadas:", duplicado_rows.shape)

    df = df.drop_duplicates() 
    duplicata_rows = df[df.duplicated()] 
    st.write("Número de linhas duplicadas após a remoção das duplicadas:", duplicata_rows.shape) 
    
    st.markdown("### Valores nulos")
    st.write(df.isnull().sum())

    st.markdown("### Balanceamento")
    #definindo todas as variavéis 
    X  = df[['Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','Asthma','Diabetic','KidneyDisease','SkinCancer','BMI']]#'AgeCategory',
    y = df['HeartDisease']

    #Treinando o modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

    #Balanceando os dados com Smote 
    oversample = SMOTE()
    X_train_resh, y_train_resh = oversample.fit_resample(X_train, y_train.ravel())

    #criando um novo dataframe com os dados balanceados
    df_balanceado = pd.DataFrame(y_train_resh,columns=['HeartDisease'])

    #montando o daframe para exibição
    depois_balanceamento=df_balanceado.groupby("HeartDisease")["HeartDisease"].count().reset_index(name='qtd_HeartDisease')
    antes_balanceamento=df.groupby("HeartDisease")["HeartDisease"].count().reset_index(name='qtd_HeartDisease')
    antes_balanceamento['HeartDisease'] = antes_balanceamento['HeartDisease'].replace({1:'Sim',0:'Não'}).astype(str)
    depois_balanceamento['HeartDisease'] = antes_balanceamento['HeartDisease'].replace({1:'Sim',0:'Não'}).astype(str)

    bar_plotly(antes_balanceamento, "Pré-Balanceamento")
    st.text_area(label="", value='Analisando o gráfico abaixo notamos que o número de casos confirmados com doenças cardíacas é muito menor do que os casos não confirmados', height=100)

    bar_plotly(depois_balanceamento,'Pós-Balanceamento')
    st.text_area(label="", value='Para fazer o balanceamento foi usado o SMOTE(Synthetic Minority Oversampling Technique) para obter melhores resultados. Já que o número de casos positivos é muito menor  do que os de casos negativos.', height=150)