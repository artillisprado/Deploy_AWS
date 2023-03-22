import streamlit as st
from dataframe import Dados
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import dataframe
from mineracao import random_forest_treinada

def pie_chart_com_doenca(coluna):#quantidade
    labels = f'Possui {coluna}', f'Não possui {coluna}'
    df_limitado = Dados.dataframe.query("HeartDisease == 1")
    df_com_doenca_coracao_e_outra_doenca = df_limitado.query(f"{coluna} == 1")
    porcentagem_possui_outra_doenca = len(df_com_doenca_coracao_e_outra_doenca.index) / len(df_limitado.index)
    sizes = [porcentagem_possui_outra_doenca, 1 - porcentagem_possui_outra_doenca]
    explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

def pie_chart_sem_doenca(coluna): #quantidade
    labels = f'Possui {coluna}', f'Não possui {coluna}'
    df_limitado = Dados.dataframe.query("HeartDisease == 0")
    # st.dataframe(df_limitado)
    df_sem_doenca_coracao_e_com_outra_doenca = df_limitado.query(f"{coluna} == 1")
    porcentagem_possui_outra_doenca = len(df_sem_doenca_coracao_e_com_outra_doenca.index) / len(df_limitado.index)
    sizes = [porcentagem_possui_outra_doenca, 1 - porcentagem_possui_outra_doenca]
    explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)
def previsoes(Diabetic, Asthma, SkinCancer, KidneyDisease):
    linha_sem_tratamento = dataframe.dataframe_nao_numerico.iloc[-1:]
    linha_sem_tratamento['Asthma'] = Asthma
    linha_sem_tratamento['Diabetic'] = Diabetic
    linha_sem_tratamento['SkinCancer'] = SkinCancer
    linha_sem_tratamento['KidneyDisease'] = KidneyDisease
    # st.dataframe(linha_sem_tratamento)
    dataframe_sem_tratamento_concatenado = dataframe.dataframe_nao_numerico.append(linha_sem_tratamento)
    linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)
    #criar metodo para treinar a regressao logisticar e usar o predict nessa linha_com_tratamento
    regressao_logistica = random_forest_treinada()
    return regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1))

def pergunta_2():
    st.title('Doenças provindas de outros órgãos do corpo, podem ser um indicativo de doenças cardíacas ?')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Dados com indivíduos que têm ou já tiveram alguma doença no coração')
        # range_chart_com_doenca = st.slider('Quantidade de dados no gráfico de indivíduos com doença no coração', 0, len(Dados.df_somente_com_doencas_do_coracao.index), 27373)
        pie_chart_com_doenca('Diabetic')
        pie_chart_com_doenca('Asthma')
        pie_chart_com_doenca('SkinCancer')
        pie_chart_com_doenca('KidneyDisease')
        

    with col2:
        st.subheader('Dados com indivíduos que não possui nenhuma doença no coração')
        # range_chart_sem_doenca = st.slider('Quantidade de dados no gráfico de indivíduos sem doença no coração', 0, len(Dados.df_somente_sem_doencas_do_coracao.index), 27373)
        pie_chart_sem_doenca('Diabetic')
        pie_chart_sem_doenca('Asthma')
        pie_chart_sem_doenca('SkinCancer')
        pie_chart_sem_doenca('KidneyDisease')
    # st.markdown('### Caracteristicas individuo 1 - Diabetes:')
    previsao_diabetes = previsoes("Yes","No","No","No")
    # st.markdown('### Caracteristicas individuo 2 - Asma:')
    previsao_asma = previsoes("No","Yes","No","No")
    # st.markdown('### Caracteristicas individuo 3 - câncer de pele:')
    previsao_cancer_de_pele = previsoes("No","No","Yes","No")
    # st.markdown('### Caracteristicas individuo 4 - doença nos rins:')
    previsao_rins = previsoes("No","No","No","Yes")
    d = {'Diabetes': [format(previsao_diabetes[0][1] * 100, '.1f')],
        'Asma': [format(previsao_asma[0][1] * 100, '.1f')],
        'Câncer de pele': [format(previsao_cancer_de_pele[0][1] * 100, '.1f')],
        'Doença nos rins': [format((previsao_rins[0][1] * 100), '.1f')],
    }
    st.markdown('### Previsão usando regressão logística')
    st.dataframe(pd.DataFrame(data=d))
    st.markdown(f'Analisando os gráficos com a mesma quantidade de indivíduos que já tiveram e que não tiveram doenças cardíacas,  podemos supor que há um leve indicativo de ter uma doença do coração já tendo uma das três doenças analisadas, com uma maior ênfase em diabetes e doenças renais. Podemos ver que tivemos uma maior diferença percentual nas doenças renais, com uma porcentagem bem pequena de indivíduos que não tiveram doenças do coração, e um crescimento de 10% nas pessoas que tiveram, já em diabetes o crescimento foi maior, foi de aproximadamente 21%.')
        



    #colunas para analisar, Diabetic, Asthma, SkinCancer, HeartDisease, 