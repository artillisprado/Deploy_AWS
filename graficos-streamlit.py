# from ast import With
# from tkinter import W
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pergunta_1, pergunta_2, pergunta_3 
import featureImportance
import outlier, pre_processamento, mineracao
import dataframe
import tratamento_desbalanceamento_dados
import categoricos_numericos
from teste_voce import testeVoce



opcoes = ["Inicio", "🤖 Tratando o desbalanceamento dos dados", "🧑 Caracteristicas imutáveis", "🧠 Doenças vindas de outros órgãos", "⛹️ Hábitos", "📈 Outlier", "⚙️ Pré-Processamento", "🔢 De categoricos para numéricos", "🤝 Feature Importance", "💼 Mineração de Dados","❤️ Teste com seus dados!"]
with st.sidebar:
    st.markdown('# 🆙GRUPO UPSCALE ')
    st.markdown('## MENU PRINCIPAL 👈')
    selection = st.radio("", opcoes)
# selection = st.sidebar



def inicio():
    st.markdown('# 🆙GRUPO UPSCALE')
    st.title('Ánalise de dados para a cadeira PI3')
    st.markdown('### A ánalise dos dados do conjunto de dados HEART_DISEASE tem como o objetivo fazer o uso de um olhar critico e questionador sobre os dados encontrados para que através desta análise seja possivel responder as perguntas deste trabalho e chegar a uma conclusão satisfatoria.')
    st.markdown('### As perguntas são 3: ')
    st.markdown('- É correto afirmar que características imutáveis dos indivíduos podem indicar que eles possuem ou podem vir a possuir doenças cardíacas? E existe algum fator que acompanhe essas características com frequência?')
    st.markdown('- Doenças provindas de outros órgãos do corpo, podem ser um indicativo de doenças cardíacas ?')
    st.markdown('- É possível prever que um indivíduo tem um grande potencial de ter uma doença cardíaca a partir dos seus hábitos?')

    df = dataframe.Dados.dataframe

    chart_data = pd.DataFrame(list(zip(df['BMI'].tolist(),df['PhysicalHealth'].tolist(),df['MentalHealth'].tolist())),
    columns = ['Body Mass Index (BMI)','PhysicalHealth','MentalHealth'])
    st.dataframe(df.head())
    range_chart = st.slider('Quantidade de dados no gráfico', 0, 300, 30)
    st.area_chart(chart_data.head(range_chart))
    st.write('O objetivo deste trabalho será analisar e expôr aqui informações necessárias para a construção do artigo.')



if selection == "🧑 Caracteristicas imutáveis":
    pergunta_1.pergunta1()
elif selection == "Inicio":
    inicio()
elif selection == "🧠 Doenças vindas de outros órgãos":
    pergunta_2.pergunta_2()
elif selection == "⛹️ Hábitos":
    pergunta_3.pergunta3()
elif selection == "📈 Outlier":
    outlier.outlier()
elif selection == "🤖 Tratando o desbalanceamento dos dados":
    tratamento_desbalanceamento_dados.desbalanceamento_dados()
elif selection == "🔢 De categoricos para numéricos":
    categoricos_numericos.categoricos_to_numericos()
# elif selection == "🤝 Feature Importance":
#     featureImportance.featureImportance()
elif selection == "⚙️ Pré-Processamento":
    pre_processamento.pre_processamento()
elif selection == "💼 Mineração de Dados":
    mineracao.mineracao()
elif selection == "❤️ Teste com seus dados!":
    testeVoce()
