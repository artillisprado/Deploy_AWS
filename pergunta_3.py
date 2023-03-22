import streamlit as st
from dataframe import Dados
import pandas as pd
import plotly.express as px
import dataframe
from mineracao import random_forest_treinada

def func(df, sexo, imc, idade, bebe, fuma):
    filter_sales_units = df[(df['Sex'] == sexo) & (df["classificacao"] == imc) & (df['AgeCategory'] == idade) & (df['Smoking'] == fuma) & (df["AlcoholDrinking"] == bebe)]


    dataframe_sem_tratamento_concatenado = df.append(filter_sales_units.drop('classificacao', axis=1))
    linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)

    regressao_logistica = random_forest_treinada()
    previsao = regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1).drop('classificacao', axis=1))
    return previsao

def pergunta3():

    st.title("⛹️ Análise de Hábitos")
    st.write("***")

    st.markdown('### Pergunta 3: ')
    st.write("É possível prever que um indivíduo tem um grande potencial de ter uma doença cardíaca a partir dos seus hábitos ?")
    st.write("R: Analisar os dados referentes a qualidade de vida do indivíduo, como Consumo de álcool, tabagismo, tempo de sono, atividade Física e idade, e procurar padrões quantitativos que mais apresentam doenças cardíacas.")
    
    st.write("***")

    df = dataframe.dataframe_nao_numerico


    st.markdown('### Fumante ')
    #count_sexMale = int(dataframe[(dataframe["Smoking"] == "Yes") & (dataframe["Sex"] == "Male") & (dataframe["HeartDisease"] == "Yes")]["Smoking"].count())
    #count_sexFemale = int(dataframe[(dataframe["Smoking"] == "Yes") & (dataframe["Sex"] == "Female") & (dataframe["HeartDisease"] == "Yes")]["Smoking"].count())
    #print(count_sexMale, '\n', count_sexFemale)

    fig_smoking = px.histogram(df, x="Smoking",
             color='HeartDisease', barmode='group',
             text_auto='.2s',
             labels={'Sex':'Sexo','Smoking':'Fumante', 'HeartDisease': 'Doença Cardíaca'},
             height=400)
    fig_smoking.update_layout(yaxis_title='Quantidade de indivíduos')
    fig_smoking.update_yaxes(showline=True, showgrid=False)
    fig_smoking.update_xaxes(categoryorder='category ascending',showline=True, showgrid=False)
    st.write(fig_smoking)

    st.markdown('### Diabético ')
    
    fig_diabetic = px.histogram(df, x="Diabetic",
             color='HeartDisease', barmode='group',
             text_auto='.2s',
             labels={'Sex':'Sexo','Diabetic':'Diabético', 'HeartDisease': 'Doença Cardíaca'},
             height=400)
    fig_diabetic.update_layout(yaxis_title='Quantidade de indivíduos')
    fig_diabetic.update_yaxes(showline=True, showgrid=False)
    fig_diabetic.update_xaxes(categoryorder='category ascending',showline=True, showgrid=False)
    st.write(fig_diabetic)

    st.markdown('### Alcool ')
    fig_Alcohol = px.histogram(df, x="AlcoholDrinking",
            color='HeartDisease', barmode='group',
            text_auto='.2s',
            labels={'Sex':'Sexo','AlcoholDrinking':'Alcool', 'HeartDisease': 'Doença Cardíaca'},
            height=400)
    fig_Alcohol.update_layout(yaxis_title='Quantidade de indivíduos')
    fig_Alcohol.update_yaxes(showline=True, showgrid=False)
    fig_Alcohol.update_xaxes(categoryorder='category ascending',showline=True, showgrid=False)
    st.write(fig_Alcohol)

    st.markdown('### Hora de Dormir ')
    figsleeptime = px.histogram(df, x="SleepTime", barmode='group',
             color='HeartDisease', histfunc='count', text_auto='.2s',
             labels={'SleepTime':'Horas de Sono', 'HeartDisease': 'Doença Cardíaca'},
             height=400)
    figsleeptime.update_layout(yaxis_title='Quantidade de indivíduos')
    figsleeptime.update_yaxes(showline=True, showgrid=False)
    figsleeptime.update_xaxes(categoryorder='category ascending',showline=True, showgrid=False)
    st.write(figsleeptime)

    st.markdown('### IMC ')
    df['BMI'] = df['BMI'].astype('float')
    df.loc[df['BMI'] <= 18.5, 'classificacao'] = 'MAGREZA'
    df.loc[(df['BMI'] >= 18.5) & (df['BMI'] <= 24.9), 'classificacao'] = 'NORMAL'
    df.loc[(df['BMI'] >= 25.0) & (df['BMI'] <= 29.9), 'classificacao'] = 'SOBREPESO'
    df.loc[(df['BMI'] >= 30.0) & (df['BMI'] <= 39.9), 'classificacao'] = 'OBESIDADE'
    df.loc[df['BMI'] >= 40.0, 'classificacao'] = 'OBESIDADE GRAVE'



    count_magreza = int(
        df[(df["classificacao"] == "MAGREZA")]["classificacao"].count())

    count_normal = int(
        df[(df["classificacao"] == "NORMAL")]["classificacao"].count())

    count_obesidade = int(
        df[(df["classificacao"] == "OBESIDADE")]["classificacao"].count())

    count_obesidadegrave = int(
        df[(df["classificacao"] == "OBESIDADE GRAVE")]["classificacao"].count())

    count_sobrepeso = int(
        df[(df["classificacao"] == "SOBREPESO")]["classificacao"].count())
    
    placeholder = st.empty()

    with placeholder.container():

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        kpi1.metric(
                label="Magreza",
                value=count_magreza,
                delta=count_magreza,
        )
        
        kpi2.metric(
                label="Normal",
                value=count_normal,
                delta=count_normal,
        )
        
        kpi3.metric(
                label="Obesidade",
                value=count_obesidade,
                delta=count_obesidade,
        )

        kpi4.metric(
                label="Obesidade Grave",
                value=count_obesidadegrave,
                delta=count_obesidadegrave,
        )

        kpi5.metric(
                label="SobrePeso",
                value=count_sobrepeso,
                delta=count_sobrepeso,
        )

    fig_bmi = px.histogram(df, x='classificacao',
            y='BMI', color="HeartDisease", barmode='group',
            histfunc='count', text_auto='.2s',
            labels={'classificacao':'IMC', 'HeartDisease': 'Doença Cardíaca'},
            height=400)
    fig_bmi.update_layout(yaxis_title='Quantidade de indivíduos')
    fig_bmi.update_yaxes(showline=True, showgrid=False)
    fig_bmi.update_xaxes(categoryorder='category ascending',showline=True, showgrid=False)
    st.write(fig_bmi)

    st.markdown('### Idade ')
    figagecategory = px.histogram(df, x="AgeCategory", color="HeartDisease", barmode='group',
             histfunc='count', text_auto='.2s',
             labels={'AgeCategory':'Idade', 'HeartDisease': 'Doença Cardíaca'},
             height=400)
    figagecategory.update_layout(yaxis_title='Quantidade de indivíduos')
    figagecategory.update_yaxes(showline=True, showgrid=False)
    figagecategory.update_xaxes(categoryorder='category ascending',showline=True, showgrid=False)
    st.write(figagecategory)

    col1, col2, col3, col4 = st.columns(4)

    option_bebe = col1.selectbox(
    'Bebe',
    ("Yes","No"))

    option_fuma = col2.selectbox(
    'Fuma',
    ("Yes","No"))

    option_sex = col3.selectbox(
    'Sexo',
    ("Male","Female"))

    option_age = col4.selectbox(
    'Idade',
    ("18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80 or older"))

    previsao = func(df, option_sex, 'MAGREZA', option_age, option_bebe, option_fuma)
    previsao2 = func(df, option_sex, 'NORMAL', option_age, option_bebe, option_fuma)
    previsao3 = func(df, option_sex, 'OBESIDADE', option_age, option_bebe, option_fuma)
    previsao4 = func(df, option_sex, 'OBESIDADE GRAVE', option_age, option_bebe, option_fuma)
    previsao5 = func(df, option_sex, 'SOBREPESO', option_age, option_bebe, option_fuma)


    p = {'MAGREZA': [format(previsao[0][1] * 100, '.1f')],
        'NORMAL': [format(previsao2[0][1] * 100, '.1f')],
        'OBESIDADE': [format(previsao3[0][1] * 100, '.1f')],
        'OBESIDADE GRAVE': [format((previsao4[0][1] * 100), '.1f')],
        'SOBREPESO': [format((previsao5[0][1] * 100), '.1f')],
    }

    st.markdown('### Previsão usando regressão logística')
    # st.dataframe(pd.DataFrame(data=d))
    st.dataframe(pd.DataFrame(data=p))

    #st.write('### ')
    st.subheader(f'''Com base nas análises feitas nos gráficos com a mesma quantidade de indivíduos que já tiveram e que não tiveram as doenças cardíaca
    Feita a análise sobre tabagismo entre indivíduos fumantes, mostrou-se que cerca de 16.037 casos de pessoas com doenças cardíacas, 
    enquanto que no Alcool são 1.141 casos. Outra análise feita com os dados sobre as horas de sono mostrou um índice maior para quem dorme entre as horas 6,7 e 8 horas.
    Os dados referente ao IMC Normal, Obesidade e SobrePeso mostraram-se com indíces ainda maiores. Uma análise feita com as idades mostrou que um aumento mais elevado e exponencial dos 45 anos até 74, sendo o maior a partir dos 80 anos.
    
    ''')
