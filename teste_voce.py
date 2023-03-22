import streamlit as st
from mineracao import random_forest_treinada
from dataframe import Dados
import dataframe
import pandas as pd

def testeVoce():
    df = Dados.dataframe_sem_tratamento
    df_sem_tratamento = Dados.dataframe.iloc[0:0]
    col1, col2, col3 = st.columns(3)
    with col1:
        option_race = st.selectbox(
        'Selecione o grupo racial :',
        df['Race'].unique().tolist())
        option_HeartDisease= st.radio(
        'já relataram ter doença cardíaca coronária (DAC) ou infarto do miocárdio (IM)?',
        ["Yes", "No"])
        option_AlcoholDrinking = st.radio(
        'Voce consome muita bebida alcoólica? ​​(homens adultos que bebem mais de 14 bebidas por semana e mulheres adultas que bebem mais de 7 bebidas por semana',
        ["Yes", "No"])
        option_MentalHealth = st.number_input(
        max_value=30, min_value=0,label='Pensando em sua saúde mental, por quantos dias nos últimos 30 dias sua saúde mental não foi boa?')
        # PhysicalActivity
        option_PhysicalActivity = st.radio(
        'fazer atividade física ou exercício durante os últimos 30 dias, além do trabalho regular',
        ["Yes", "No"])
        option_Asthma = st.radio(
        'Voce ja teve ou tem asma?',
        ["Yes", "No"])

        
    with col2:
        option_sex = st.radio("Selecione o sexo", ["Female", "Male"])
        option_bmi = st.number_input(
        max_value=100, min_value=0,label='IMC (O IMC é uma medida da inclinação ou corpulência de uma pessoa com base em sua altura e peso, e tem a intenção de quantificar a massa tecidual)')
        option_stroke = st.radio(
        'voce ja teve um derrame?',
        ["Yes", "No"])
        option_DiffWalking = st.radio(
        'Você tem muita dificuldade para andar ou subir escadas?',
        ["Yes", "No"])
        option_GenHealth = st.selectbox(
        'Você diria que, em geral, sua saúde é...',
        df['GenHealth'].unique().tolist())
        option_KidneyDisease = st.radio(
        'Sem incluir pedras nos rins, infecção na bexiga ou incontinência, já lhe disseram que tinha doença renal?',
        ["Yes", "No"])
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')

    
    with col3:
        option_age = st.selectbox(
        'Selecione a faixa etaria:',
        df['AgeCategory'].unique().tolist())
        option_smoking = st.radio(
        'Você fumou pelo menos 100 cigarros em toda a sua vida? [Nota: 5 maços = 100 cigarros]?',
        ["Yes", "No"])
        option_PhysicalHealth = st.number_input(
        max_value=30, min_value=0,label='Agora pensando em sua saúde física, que inclui doenças físicas e lesões, por quantos dias nos últimos 30:')
        # Diabetic
        option_Diabetic = st.radio(
        'Você ja teve ou tem diabetes ?',
        ["Yes", "No"])
        option_SleepTime = st.number_input(
        max_value=24, min_value=0,label='Em média, quantas horas de sono você tem em um período de 24 horas?')
        option_SkinCancer = st.radio(
        'Voce tem ou ja teve cancer de pele',
        ["Yes", "No"])
    if st.button('Previsao!'): 
        linha_sem_tratamento = pd.DataFrame([[option_HeartDisease,option_bmi,option_smoking,option_AlcoholDrinking,option_stroke,option_PhysicalHealth,option_MentalHealth,option_DiffWalking,option_sex,option_age,option_race,option_Diabetic,option_PhysicalActivity,option_GenHealth,option_SleepTime,option_Asthma,option_KidneyDisease,option_SkinCancer]], columns=['HeartDisease','BMI','Smoking','AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','SleepTime','Asthma','KidneyDisease','SkinCancer'])
        dataframe_sem_tratamento_concatenado = dataframe.dataframe_nao_numerico.append(linha_sem_tratamento)
        linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)
        #criar metodo para treinar a regressao logisticar e usar o predict nessa linha_com_tratamento
        regressao_logistica = random_forest_treinada()
        linha_com_tratamento.shape
        st.write(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1).shape)
        previsao = regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1))
        st.subheader(f'Voce tem {float(previsao[0][1]) * 100:.3f}% de chance de ter uma doenca cardieca segundo nosso algoritmo! :)')
