import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dataframe
from mineracao import random_forest_treinada

# import plotly.express as px
def analisaCaracteristica(coluna, valor):
    print('entrou na função')
    linha_sem_tratamento = dataframe.dataframe_nao_numerico.iloc[-1:]
    linha_sem_tratamento[coluna] = valor
    dataframe_sem_tratamento_concatenado = dataframe.dataframe_nao_numerico.append(linha_sem_tratamento)
    linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)
    #criar metodo para treinar a regressao logisticar e usar o predict nessa linha_com_tratamento
    regressao_logistica = random_forest_treinada()
    previsao = regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1))
    return previsao

def pergunta1():
    def analisaCaracteristica(coluna, valor):
        print('entrou na função')
        linha_sem_tratamento = dataframe.dataframe_nao_numerico.iloc[-1:]
        linha_sem_tratamento[coluna] = valor
        dataframe_sem_tratamento_concatenado = dataframe.dataframe_nao_numerico.append(linha_sem_tratamento)
        linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)
        #criar metodo para treinar a regressao logisticar e usar o predict nessa linha_com_tratamento
        regressao_logistica = random_forest_treinada()
        previsao = regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1))
        return previsao

    st.title('É correto afirmar que características imutáveis dos indivíduos podem indicar que eles possuem ou podem vir a possuir doenças cardíacas? E existe algum fator que acompanhe essas características com frequência?')
    df = dataframe.Dados.dataframe_sem_tratamento_csv
    st.markdown('### Objetivo da Análise: Comparar os dados referentes a raça, idade e sexo afim de enxergar quais se repetem com mais frequencia entre os individuos portadores de doenças cardiacas. E em seguida analisar se algum outro dado se repete junto a essas características.')
    st.markdown('#### De início iremos comparar a quantidade de indivíduos que possuem doenças cardíacas com a quantidade de indivíduos que não possuem em cada coluna.')
    st.markdown('###### 5 primeiros itens de cada coluna.')
    st.dataframe(df[['Sex', 'AgeCategory', 'Race']].head())

    st.markdown("***")

    st.markdown('##### Entre os sexos, quem possui mais indivíduos com doenças cardíacas?')
    col1, col2 = st.columns(2)
    with col1:
        selection = st.radio("Selecionar o sexo dos individuos", ["Feminino", "Masculino"])
        
    with col2:
        range_chart = st.slider('Quantidade de individuos por sexo:', 10, 25000, 10)
    if selection == "Masculino":

        dataframe_yes = df[(df['Sex'] == 'Male')].iloc[:range_chart]
        st.markdown('##### Individuos do sexo '+selection+':')
        st.write(" 'HeartDisease'= Possuir ou não doenças cardiacas")
        fig, axes = plt.subplots()
        fig = plt.figure(figsize=(10,4), dpi=200)
        axes = sns.countplot(data=dataframe_yes,x='Sex', hue='HeartDisease', palette='Set1')
        st.pyplot(fig)
    else:

        dataframe_yes = df[(df['Sex'] == 'Female')].iloc[:range_chart]
        st.markdown('##### Individuos do sexo '+selection+':')
        st.write(" 'HeartDisease'= Possuir ou não doenças cardiacas")
        fig, axes = plt.subplots()
        fig = plt.figure(figsize=(10,4), dpi=200)
        axes = sns.countplot(data=dataframe_yes,x='Sex', hue='HeartDisease', palette='Set1')
        st.pyplot(fig)
    
    st.markdown("***")


    st.markdown('##### Entre os grupos de idade, quem possui mais indivíduos com doenças cardíacas?')
    dataframe_age_yes = df[(df['HeartDisease'] == 'Yes')]
    fig, axes = plt.subplots()
    fig = plt.figure(figsize=(10, 4), dpi=200)
    axes = sns.countplot(data=dataframe_age_yes,y='HeartDisease', hue='AgeCategory',linewidth=5, palette='Set2')
    axes.set_xlabel('Numero de indivíduos com doenças cardíacas', fontsize=10)
    axes.set_ylabel('Catergorias de idade', fontsize=10)
    axes.legend(prop={"size":10})
    st.pyplot(fig)

    st.markdown('##### Observando isoladamente cada grupo:')

    col1, col2 = st.columns(2)
    with col1:
        grupos_etarios = df['AgeCategory'].unique().tolist()
        option = st.selectbox(
        'Grupo de idade:',
        grupos_etarios)

    with col2:

        dataframe_age = df[(df['AgeCategory'] == option)]
        st.write('Observando em quais grupos a quantidade de portadores de doenças cardiacas é mais significativa.')
        fig, axes = plt.subplots()
        fig = plt.figure(figsize=(10, 4), dpi=200)

        axes = sns.countplot(data=dataframe_age,y='AgeCategory', hue='HeartDisease',linewidth=5, palette='Set2')
        axes.set_xlabel('Numero de individuos com doenças cardiacas', fontsize=23)
        axes.set_ylabel('Catergorias de idade', fontsize=23)

        axes.legend(prop={"size":23})
        st.pyplot(fig)

    
    st.markdown("***")

    st.markdown('##### Entre as raças, quem possui mais individuos com doenças cardiacas?')
    col1, col2 = st.columns(2)
    with col1:
        grupos_raciais = df['Race'].unique().tolist()
        raca = st.selectbox(
        'Selecione o grupo racial:',
        grupos_raciais)
        
    with col2:
        range = st.slider('Quantidade individuos por raça: ', 1000, 25000, 1000)

    dataframe_race = df[(df['Race'] == raca)].iloc[:range]
    st.markdown('##### Individuos '+raca+':')
    st.write(" 'HeartDisease'= Possuir ou não doenças cardiacas")
    fig, axes = plt.subplots()
    fig = plt.figure(figsize=(10,4), dpi=200)
    axes = sns.countplot(data=dataframe_race,x='Race', hue='HeartDisease', palette='Set1')
    st.pyplot(fig)

    
    dataframe_yes_race = df[(df['HeartDisease'] == 'Yes')].iloc[:range]
    st.markdown('##### Comparando a quantidade de individuos portadores de doenças cardiacas em cada um dos grupos raciais')
    fig, axes = plt.subplots()
    fig = plt.figure(figsize=(10,4), dpi=200)
    axes = sns.countplot(data=dataframe_yes_race,x='HeartDisease', hue='Race', palette='Set2')
    st.pyplot(fig)
    
    st.markdown("***")

    st.markdown('##### Analisando os individuos através da junção da sua faixa etaria, sua raça e seu sexo:')
    col1, col2, col3 = st.columns(3)
    with col1:
        grupos_raciais = df['Race'].unique().tolist()
        option_race = st.selectbox(
        'Selecione o grupo racial :',
        grupos_raciais)
        
    with col2:
        option_sex = st.radio("Selecione o sexo dos individuos", ["Female", "Male"])
    
    with col3:
        grupos_etarios = df['AgeCategory'].unique().tolist()
        option_age = st.selectbox(
        'Selecione a faixa etaria:',
        grupos_etarios)

    dataframe_yes = df[(df['Race'] == option_race ) & (df['Sex'] == option_sex ) & (df['AgeCategory'] == option_age )].iloc[:range]
    st.markdown('##### Faixa etaria: ' +option_age)
    st.markdown('##### Grupo racial: ' +option_race)
    st.markdown('##### Sexo: ' +option_sex)

    st.write(" 'HeartDisease'= Possuir ou não doenças cardiacas")
    fig, axes = plt.subplots()
    fig = plt.figure(figsize=(10,4), dpi=200)
    axes = sns.countplot(data=dataframe_yes,x='HeartDisease', palette='Set1')
    st.pyplot(fig)

#TESTES PARA COMPARAÇÃO DO SEXO
    analisaCaracteristica('Race', 'White')
    
    st.markdown('### Caracteristicas individuo 2:')
    linha_sem_tratamento = dataframe.dataframe_nao_numerico.iloc[-1:]
    linha_sem_tratamento['Sex'] = 'Male'
    linha_sem_tratamento['Race'] = 'White'
    linha_sem_tratamento['AgeCategory'] = '75-79'
    st.dataframe(linha_sem_tratamento)
    dataframe_sem_tratamento_concatenado = dataframe.dataframe_nao_numerico.append(linha_sem_tratamento)
    linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)
    #criar metodo para treinar a regressao logisticar e usar o predict nessa linha_com_tratamento
    regressao_logistica = random_forest_treinada()
    previsao2 = regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1))
    # st.markdown(f'##### O individuo 1, representando o sexo feminino, apresentou {float(previsao[0][1]) * 100:.3f}% de chances de possuir uma doença cardiaca, enquanto o individuo 2, que representa o sexo masculino, apresentou {float(previsao2[0][1]) * 100:.3f}%')
    # diferenca = (float(previsao2[0][1]) * 100) - (float(previsao[0][1]) * 100)
    # st.markdown(f'##### Como é possivel ver, uma diferença de {diferenca:.3f}% para {"o sexo masculino" if previsao2[0][1] > previsao[0][1] else "o sexo feminino"}.')
#TESTES PARA COMPARAÇÃO DO SEXO

    st.markdown('### Caracteristicas individuo 1:')
    linha_sem_tratamento = dataframe.dataframe_nao_numerico.iloc[-1:]
    linha_sem_tratamento['Sex'] = 'Female'
    linha_sem_tratamento['Race'] = 'Black'
    linha_sem_tratamento['AgeCategory'] = '75-79'
    st.dataframe(linha_sem_tratamento)
    dataframe_sem_tratamento_concatenado = dataframe.dataframe_nao_numerico.append(linha_sem_tratamento)
    linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)
    #criar metodo para treinar a regressao logisticar e usar o predict nessa linha_com_tratamento
    regressao_logistica = random_forest_treinada()
    previsao = regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1))
    
    st.markdown('### Caracteristicas individuo 2:')
    linha_sem_tratamento = dataframe.dataframe_nao_numerico.iloc[-1:]
    linha_sem_tratamento['Sex'] = 'Male'
    linha_sem_tratamento['Race'] = 'White'
    linha_sem_tratamento['AgeCategory'] = '75-79'
    st.dataframe(linha_sem_tratamento)
    dataframe_sem_tratamento_concatenado = dataframe.dataframe_nao_numerico.append(linha_sem_tratamento)
    linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)
    #criar metodo para treinar a regressao logisticar e usar o predict nessa linha_com_tratamento
    regressao_logistica = random_forest_treinada()
    previsao2 = regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1))

    st.markdown('### Caracteristicas individuo 3:')
    linha_sem_tratamento = dataframe.dataframe_nao_numerico.iloc[-1:]
    linha_sem_tratamento['Sex'] = 'Male'
    linha_sem_tratamento['Race'] = 'Asian'
    linha_sem_tratamento['AgeCategory'] = '75-79'
    st.dataframe(linha_sem_tratamento)
    dataframe_sem_tratamento_concatenado = dataframe.dataframe_nao_numerico.append(linha_sem_tratamento)
    linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)
    #criar metodo para treinar a regressao logisticar e usar o predict nessa linha_com_tratamento
    regressao_logistica = random_forest_treinada()
    previsao3 = regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1))

    st.markdown('### Caracteristicas individuo 4:')
    linha_sem_tratamento = dataframe.dataframe_nao_numerico.iloc[-1:]
    linha_sem_tratamento['Sex'] = 'Male'
    linha_sem_tratamento['Race'] = 'Hispanic'
    linha_sem_tratamento['AgeCategory'] = '75-79'
    st.dataframe(linha_sem_tratamento)
    dataframe_sem_tratamento_concatenado = dataframe.dataframe_nao_numerico.append(linha_sem_tratamento)
    linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)
    #criar metodo para treinar a regressao logisticar e usar o predict nessa linha_com_tratamento
    regressao_logistica = random_forest_treinada()
    previsao4 = regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1))

    st.markdown('### Caracteristicas individuo 4:')
    linha_sem_tratamento = dataframe.dataframe_nao_numerico.iloc[-1:]
    linha_sem_tratamento['Sex'] = 'Male'
    linha_sem_tratamento['Race'] = 'American Indian/Alaskan Native'
    linha_sem_tratamento['AgeCategory'] = '75-79'
    st.dataframe(linha_sem_tratamento)
    dataframe_sem_tratamento_concatenado = dataframe.dataframe_nao_numerico.append(linha_sem_tratamento)
    linha_com_tratamento = dataframe.returnDataFrame(dataframe_sem_tratamento_concatenado)
    #criar metodo para treinar a regressao logisticar e usar o predict nessa linha_com_tratamento
    regressao_logistica = random_forest_treinada()
    previsao5 = regressao_logistica.predict_proba(linha_com_tratamento.iloc[[-1]].drop('HeartDisease', axis=1))
    d = {'Black': [previsao[0][1] * 100],
         'White': [previsao2[0][1] * 100],
         'Asian': [previsao3[0][1] * 100],
         'Hispanic': [format((previsao4[0][1] * 100), '.1f')],
         'American Indian/Alaskan Native': [(previsao5[0][1]*100)]
        }

    tabelaRacas = pd.DataFrame(data=d)
    st.dataframe(tabelaRacas)
    st.markdown(f'##### O individuo 1, representando o grupo de pessoas negras, apresentou {float(previsao[0][1]) * 100:.3f}% de chances de possuir uma doença cardiaca, enquanto o individuo 2, que representa o grupo de pessoas brancas, apresentou {float(previsao2[0][1]) * 100:.3f}%, o individuo 3, representando o grupo de pessoas asiaticas, apresentou {float(previsao3[0][1]) * 100:.3f}%, o individuo 4, representando o grupo de pessoas hispanicas ou latinas, apresentou {float(previsao4[0][1]) * 100:.3f}%.')
    diferenca = (float(previsao2[0][1]) * 100) - (float(previsao[0][1]) * 100)
    st.markdown(f'##### Como é possivel ver, uma diferença de {diferenca:.3f}% para {"o sexo masculino" if previsao2[0][1] > previsao[0][1] else "o sexo feminino"}.')



    st.markdown("***")

    

    st.markdown('### O que se pode entender sobre a análise de cada coluna que representam as características imutáveis dos indivíduos:')
    st.write(" Começando pelo sexo dos indivíduos, é visível que os homens possuem mais indivíduos portadores de doenças cardiacas do que as melhores. Em seguida analisando a faixa etaria, é possivel observar que as pessoas de 80 anos ou mais é o grupo com mais portadores, mesmo após serem analisados isoladamente. Quanto ao grupo racial de cada um fica claro que os brancos possuem mais pessoas portadoras de doenças cardiacas entre si.")
    st.write("  Por fim, essas análises nos leva a indicação que homens brancos a partir dos 80 anos são os que, considerando suas caracteristicas imutaveis, mais possuem doenças cardiacas.")


    

