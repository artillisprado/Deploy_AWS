import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import seaborn as sns
import pandas as pd
import dataframe

def graph(df, selection):
    bmi = px.box(df ,y=df[selection])
    return st.plotly_chart(bmi)

def outlier():
    st.title('📈 Detecção de Outliers')

    df = dataframe.Dados.dataframe

    #Detecção de Outliers
    st.markdown("### Análise de outliers com variáveis númericas")
    selection = st.selectbox("Selecione uma categoria", ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"])

    if selection == "BMI":
        graph(df, "BMI")
    elif selection == "PhysicalHealth":
        graph(df, "PhysicalHealth")
    elif selection == "MentalHealth":
        graph(df, "MentalHealth")
    else:
        graph(df, "SleepTime") 

    
    #Encontre o intervalo interquartil 
    Q1 = df.quantile(0.25) 
    Q3 = df.quantile(0.75) 
    IQR = Q3-Q1 
    st.markdown("### Intervalo interquartil")
    st.write("Método de remoção de Outlier")
    st.write(IQR) 

    data2 = df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis= 1)] 
    st.write(f"Resultado após o cálculo de estatística com o interquartil: {data2.shape}.")

    st.markdown("### Mapa de Correlação")
    pearsonCorr = df.corr(method='spearman') 

    fig = px.imshow(
    pearsonCorr.round(2), 
    text_auto=True, 
    color_continuous_scale="greens",
    zmin=-1,
    zmax=1)

    st.write(fig)