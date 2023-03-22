import streamlit as st
import dataframe
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

df = pd.read_parquet('./heart_2020_cleaned.parquet')
labelEncoder = LabelEncoder()
oneHotEncoder = OneHotEncoder()

diabeticos = {
    'Yes': 'Yes',
    'No': 'No',
    'No, borderline diabetes' : 'No'
}
    
def categoricos_to_numericos():
    st.markdown('# Passando atributos categoricos para numericos')
    st.markdown('___')

    st.dataframe(df.head())

    st.markdown('#### O objetivo dessa fase do processamento dos dados é converter todas as colunas que são categoricas para numericas.')
    st.markdown('#### Pra isso serão utilizados dois algoritimos.')
    st.markdown('- LabelEncoder')
    st.markdown('- OneHotEncoder')
    st.markdown('#### Basicamente, o LabelEncoder transforma cada um das categorias em um numero enquanto o OneHotEncoder transforma cada categoria em uma coluna. No caso do OneHotEncoder, iremos utilizar o método "get_dummies" da biblioteca Pandas, pois ela irá executar a mesma tarefa de forma mais simples.')
    
    st.markdown('___')

    st.markdown('#### Primeiramente iremos fazer a conversão das colunas "HeartDisease", "Smoking", "Stroke", "DiffWalking", "Diabetic", "PhysicalActivity", "Asthma", "Kidney Disease", "SkinCancer", "Alcohol Drinking" e "Sex"')
    st.markdown('#### As categorias possiveis dentro desses atributos são binárias, ou seja, só existem duas opções. Uma delas será representada pelo 0 e a outra pelo 1.')
    labels_heartD = labelEncoder.fit_transform(df.HeartDisease)
    df['HeartDisease'] = labels_heartD
    labels_smoking = labelEncoder.fit_transform(df.Smoking)
    df['Smoking'] = labels_smoking
    labels_stroke = labelEncoder.fit_transform(df.Stroke)
    df['Stroke'] = labels_stroke
    labels_diffwalk = labelEncoder.fit_transform(df.DiffWalking)
    df['DiffWalking'] = labels_diffwalk
    labels_sex = labelEncoder.fit_transform(df.Sex)
    df['Sex'] = labels_sex
    dataframa = df
    df.Diabetic = df.Diabetic.map(diabeticos)
    labels_diabeticos = labelEncoder.fit_transform(df.Diabetic)
    df['Diabetic'] = labels_diabeticos
    labels_physical = labelEncoder.fit_transform(df.PhysicalActivity)
    df['PhysicalActivity'] = labels_physical
    labels_asma = labelEncoder.fit_transform(df.Asthma)
    df['Asthma'] = labels_asma
    labels_kidney = labelEncoder.fit_transform(df.KidneyDisease)
    df['KidneyDisease'] = labels_kidney
    labels_cancer = labelEncoder.fit_transform(df.SkinCancer)
    df['SkinCancer'] = labels_cancer
    labels_alcohol = labelEncoder.fit_transform(df.AlcoholDrinking)
    df['AlcoholDrinking'] = labels_alcohol
    st.dataframe(df.head())

    st.markdown('#### ')
    st.markdown(" ")
    st.markdown('#### Agora iremos adicionar o "dummies" a categoria "AgeCategory".')
    dataframa = df
    feature_Arr = pd.get_dummies(dataframa['AgeCategory'])
    dataframa = pd.concat([dataframa, feature_Arr], axis=1)
    dataframa = dataframa.drop('AgeCategory', axis=1)
    st.dataframe(dataframa.head())
    st.markdown(" ")
    st.markdown('#### "Dummies" para a categoria "Race".')
    
    feature_Arr = dataframa['Race'].str.get_dummies()
    dataframa = pd.concat([dataframa, feature_Arr], axis=1)
    dataframa = dataframa.drop('Race', axis=1)
    st.dataframe(dataframa.head())
    st.markdown('#### Após aplicar o "Dummies" para a categoria "GenHealth" nós temos o nosso conjunto de dados final com todas as colunas agora numericas.')
    feature_Arr = pd.get_dummies(dataframa['GenHealth'])
    dataframa = pd.concat([dataframa, feature_Arr], axis=1)
    dataframa = dataframa.drop('GenHealth', axis=1)
    st.dataframe(dataframa.head())

    return dataframa

