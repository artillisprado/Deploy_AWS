import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import model_selection, utils
import streamlit as st


dataframe_nao_numerico = pd.read_parquet('./heart_2020_cleaned.parquet')
labelEncoder = LabelEncoder()
oneHotEncoder = OneHotEncoder()

# def tratamento_csv(dataframe,list_columns):
#     #nas colunas booleanas trocar 'sim' e 'não' por 0 e 1
#     for column in list_columns:
#         if column == 'Diabetic':
#             dataframe[column] = np.where(dataframe[column] == 'No, borderline diabetes', 0, dataframe[column])
#             dataframe[column] = np.where(dataframe[column] == 'Yes (during pregnancy)', 1, dataframe[column])
#         dataframe[column] = np.where(dataframe[column] == 'Yes', 1, dataframe[column])
#         dataframe[column] = np.where(dataframe[column] == 'No', 0, dataframe[column])
#         dataframe[f'{column}'] = pd.to_numeric(dataframe[f'{column}'])
#     return dataframe



def returnDataFrame(df):
    diabeticos = {
    'Yes': 'Yes',
    'Yes (during pregnancy)': 'Yes',
    'No': 'No',
    'No, borderline diabetes' : 'No'
    }
    
    labels_heartD = labelEncoder.fit_transform(df.HeartDisease)
    df['HeartDisease'] = labels_heartD
    # print(df)
    labels_smoking = labelEncoder.fit_transform(df.Smoking)
    df['Smoking'] = labels_smoking
    # print(df)
    labels_stroke = labelEncoder.fit_transform(df.Stroke)
    df['Stroke'] = labels_stroke
    # print(df)
    labels_diffwalk = labelEncoder.fit_transform(df.DiffWalking)
    df['DiffWalking'] = labels_diffwalk
    labels_sex = labelEncoder.fit_transform(df.Sex)
    df['Sex'] = labels_sex
    dataframe = df
    # print(df)
    df.Diabetic = df.Diabetic.map(diabeticos)
    # print(df.Diabetic)
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
    # print(df)

    feature_Arr = pd.get_dummies(dataframe['AgeCategory'])
    dataframe = pd.concat([dataframe, feature_Arr], axis=1)
    dataframe = dataframe.drop('AgeCategory', axis=1)
    # print(dataframa['Race'])
    feature_Arr = pd.get_dummies(dataframe['Race'])
    dataframe = pd.concat([dataframe, feature_Arr], axis=1)
    dataframe = dataframe.drop('Race', axis=1)
    # print(dataframa)
    feature_Arr = pd.get_dummies(dataframe['GenHealth'])
    dataframe = pd.concat([dataframe, feature_Arr], axis=1)
    dataframe = dataframe.drop('GenHealth', axis=1)
    # st.write(dataframe)
    return dataframe


def balanceamentoDados(df):
    y = df.HeartDisease
    X = df.drop('HeartDisease', axis=1)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=27)

    X = pd.concat([X_train, y_train], axis=1)
    sem_doencas_do_coracao = X[X.HeartDisease==0]
    com_doencas_do_coracao = X[X.HeartDisease==1]
    com_doencas_expandido = utils.resample(com_doencas_do_coracao,
                          replace=True, # sample with replacement
                          n_samples=len(sem_doencas_do_coracao), # match number in majority class
                          random_state=27) # reproducible results
    combinacao_expandido_com_sem_doencas = pd.concat([sem_doencas_do_coracao, com_doencas_expandido])
    return combinacao_expandido_com_sem_doencas

class Dados:
    
    dataframe_sem_tratamento = pd.read_parquet('./heart_2020_cleaned.parquet')
    dataframe_sem_tratamento_csv = pd.read_csv('./heart_2020_cleaned.csv')
    dataframe = returnDataFrame(dataframe_sem_tratamento)
    dataframe = balanceamentoDados(dataframe)
    # dataframe_somente_com_colunas_numericas = dataframe.drop(columns=['AgeCategory','GenHealth', 'Race','Sex', ])
    # dataframe['SkinCancer'] = np.where(dataframe['SkinCancer'] == 'Yes', 1, dataframe['SkinCancer'])
    # dataframe['SkinCancer'] = np.where(dataframe['SkinCancer'] == 'No', 0, dataframe['SkinCancer'])
    df_somente_com_doencas_do_coracao = dataframe.query("HeartDisease == 0")
    df_somente_sem_doencas_do_coracao = dataframe.query("HeartDisease == 1")
    # dataframe['Sex'] = dataframe['Sex'].replace({'Male':0,'Female':1}).astype(np.uint8)
    # dataframe['GenHealth'] = dataframe['GenHealth'].replace({'Excellent':0,'Fair':1,'Good':2,'Poor':3,'Very good':4}).astype(np.uint8)
    # dataframe['Race'] = dataframe['Race'].replace({'American Indian/Alaskan Native':0,'Asian':1,'Black':2,'Hispanic':3,'Other':4,'White':5}).astype(np.uint8)



