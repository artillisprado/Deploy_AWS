import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from dataframe import Dados
#biblioteca que auxilia padronizar os dados para aplicar a regressão logística
import sklearn.preprocessing as sk

#Criar x0's com valores iguais a 1 (?)
def insert_ones(X):
    ones = np.ones([X.shape[0], 1])
    return np.concatenate((ones, X), axis=1)

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def binary_cross_entropy(w, X, y):
    m = len(X)
    parte1 = np.multiply(-y, np.log(sigmoid(X @ w.T)))
    parte2 = np.multiply((1 - y), np.log(1 - sigmoid(X @ w.T)))
    somatorio = np.sum(parte1 - parte2)
    return somatorio / m

def gradient_descent(w, X, y, alpha, epoch):
    const = np.zeros(epoch)
    for i in range(epoch):
        w = w - (alpha/len(X)) * np.sum((sigmoid(X@w.T) - y) * X, axis=0)
        const[i] = binary_cross_entropy(w,X,y)
    return w, const

def predict(w, X, threshold = 0.5):
    p = sigmoid(X @ w.T) >= threshold
    return (p.astype('int'))




    

df = Dados.dataframe_somente_com_colunas_numericas
#pegar o número da coluna booleana, no nosso caso, SkinCancer
features = len(df.columns) - 1
# training_data = os dados em formato de lista sem a coluna passada no parâmetro de df.drop()
training_data = np.array(df.drop('SkinCancer', axis=1))
# target_variable = pegando os dados da coluna target em forma de lista também
target_variable = df.iloc[:,features:features + 1 ].values
#precisamos somente de valores número em todas as colunas, fazer esse tratamento da média
media_valores = training_data.mean(axis=0)
desvio_padrao = training_data.std(axis=0)
#padronizar os dados para aplicação da regressão logística
scaler = sk.StandardScaler()
scaler.fit(training_data)
training_data = scaler.transform(training_data)
#criando uma matriz do tamanho de uma linha da tabela com valores aleatórios entre 0 e 1
linha_w = np.random.rand(1, features+1)
#CRIANDO VALORES EM -10 E 10 E APLICAR A SIGMOID PARA VISUALIZAÇÃO DO GRÁFICO
valores = np.arange(-10,10,step=1)
fix, ax = plt.subplots(figsize=(6,4))
ax.plot(valores, sigmoid(valores), 'r')
# plt.show()

#INICIANDO
training_data = insert_ones(training_data)
#alpha=taxa de aprendizado
alpha = 0.01
#epoch = quantidade de repetições
epoch = 10000
w, cost = gradient_descent(linha_w, training_data, target_variable, alpha, epoch)

# plotar a queda do custo
fix, ax = plt.subplots()
ax.plot(np.arange(epoch), cost, 'r')
ax.set_xlabel('Iterações')
ax.set_ylabel('Custo')
ax.set_title('Erro vs Epoch')
plt.show()

# print(predict(linha_w, training_data[0]))










