from random import uniform
import math
import pandas as pd
import numpy as np
from numpy import linalg

####################### Funções ##############################################
def initialize_network(D, nh, c):
    Wh = np.zeros((D+1,nh+1)) # +1 do bias
    for j in range(D+1): # +1 do bias
        for i in range(nh+1): # +1 do bias
            weight = uniform(-1/math.sqrt(D),1/math.sqrt(D)) # pesos iniciais com distribuição uniforme
            Wh[j][i] = weight
            
    Ws = np.zeros((nh+1,c)) # +1 do bias
    for j in range(nh+1): # +1 do bias
        for i in range(c):
            weight = uniform(-1/math.sqrt(D),1/math.sqrt(D)) # pesos iniciais com distribuição uniforme
            Ws[j][i] = weight
    return Wh,Ws

# Função de ativação
def phi(activation):
	a = 1.716 # constante de nao linearidade
	b = 2/3 # constante de nao linearidade
	return a*math.tanh(b*activation)

# Derivada da função de ativação
def phi_derivative(output):
	a = 1.716 # constante de nao linearidade
	b = 2/3 # constante de nao linearidade
	return a*b*(1 - (math.tanh(b*output))**2)

def forward_propagate(xk,Wh,Ws):
    # FORWARD PROPAGATION
    vj = np.dot(xk,Wh) # soma das entradas multiplicadas pelos pesos da camada escondida para cada neuronio (resulta em um vetor 1x2)
    Y = np.zeros(len(vj)) 
    for i in range(len(vj)): # aplica a funcao phi para os neuronios
        Y[i] = phi(vj[i])
    vk = np.dot(Y,Ws)
    Z = np.zeros(len(vk))
    for j in range(len(vk)): # aplica a funcao phi para os neuronios
        Z[j] = phi(vk[j])

    return Y,Z

def backward_propagate(e,Z,Y,Ws,D,nh,eta,xk,Wh,c):
    # BACKPROPAGATION
    phi_linhak = np.zeros(len(Z))
    for k in range(len(Z)): # aplica a derivada de phi
        phi_linhak[k] = phi_derivative(Z[k])

    # eq. 4.46 do livro
    deltak = np.multiply(e,phi_linhak)

    phi_linhaj = np.zeros(len(Y))
    for l in range(len(Y)): # aplica a derivada de phi
        phi_linhaj[l] = phi_derivative(Y[l])

    # eq. 4.46 do livro
    deltaj = np.multiply(phi_linhaj,np.dot(deltak,Ws.T)) # delta para os neuronios da camada escondida

    deltawh_vec = list()
    # eq. 4.47 do livro - atualizacao de pesos da camada escondida
    for i in range(D+1): # +1 do bias
        for j in range(nh+1): # +1 do bias
            deltawh = eta*deltaj[j]*xk[i]
            Wh[i][j] = Wh[i][j] + deltawh # atualização dos pesos
            deltawh_vec.append(deltawh)

    deltaws_vec = list()
    # eq. 4.47 do livro - atualizacao de pesos da camada de saida
    for i in range(nh+1): # +1 do bias
        for j in range(c):
            deltaws = eta*deltak[j]*Y[i]
            Ws[i][j] = Ws[i][j] + deltaws # atualização dos pesos
            deltaws_vec.append(deltaws)
            
    return Wh,Ws

####################### Script principal ##############################################
x = pd.read_csv('Anexo 1 da Lista 4 - treina.txt',sep='        ',header=None, engine='python')
N = len(x)
x.columns = ['x1','x2','x3','x4','classe']
expected = x['classe']
x = x.drop(columns=['classe'])

# normalizacao dos dados de entrada
x = (x - x.mean())/x.std()
x.insert(0,'b',1) # adicionando +1 como entrada do bias

d = list()

# converte o rotulo da saida para um vetor numerico com -1 ou +1
for row in expected:
    cod = [-1,-1,-1]
    cod[int(row)-1] = 1
    d.append(cod)
d = np.array(d)
x = x.to_numpy()

D = 4 # dimensão do espaço de entrada
nh = 2 # numero de neuronios na camada escondida
c = 3 # numero de classes
eta = 0.05 # taxa de aprendizagem
nmax = 200 # numero maximo de epocas
Wh,Ws = initialize_network(D, nh, c)

# TREINAMENTO AMOSTRA POR AMOSTRA
n = 1
theta = 0
while n < nmax: # limita o número de épocas
    err_vec = list()
    if theta > 0.1 or n == 1: # finaliza o treino caso o critério seja atingido
        for idx, xk in enumerate(x):
            Y,Z = forward_propagate(xk,Wh,Ws)
            # eq. 4.45 do livro
            e = d[idx]-Z
            Wh,Ws = backward_propagate(e,Z,Y,Ws,D,nh,eta,xk,Wh,c) 
            err_vec.append([i*i for i in e]) # erro quadrático
        err_vec = np.array(err_vec)
        avg_err = np.mean(err_vec, axis=0) # MSE
        theta = linalg.norm(avg_err) # Norma
        # print(f'epoca: {n}, theta: {theta}\n')
    else:
        n = nmax
    n += 1

# PREDIÇÃO
pred = list()
x = pd.read_csv('Anexo 2 da Lista 4 - teste.txt',sep='	',header=None, engine='python')
x = (x - x.mean())/x.std()
x.insert(0,'b',1) # adicionando +1 como entrada do bias
x = x.to_numpy()

for xk in x:
    Y,Z = forward_propagate(xk,Wh,Ws)
    index = np.where(Z == max(Z))[0][0] # index do maior elemento do vetor
    Ck = index + 1 # classe
    pred.append(Ck)