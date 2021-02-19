# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 23:08:48 2017

@author: pedro
"""
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
#import matplotlib.pyplot as plt

arquivo = open('dadosteste.txt','r')
texto = arquivo.readlines()
separado = []
for linha in texto:
    separado.append(linha.split())

ds = SupervisedDataSet(2, 1)

for i in range(len(separado)):   
    a = float(separado[i][0])
    b = float(separado[i][1])
    c = float(separado[i][2])
    ds.addSample((a,b),(c))

nn = buildNetwork(2,4,1, bias=True) #2 entradas, 4 ocultos e um saida

trainer = BackpropTrainer(nn,ds)
curva_treinamento = []
for i in xrange(2000):
    curva_treinamento.append(trainer.train())
    print 'Epoca: ',i,'Erro: ',curva_treinamento[i]
#plt.plot(curva_treinamento)    
    
while True:
    #dormiu = float(raw_input('Tempo que dormiu: '))
    #estudou = float(raw_input('Tempo que estudou: '))
    dormiu = input('Tempo que dormiu: ')
    estudou = input('Tempo que estudou: ')
    
    resultado = nn.activate((dormiu * 0.1, estudou * 0.1))
    #resultado = nn.activate((dormiu, estudou))[0] * 10
    
    print 'Previsao de nota: ', str(resultado*10)