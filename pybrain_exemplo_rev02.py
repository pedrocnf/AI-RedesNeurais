# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 23:08:48 2017

@author: pedro
"""
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
#import matplotlib.pyplot as plt

arquivo = open('dados_granito.txt','r')
texto = arquivo.readlines()
separado = []
for linha in texto:
    separado.append(linha.split())

ds = SupervisedDataSet(18, 1)

for i in range(len(separado)):   
    a = float(separado[i][0])
    b = float(separado[i][1])
    c = float(separado[i][2])
    d = float(separado[i][3])
    e = float(separado[i][4])
    f = float(separado[i][5])
    g = float(separado[i][6])
    h = float(separado[i][7])
    i = float(separado[i][8])
    j = float(separado[i][9])
    k = float(separado[i][10])
    l = float(separado[i][11])
   # m = float(separado[i][12])
    #n = float(separado[i][13])
    #o = float(separado[i][14])
    #p = float(separado[i][15])
    #r = float(separado[i][16])
    s = float(separado[i][12])
     
    
    
    
    
    
    
    ds.addSample((a,b,c,d,e,f,g,h,i,j,k,l),(s))

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