# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 23:08:48 2017

@author: pedro
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as plt

ds = SupervisedDataSet(2, 1) #Base de dados com 2 parametros e 1 saida

ds.addSample((0.8,0.4), (0.7)) #Estudou 8 horas e dormiu 4 horas, tirou 7
ds.addSample((0.5,0.7), (0.5)) #Estudou 5 horas e dormiu 7 horas, tirou 5
ds.addSample((1.0,0.8), (0.95)) #Estudou 10 horas e dormiu 8 horas, tirou 9.5

nn = buildNetwork(2,4,1, bias=True) #2 entradas, 4 ocultos e um saida

trainer = BackpropTrainer(nn,ds)

for i in xrange(100):
    print (trainer.train())
erros = []   
while True:
    #dormiu = float(raw_input('Tempo que dormiu: '))
    #estudou = float(raw_input('Tempo que estudou: '))
    dormiu = input('Tempo que dormiu: ')
    estudou = input('Tempo que estudou: ')
    
    resultado = nn.activate((dormiu * 0.1, estudou * 0.1))
    erros.append(resultado)
    
    plt.plot(erros)
    
    print ('Previsao de nota: ', resultado * 10)