# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:28:57 2017

@author: Pedro
"""

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
import matplotlib.pyplot as plt


dataset = SupervisedDataSet(5,1)

"""
2,58	200	   21	4	500	 2693,6
3,18	230 	16	3	600	 3492,8
1,65	211 	18	5	900	 2094,5
1,99	158	   19	6	380 1565,3
"""

dataset.addSample([0.000258,0.2,0.0021,0.0004,0.5],[2.694])
dataset.addSample([0.000318,0.23,0.0016,0.0003,0.6],[3.493])
dataset.addSample([0.000165,0.211,0.0018,0.0005,0.9],[2.095])
dataset.addSample([0.000199,0.158,0.0019,0.0006,0.38],[1.565])


network = buildNetwork(dataset.indim, 5, dataset.outdim, bias=False)


trainer = BackpropTrainer(network, dataset, learningrate=0.001, momentum=0.99)

erros = []
for epoch in range(0, 300):
    erros.append(trainer.train())
    print erros[-1]
    
plt.xlabel("Epocas")
plt.ylabel("Erros")
plt.plot(erros)


