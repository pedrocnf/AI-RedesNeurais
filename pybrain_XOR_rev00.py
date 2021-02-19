# encoding:utf-8

# PyBrain - XOR
# Documentação: http://pybrain.org/docs/index.html

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

# cria-se um conjunto de dados (dataset) para treinamento
# são passadas as dimensões dos vetores de entrada e do objetivo
dataset = SupervisedDataSet(2,1)

# adiciona-se as amostras que consiste numa entrada e num objetivo
# como vamos resolver o XOR, inserimos como na tabela-verdade
dataset.addSample([1,1],[0])
dataset.addSample([1,0],[1])
dataset.addSample([0,1],[1])
dataset.addSample([0,0],[0])

'''
Agora iremos construir a rede utilizando a função buildNetwork
dataset.indim é o tamanho da camada de entrada
4 é a quantidade de camadas intermediárias
dataset.outdim é o tamanho da camada de saída
iremos utilizar o "bias" para permitir uma melhor adaptação por parte da rede neural
ao conhecimento à ela fornecido
'''
network = buildNetwork(dataset.indim, 4, dataset.outdim, bias=True)

'''
O procedimento que iremos utilizar para treinar a rede é o backpropagation.
É pasada a rede, o conjunto de dados (dataset), "learningrate" é a taxa de aprendizado,
"momentum" tem por objetivo aumentar a velocidade de treinamento da rede neural e
diminuir o perigo da instabilidade.
'''
trainer = BackpropTrainer(network, dataset, learningrate=0.01, momentum=0.99)

# Logo em seguinda é feito de fato o treinamento da rede
for epoch in range(0, 1000): # treina por 1000 épocas
    trainer.train()

'''
Outras formas de treinar:
    trainer.trainEpochs(1000)
    treinar até a convergência: trainer.trainUntilConvergence()
'''

# Agora iremos testar a rede com um conjunto de dados
test_data = SupervisedDataSet(2,1)
test_data.addSample([1,1],[0])
test_data.addSample([1,0],[1])
test_data.addSample([0,1],[1])
test_data.addSample([0,0],[0])
# verbose=True indica que deve ser impressas mensagens
trainer.testOnData(test_data, verbose=True)
