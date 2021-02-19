import matplotlib.pyplot as plt
import numpy as np
import math

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer    
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.validation import ModuleValidator

if __name__ == '__main__':

    # Definicao da classe de rede neural
    dimensaoDaEntrada=1
    dimensaoDaCamadaEscondida=5
    dimensaoDaSaida=1

    rn=buildNetwork(dimensaoDaEntrada,dimensaoDaCamadaEscondida,dimensaoDaSaida,bias=True,hiddenclass=TanhLayer)


    #Criacao dos dados
    tamanhoDaAmostra=100
    dados = SupervisedDataSet(dimensaoDaEntrada,dimensaoDaSaida)    

    comRuido=True

    for i in range(tamanhoDaAmostra):
        if(comRuido):
            x=np.random.uniform(0,2*math.pi,1)
            dados.addSample((x), (math.sin(x)+ np.random.normal(0, 0.1,1),))
        else:    
            x=np.random.uniform(0,2*math.pi,1)
            dados.addSample((x), (math.sin(x),))

    treinadorSupervisionado = BackpropTrainer(rn, dados)

    numeroDeAcessos=10
    numeroDeEpocasPorAcesso=50


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)  
    ax1.axis([0, 2*math.pi, -1.5, 1.5]) 
    fig1.hold()

    fig2=plt.figure()
    ax2 = fig2.add_subplot(111)      
    ax2.axis([-50, numeroDeAcessos*numeroDeEpocasPorAcesso+50, 0.00001, 4])
    ax2.set_yscale('log')
    fig2.hold()    
    meansq = ModuleValidator() 
    erro2=meansq.MSE(treinadorSupervisionado.module,dados)
    print erro2
    ax2.plot([0],[erro2],'bo')

    tempoPausa=1
    for i in range(numeroDeAcessos):
        treinadorSupervisionado.trainEpochs(numeroDeEpocasPorAcesso)
        meansq = ModuleValidator() 
        erro2=meansq.MSE(treinadorSupervisionado.module,dados)
        print erro2
        ax1.plot(dados['input'],dados['target'],'bo',markersize=7, markeredgewidth=0)
        ax1.plot(dados['input'],np.array([rn.activate(datax) for datax, _ in dados]),'ro',markersize=7, markeredgewidth=0)
        ax2.plot([numeroDeEpocasPorAcesso*(i+1)],[erro2],'bo')
        plt.pause(tempoPausa) 
        ax1.plot(dados['input'],np.array([rn.activate(datax) for datax, _ in dados]),'wo',markersize=9, markeredgewidth=0)
