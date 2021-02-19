# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 10:41:47 2017

@author: pedro
"""

# aplicativo para verificar se o ser vivo eh quadrupede ou bipede
# quadrupede = 1, bipede = -1
# cao = [-1,-1,1,1] | resposta = 1
# gato = [1,1,1,1] | resposta = 1
# cavalo = [1,1,-1,1] | resposta = 1
# homem = [-1,-1,-1,1] | resposta = -1

# pesos (sinapses)
w = [0,0,0,0]
# entradas
x = [[-1,-1,1,1],
     [1,1,1,1],
     [1,1,-1,1],
     [-1,-1,-1,1]]
# respostas esperadas
t = [1,1,1,-1]
# bias (ajuste fino)
b = 0
#saida
y = 0
# numero maximo de interacoes
max_int = 10
# taxa de aprendizado
taxa_aprendizado = 1
#soma
soma = 0
#theshold
threshold = 1
# nome do animal
animal = ""
# resposta = acerto ou falha
resposta = ""

# dicionario de dados
d = {'-1,-1,1,1' : 'cao',
     '1,1,1,1' : 'gato',
     '1,1,-1,1' : 'cavalo',
     '-1,-1,-1,1' : 'homem' }

print("Treinando")

# funcao para converter listas em strings
def listToString(list):
    s = str(list).strip('[]')
    s = s.replace(' ', '')
    return s

# inicio do algoritmo
for k in range(1,max_int):
    acertos = 0    
    print("INTERACAO "+str(k)+"-------------------------")
    for i in range(0,len(x)):
        soma = 0

        # pega o nome do animal no dicionário
        if d.has_key(listToString(x[i])):
            animal = d[listToString(x[i])]  
        else:
            animal = ""

        # para calcular a saida do perceptron, cada entrada de x eh multiplicada
        # pelo seu peso w correspondente
        for j in range(0,len(x[i])):
            soma += x[i][j] * w[j]

        # a saida eh igual a adicao do bias com a soma anterior
        y_in = b + soma
        #print("y_in = ",str(y_in))

        # funcao de saida eh determinada pelo threshold
        if y_in > threshold:
            y = 1
        elif y_in >= -threshold and y_in <= threshold:
            y = 0
        else:
            y = -1        

        # atualiza os pesos caso a saida nao corresponda ao valor esperado
        if y == t[i]:
            acertos+=1
            resposta = "acerto"
        else:
            for j in range (0,len(w)):                
                w[j] = w[j] + (taxa_aprendizado * t[i] * x[i][j])
            b = b + taxa_aprendizado * t[i]
            resposta = "Falha - Peso atualizado"

        #imprime a resposta
        if y == 1:
            print(animal+" = quadrupede = "+resposta)
        elif y == 0:
            print(animal+" = padrao nao identificado = "+resposta)
        elif y == -1:
            print(animal+" = bipede = "+resposta)

    if acertos == len(x):
        print("Funcionalidade aprendida com "+str(k)+" interacoes")
        break;
    print("")
print("Finalizado")