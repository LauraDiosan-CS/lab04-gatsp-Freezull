import os
import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings

# read the network details
from GA import GA
from utils import get_dist


def transformFile(filepath):

    import numpy as np
    import networkx as nx
    gmlGraph = nx.read_gml(filepath, label='id')
    adjMatrix = nx.to_numpy_matrix(gmlGraph, dtype=int)
    with open('ceva.in', 'w') as f:
        f.write(str(gmlGraph.number_of_nodes()) + '\n')
        for line in adjMatrix:
            np.savetxt(f, line, fmt='%d')


def readDataGML():
    matrix = []
    for i in range(62):
        matrix.append([])
        for j in range(62):
            matrix[i].append(0)

    with open("ceva.in") as ceva:
        lines = ceva.readlines()
        source = 0
        destination = 0
        for line in lines:
            line = line.split(" ")
            if len(line) >= 5:
                if line[4] == "source":
                    source = int(line[5])
                else:
                    destination = int(line[5])
                    matrix[source][destination] = 1
                    matrix[destination][source] = 1

        net = {}
        n = 62
        net['noNodes'] = n
        net["mat"] = matrix
        degrees = []
        noEdges = 0
        for i in range(n):
            d = 0
            for j in range(n):
                if (matrix[i][j] == 1):
                    d += 1
                if (j > i):
                    noEdges += matrix[i][j]
            degrees.append(d)
        net["noEdges"] = noEdges
        net["degrees"] = degrees
        return net


def readNet(fileName, sep):
    f = open(fileName, "r")
    net = {}
    n = int(f.readline())
    net['noNodes'] = n
    mat = []
    for i in range(n):
        mat.append([])
        line = f.readline()
        elems = line.split(sep)
        for j in range(len(elems)):
            if elems[j] != "":
                mat[-1].append(int(elems[j]))
    net["mat"] = mat
    degrees = []
    noEdges = 0
    for i in range(n):
        d = 0
        for j in range(n):
            if (mat[i][j] == 1):
                d += 1
            if (j > i):
                noEdges += 1
        degrees.append(d)
    net["noEdges"] = noEdges
    net["degrees"] = degrees
    f.close()
    return net


def citire_fisier(filename):
    with open(filename) as fp:
        lines = fp.readlines()
        lines = lines[3:]
        n = int(lines[0].split(" ")[2])
        lines = lines[3:-1]

        net = {}
        net['noNodes'] = n

        mat = []

        for i in range(n):
            mat.append([])
            for _ in range(n):
                mat[i].append(0)

        pos = []
        for line in lines:
            vect = line.split(" ")
            pos.append([int(vect[0]), int(vect[1]), int(vect[2])])

        for i in range(n):
            for j in range(n):
                mat[pos[i][0]-1][pos[j][0]-1] = int(get_dist([pos[i][1], pos[i][2]], [pos[j][1], pos[j][2]]))
        net["mat"] = mat
        degrees = []
        noEdges = 0
        for i in range(n):
            d = 0
            for j in range(n):
                if (mat[i][j] == 1):
                    d += 1
                if (j > i):
                    noEdges += 1
            degrees.append(d)
        net["noEdges"] = noEdges
        net["degrees"] = degrees
        return net



def modularity(communities, param):
    noNodes = param['noNodes']
    mat = param['mat']
    degrees = param['degrees']
    noEdges = param['noEdges']
    S = mat[communities[0]][communities[len(communities)-1]]
    for i in range(0, len(communities)-1):
        S += mat[communities[i]][communities[i+1]]
    return S


# load a tetwork
crtDir = os.getcwd()
filePath = os.path.join(crtDir, 'ceva.in')


filepath_easy = "input/easy_01_tsp.txt"
filepath_medium = "input/medium_01_tsp.txt"
filepath_hard = "input/eil51.tsp"


middle_file = "ceva.in"

network = readNet(filepath_easy, ",")
#network = readNet(filepath_medium, " ")
#network = citire_fisier(filepath_hard)


problParam = network
gaParam = {'popSize': problParam["noNodes"], 'noGen': 10000, 'pc': 0.8, 'pm': 0.1}
# problem parameters

problParam["function"] = modularity
problParam["min"] = 1
problParam["max"] = problParam["noNodes"] + 1
# problParam["max"] = len(set(problParam["degrees"]))+1

# print(network)

# modify best
def get_best_repres(repres):
    a = repres
    v = set(a.copy())

    lista = []
    for i in range(len(v)):
        lista.append((v.pop(), i + 1))

    dic = dict(lista)

    return [dic.get(n, n) for n in a]


def main(*init):
    to_list = []
    ga = GA(gaParam, problParam)
    ga.initialisation()

    best = ga.bestChromosome()
    best_all = []

    for i in range(gaParam["noGen"]):
        ga.oneGenerationElitism()

        best = ga.bestChromosome()
        if (i+1) % int(gaParam["noGen"]/10) == 0:
            to_list.append([i+1, best.fitness, best.repres])
            #print(len(set(best.repres)), best.fitness)
        best_all.append(best)

    #print(len(set(best.repres)), best)
    return best_all, to_list

start_time = time.time()
it1, list = main()
print(str(int(time.time() - start_time))+"s")

no_comm = len(set(it1[-1].repres))
final_repres = get_best_repres(it1[-1].repres)
print(str(no_comm), final_repres, sep="\n")

for pair in list:
    print("generatia : " + str(pair[0]) + ", fitness : " + str(pair[1]) + ", comunitati : " + str(pair[2]))


'''
#
while gaParam["popSize"] >= 2:
    it1.sort(key=lambda x:x.fitness)
    gaParam["popSize"] = int(gaParam["popSize"] / 2)
    it1 = main(it1)
'''

#txt
vect = []
for i in range(len(final_repres)):
    vect.append([i + 1, final_repres[i]])

with open("output.txt", "w") as out:
    for pair in vect:
        out.writelines(" " + str(pair[0]) + " " + str(pair[1])+"\n")


