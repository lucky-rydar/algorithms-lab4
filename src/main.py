import numpy as np
import operator

from numpy.random.mtrand import random
import math
import random
import matplotlib.pyplot as plt
import sys

class Graph:
    def __init__(self, graph_struct = {}):
        self.graph = graph_struct

    def __str__(self):
        grh = ''
        for vrt in self.getVertices():
            for adj in self.getAdjacent(vrt):
                grh += '({0}, {1}, {2})\t'.format(vrt, adj, self.graph[vrt][adj])
        return grh

    def setVertex(self, vertex):
        if vertex not in self.graph.keys():
            self.graph[vertex] = {}
        return self

    def setAdjacent(self, vertex, adj, weight=0):
        if vertex not in self.graph.keys():
            self.graph[vertex] = {}
        if adj not in self.graph.keys():
            self.graph[adj] = {}
        
        self.graph[vertex][adj] = weight
        #self.graph[adj][vertex] = weight
        return self
    
    def getVertices(self):
        return list(self.graph.keys())

    def getAdjacent(self, vertex):
        if vertex in self.graph.keys():
            return self.graph[vertex]

    def getPathCost(self, path):
        # path is: ['1', '123', '299']
        pathCost = 0
        for i in range(len(path)-1):
            vrt = path[i]
            adj = path[i+1]
            pathCost += self.graph[vrt][adj]
        return pathCost


class GeneticAlgorithmTSP:
    def __init__(self, generations=100, population_size=10, tournamentSize=4, mutationRate=0.1, elitismRate=0.1):
        self.generations = generations
        self.population_size = population_size
        self.tournamentSize = tournamentSize
        self.mutationRate = mutationRate
        self.elitismRate = elitismRate
        
        self.way_data_plot = []
        self.best_route = []
        best_distane = 0
    
    def optimize(self, graph):
        population = self.__makePopulation(graph.getVertices())
        elitismOffset = math.ceil(self.population_size*self.elitismRate)

        if (elitismOffset > self.population_size):
            raise ValueError('Elitism Rate must be in [0,1].')
        
        for generation in range(self.generations):
            print(self.generations)

            print ('\nGeneration: {0}'.format(generation + 1))
            #print ('Population: {0}'.format(population))
            
            newPopulation = []
            fitness = self.__computeFitness(graph, population)
            print ('Fitness:    {0}'.format(fitness))
            fittest = np.argmin(fitness)

            self.best_route = population[fittest]
            self.best_distane = fitness[fittest]
            self.way_data_plot.append(fitness[fittest])
            
            if elitismOffset:
                elites = np.array(fitness).argsort()[:elitismOffset]
                [newPopulation.append(population[i]) for i in elites]
            for gen in range(elitismOffset, self.population_size):
                parent1 = self.__tournamentSelection(graph, population)
                # do tournament selection while it is the same as parent1
                parent2 = self.__tournamentSelection(graph, population)
                #while parent2 == parent1:
                #    parent2 = self.__tournamentSelection(graph, population)
                
                offspring = self.__crossover(parent1, parent2)
                newPopulation.append(offspring)

            for gen in range(elitismOffset, self.population_size):
                newPopulation[gen] = self.__mutate(newPopulation[gen])
    
            population = newPopulation

            #if self.__converged(population):
            #    print ('\nConverged to a local minima.', end='')
            #    break

        return (population[fittest], fitness[fittest])


    def __makePopulation(self, graph_nodes):
        # return graph_nodes
        ret = []
        for i in range(self.population_size):
            vs = np.random.permutation(graph_nodes)
            ret.append(list(vs))
        return ret

        # instead of that shit it should return arrays
        return [''.join(v for v in np.random.permutation(graph_nodes)) for i in range(self.population_size)]
    

    def __computeFitness(self, graph, population):
        ret = []

        for path in population:
            ret.append(graph.getPathCost(path))
        return ret


    def __tournamentSelection(self, graph, population):
        # generate random unique indexes
        indexes = random.sample(range(self.population_size), self.tournamentSize)
        tournament_contestants = []
        for i in indexes:
            tournament_contestants.append(population[i])

        tournament_contestants_fitness = self.__computeFitness(graph, tournament_contestants)

        return tournament_contestants[np.argmin(tournament_contestants_fitness)]
    

    def __crossover(self, parent1, parent2):
        low, high = self.__computeLowHighIndexes(parent1)
        to_inherit = [parent1[i] for i in range(low, high)]
        offspring = parent2[:]
        for item in to_inherit:
            p1_idx = parent1.index(item)
            p2_idx = offspring.index(item)
            offspring[p1_idx], offspring[p2_idx] = offspring[p2_idx], offspring[p1_idx]
        return offspring


    def __mutate(self, genome):
        if np.random.random() < self.mutationRate:
            index_low, index_high = self.__computeLowHighIndexes(genome)
            return self.__swap(index_low, index_high, genome)
        else:
            return genome


    def __computeLowHighIndexes(self, arr):
        index_low = np.random.randint(0, len(arr)-1)
        index_high = np.random.randint(index_low+1, len(arr))
        return (index_low, index_high)


    def __swap(self, index_low, index_high, arr):
        arr[index_low], arr[index_high] = arr[index_high], arr[index_low]
        return arr


def generate_graph(how_much):
    nodes = [str(i) for i in range(how_much)]

    g = Graph()
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if(j != i):
                g.setAdjacent(nodes[i], nodes[j], random.randint(5, 150))

    return g


def main():
    graph = generate_graph(300)

    ga_tsp = GeneticAlgorithmTSP(generations=10000,
                        population_size=10,
                        tournamentSize=4,
                        mutationRate=0.1,
                        elitismRate=0.2)
    
    optimal_path, path_cost = ga_tsp.optimize(graph)
    print ('\nPath: {0}, Cost: {1}'.format(optimal_path, path_cost))

    plt.plot(ga_tsp.way_data_plot)
    plt.show()


if __name__ == '__main__':
    main()


