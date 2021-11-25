import numpy as np
import operator
from numpy.core.numeric import cross

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
    def __init__(self, generations=100, population_size=10, tournamentSize=4, mutationRate=0.1, elitismRate=0.1, crossover_type = 'two_points'):
        self.generations = generations
        self.population_size = population_size
        self.tournamentSize = tournamentSize
        self.mutationRate = mutationRate
        self.elitismRate = elitismRate
        self.crossover_type = crossover_type
        
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
                offspring = []
                if(self.crossover_type == 'two_points'):
                    offspring = self.__crossover_two_points(parent1, parent2)
                elif(self.crossover_type == 'one_point'):
                    offspring = self.__crossover_one_point(parent1, parent2)
                elif(self.crossover_type == 'uniform'):
                    offspring = self.__crossover_uniform(parent1, parent2)

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
    

    def __crossover_two_points(self, parent1, parent2):
        low, high = self.__computeLowHighIndexes(parent1)
        to_inherit = [parent1[i] for i in range(low, high)]
        offspring = parent2[:]
        for item in to_inherit:
            p1_idx = parent1.index(item)
            p2_idx = offspring.index(item)
            offspring[p1_idx], offspring[p2_idx] = offspring[p2_idx], offspring[p1_idx]
        return offspring


    def __crossover_one_point(self, parent1, parent2):
        # it is  the same as upper but generate one random number
        # and then get some part from first parent and another from the second
        
        point = np.random.randint(0, len(parent1))
        # everything before point  and point is from parent1
        # everything after  point is from parent2

        offspring = parent2[:]
        to_inherit = [parent1[i] for i in range(0, point)]
        for item in to_inherit:
            p1_idx = parent1.index(item)
            p2_idx = offspring.index(item)
            offspring[p1_idx], offspring[p2_idx] = offspring[p2_idx], offspring[p1_idx]
        return offspring


    def __crossover_uniform(self, parent1, parent2):
        # here just generate array with 0 and 1
        # and then take some gens from 1 parent and some from 2 perent
        # 
        # in my case it is needed to swap gens where 1 inside of parent 2
        
        points = [np.random.randint(0, 2) for _ in range(len(parent1))]

        to_inherit = []
        offspring = parent2[:]
        for i in range(len(points)):
            if points[i] == 1:
                to_inherit.append(parent1[i])

        for item in to_inherit:
            p1_idx = parent1.index(item)
            p2_idx = offspring.index(item)
            offspring[p1_idx], offspring[p2_idx] = offspring[p2_idx], offspring[p1_idx]
        return offspring


    def __mutate(self, genome):
        if self.mutationRate > 1:
            for i in range(int(self.mutationRate)):
                if np.random.random() < self.mutationRate:
                    index_low, index_high = self.__computeLowHighIndexes(genome)
                    return self.__swap(index_low, index_high, genome)
                else:
                    return genome
        else:
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

    gs = [graph, graph, graph]
    crossovers = ['two_points', 'one_point', 'uniform']
    shortest_ways = []

    results_plot = []

    for i in range(len(gs)):
        ga_tsp = GeneticAlgorithmTSP(generations=1000,
                        population_size=20,
                        tournamentSize=4,
                        mutationRate=10,
                        elitismRate=0.1,
                        crossover_type=crossovers[i])
    
        optimal_path, path_cost = ga_tsp.optimize(gs[i])
        shortest_ways.append((optimal_path, path_cost))
        results_plot.append(ga_tsp.way_data_plot)

    print('\n'*2)
    for i in range(len(shortest_ways)):
        print ('Crossover type: {0}, Cost: {1}\n'.format(crossovers[i], shortest_ways[i][1]))
    
    fig, axs = plt.subplots(3, 1, figsize=(5, 15))
    for i in range(len(gs)):
        axs[i].plot(results_plot[i])
        axs[i].set_title(crossovers[i])


    plt.show()


if __name__ == '__main__':
    main()


