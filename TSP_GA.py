""" This is a simple genetic algorithm implementation to solve the TSP.
    It was made as a small research project for one of the classes I took.
    The goal was to test the performance of the combinations between order crossover and
    partially-mapped crossover, for the crossover operators, and the tournament and 
    roulette wheel selection operators.

	Widerlani Campos, 2017 """

import numpy as np
import networkx as nx
import random
import heapq as hq

# global variables
A = []
G = []

class Tour:
	""" Class that represents a tour.
		.cities: list of cities that represent the tour;
		.length: tour length.

		Operators > and < redefined to compare the length between Tours.
	"""
	def __init__(self, cities, length):
		self.cities = cities
		self.length = length

	def updateLength(self):
		self.length = tourLength(self.cities)
	
	def __gt__(self, other):
		return self.length > other.length

	def __lt__(self, other):
		return self.length < other.length

class Population:
	""" Represents a population.
		.tours: actual population. List of tours;
		.popSize: number of Tours in the population.
	"""

	# randomize is whether or not you want to generate a random population
	def __init__(self, popSize, randomize):
		self.tours = []
		self.popSize = popSize

		if randomize:
			self.tours = genRandomTours(popSize)

	def getShortest(self):
		return min(self.tours)

def tourLength(cities):
	""" Given a list of cities, returns the sum of the edge weights between
		these cities, aka the length of a tour.
	"""
	length = 0
	# sums the distance up to the last city of the tour
	for i in range(len(cities) - 1):
		cityA = cities[i]
		cityB = cities[i+1]
		length = length + G[cityA][cityB]['weight']
	# now sums the distance from the last city to the first
	cityA = cities[len(cities) - 1]
	cityB = cities[0]
	length = length + G[cityA][cityB]['weight']

	return length

def genRandomTour():
	""" generates and returns a random Tour """
	# n is the number of cities in the TSP
	n = nx.number_of_nodes(G)
	# generates a random list of the cities
	cities = random.sample(range(n), n)
	length = tourLength(cities)
	return Tour(cities, length)

def genRandomTours(n):
	""" generates and returns a list of n random Tours """
	tours = []

	for i in range(n):
		tour = genRandomTour()
		tours.append(tour)

	return tours

def tournamentSelection(pop, tournamentSize):
	"""
		Selects individuals from a population using tournament selection.
		Assumes tournamentSize >= 2.
		Returns the selected Tour.
	"""

	# tournament is a list with at least 2 elements
	tournament = random.sample(pop.tours, tournamentSize)

	# returns the greatest tour from those in the tournament
	return hq.heappop(tournament)

def rouletteWheelSelection(pop, sumLengths):
	"""
		Selects individuals from a population using proportional roulette wheel
		selection.
		Returns the selected Tour.

		Since I'm working with tourLength and not fitness, the lowest length
		represents the best individual. To make the simple roulette selection
		work, I'm using the opposite of the lengths and the sum of lenghts.
	"""

	negSumLengths = -sumLengths

	# r gets a random number in the interval [negSumLengths, 0]
	r = random.uniform(negSumLengths, 0)

	# when the "negative sum" is lesser than or equal to the random number r
	# that tour is selected
	negSum = 0
	for tour in pop.tours:
		negSum -= tour.length
		if negSum <= r:
			return tour

def orderCrossover(parent1, parent2):
	"""
		Performs order crossover with parent1 and parent2 and returns the
		resulting child.

		Parents are assumed to be the same size.

		The child gets the region between the cutpoints (including the position
		of both cutpoints) from parent1 and	the rest from parent2, starting from
		after the second cutpoint.
	"""
	# tour size
	size = len(parent1.cities)

	# initialize the child as an empty list with the same size as the parents
	childCities = [None] * size

	# a and b represent the first and second cutpoints, respectively
	a = random.randint(0, size - 1)
	b = random.randint(0, size - 1)

	# Check if the cutpoints are valid, that is, a must be different than b,
	# and b must be greater than a.
	# If b is not greater than a, but they're different, simply swap them.
	validCutPoints = a < b
	while not validCutPoints:
		if a == b:
			a = random.randint(0, size - 1)
			b = random.randint(0, size - 1)
		else: # a != b, but a > b, so swap(a,b)
			a,b = b,a
		validCutPoints = a < b

	# DEBUG
	# print("childSize = ", len(childCities), ", parent1 size = ", len(parent1.cities), "(a, b) = ", (a, b))

	# first, simply copy the region between a and b from parent1 to the child
	for i in range(a, b + 1):
		childCities[i] = parent1.cities[i]

	# now, the child will get cities from parent2, starting from position b + 1
	# until the end of the list, and then going around and filling positions
	# from 0 to a

	# i represents the position in the child
	# k represents the position in parent2
	# if b is the final position, then must wrap around now
	i = b + 1 if b < size - 1 else 0
	k = b + 1 if b < size - 1 else 0


	while i != a:
		# if the city in position k in parent2 isn't already in the child
		# add that city to position i in the child, and increment both counters

		#DEBUG
		# print('k = ', k, '; size = ', size)
		if parent2.cities[k] not in childCities:
			childCities[i] = parent2.cities[k]
			# if i or k < size - 1, simply increment
			# if not, wrap around to 0
			i = i + 1 if i < (size - 1) else 0
			k = k + 1 if k < (size - 1) else 0
		else: # parent.cities[k] in childCities
		# increment only counter k, or else position i in the child will be empty
			k = k + 1 if k < (size - 1) else 0

	return Tour(childCities, tourLength(childCities))

def partiallyMappedCrossover(parent1, parent2):
	""" Performs partially-mapped crossover with parent1 and parent2 and returns
		the	resulting child.

		Parents are assumed to be the same size.
	"""
	# tour size
	size = len(parent1.cities)

	# initialize the child as an empty list with the same size as the parents
	childCities = [None] * size

	# a and b represent the first and second cutpoints, respectively
	a = random.randint(0, size - 1)
	b = random.randint(0, size - 1)

	# Check if the cutpoints are valid, that is, a must be different than b,
	# and b must be greater than a.
	# If b is not greater than a, but they're different, simply swap them.
	validCutPoints = a < b
	while not validCutPoints:
		if a == b:
			a = random.randint(0, size - 1)
			b = random.randint(0, size - 1)
		else: # a != b, but a > b, so swap(a,b)
			a,b = b,a
		validCutPoints = a < b

	# DEBUG
	# print("childSize = ", len(childCities), ", parent1 size = ", len(parent1.cities), "(a, b) = ", (a, b))

	# first, simply copy the region between a and b from parent1 to the child
	for i in range(a, b + 1):
		childCities[i] = parent1.cities[i]

	# now the child will get cities from parent2
	# I don't even know how to explain this logic
	for i in range(size):
		if not a <= i <= b:
			if parent2.cities[i] not in childCities:
				childCities[i] = parent2.cities[i]
			else:
				k = parent1.cities.index(parent2.cities[i])
				while childCities[i] is None:
					if parent2.cities[k] not in childCities:
						childCities[i] = parent2.cities[k]
					else:
						k = parent1.cities.index(parent2.cities[k])

	return Tour(childCities, tourLength(childCities))

def swapMutation(tour):
	"""
		Simple swap mutation operator. Selects at random two positions in the
		tour and swaps the cities in those positions.
	"""

	size = len(tour.cities)
	# a and b represent the first and second sawp points, respectively
	a = random.randint(0, size - 1)
	b = random.randint(0, size - 1)

	# DEBUG
	#print("a = ", a, ", b = ", b, ", size = ", size)

	(tour.cities[a], tour.cities[b]) = (tour.cities[b], tour.cities[a])

	tour.updateLength()
	return tour

def evolvePopulationA(pop, elitism, mutationRate):
	"""
		Evolve a population using tournament selection and order crossover, with
		elitism percentage and mutation rate given.
	"""
	newPop = Population(pop.popSize, False)
	tournamentSize = 2

	# elitism represents a percentage of individuals from the population
	# that will pass on to the new population
	if elitism:
		# offset = number of individuals that will pass on to the new population
		offset = int(elitism/100 * pop.popSize)
		# # DEBUG
		# print("popSize: ", pop.popSize, ", elitism: ", elitism, "%, offset: ", offset)
		# selects the <offset> best individuals from pop and puts them in newPop
		best = hq.nsmallest(offset, pop.tours)
		# #DEBUG
		# print("best: ", best)
		for i in range(offset):
			# #DEBUG
			# print("i: ", i ,", popSize: ", pop.popSize)
			newPop.tours.append(best[i])
	else: # no elitism
		offset = 0

	# now fills the rest of the population by breeding two random individuals
	# first selects two parents and then applies crossover between them
	# then mutates
	for i in range(offset, newPop.popSize):

		# tournament selection:
		parent1 = tournamentSelection(pop, tournamentSize)
		parent2 = tournamentSelection(pop, tournamentSize)

		# order crossover
		child = orderCrossover(parent1, parent2)

		# mutation
		if (random.uniform(0,100) <= mutationRate):
			child = swapMutation(child)

		# newPop.tours[i] = child
		newPop.tours.append(child)

	return newPop

def evolvePopulationB(pop, elitism, mutationRate):
	""" Evolve a population using roulete wheel selection and order crossover,
		with elitism percentage and mutation rate given.
	"""
	newPop = Population(pop.popSize, False)
	tournamentSize = 2

	# elitism represents a percentage of individuals from the population
	# that will pass on to the new population
	if elitism:
		# offset = number of individuals that will pass on to the new population
		offset = int(elitism/100 * pop.popSize)
		# # DEBUG
		# print("popSize: ", pop.popSize, ", elitism: ", elitism, "%, offset: ", offset)
		# selects the <offset> best individuals from pop and puts them in newPop
		best = hq.nsmallest(offset, pop.tours)
		# #DEBUG
		# print("best: ", best)
		for i in range(offset):
			# #DEBUG
			# print("i: ", i ,", popSize: ", pop.popSize)
			newPop.tours.append(best[i])
	else: # no elitism
		offset = 0

	# now fills the rest of the population by breeding two random individuals
	# first selects two parents and then applies crossover between them
	# then mutates
	for i in range(offset, newPop.popSize):

		# roulette wheel selection:
		sumLengths = 0
		for tour in pop.tours:
			sumLengths += tour.length
		parent1 = rouletteWheelSelection(pop, sumLengths)
		parent2 = rouletteWheelSelection(pop, sumLengths)

		# order crossover
		child = orderCrossover(parent1, parent2)

		# mutation
		if (random.uniform(0,100) <= mutationRate):
			child = swapMutation(child)

		# newPop.tours[i] = child
		newPop.tours.append(child)

	return newPop

def evolvePopulationC(pop, elitism, mutationRate):
	""" Evolve a population using tournament selection and partially-mapped
	 	crossover, with	elitism percentage and mutation rate given.
	"""
	newPop = Population(pop.popSize, False)
	tournamentSize = 2

	# elitism represents a percentage of individuals from the population
	# that will pass on to the new population
	if elitism:
		# offset = number of individuals that will pass on to the new population
		offset = int(elitism/100 * pop.popSize)
		# # DEBUG
		# print("popSize: ", pop.popSize, ", elitism: ", elitism, "%, offset: ", offset)
		# selects the <offset> best individuals from pop and puts them in newPop
		best = hq.nsmallest(offset, pop.tours)
		# #DEBUG
		# print("best: ", best)
		for i in range(offset):
			# #DEBUG
			# print("i: ", i ,", popSize: ", pop.popSize)
			newPop.tours.append(best[i])
	else: # no elitism
		offset = 0

	# now fills the rest of the population by breeding two random individuals
	# first selects two parents and then applies crossover between them
	# then mutates
	for i in range(offset, newPop.popSize):

		# tournament selection:
		parent1 = tournamentSelection(pop, tournamentSize)
		parent2 = tournamentSelection(pop, tournamentSize)

		# partially mapped crossover
		child = partiallyMappedCrossover(parent1, parent2)

		# swap mutation
		if (random.uniform(0,100) <= mutationRate):
			child = swapMutation(child)

		# newPop.tours[i] = child
		newPop.tours.append(child)

	return newPop

def evolvePopulationD(pop, elitism, mutationRate):
	""" Evolve a population using roulette wheel selection and partially-mapped
	 	crossover, with	elitism percentage and mutation rate given.
	"""
	newPop = Population(pop.popSize, False)
	tournamentSize = 2

	# elitism represents a percentage of individuals from the population
	# that will pass on to the new population
	if elitism:
		# offset = number of individuals that will pass on to the new population
		offset = int(elitism/100 * pop.popSize)
		# # DEBUG
		# print("popSize: ", pop.popSize, ", elitism: ", elitism, "%, offset: ", offset)
		# selects the <offset> best individuals from pop and puts them in newPop
		best = hq.nsmallest(offset, pop.tours)
		# #DEBUG
		# print("best: ", best)
		for i in range(offset):
			# #DEBUG
			# print("i: ", i ,", popSize: ", pop.popSize)
			newPop.tours.append(best[i])
	else: # no elitism
		offset = 0

	# now fills the rest of the population by breeding two random individuals
	# first selects two parents and then applies crossover between them
	# then mutates
	for i in range(offset, newPop.popSize):

		# roulette wheel selection:
		sumLengths = 0
		for tour in pop.tours:
			sumLengths += tour.length
		parent1 = rouletteWheelSelection(pop, sumLengths)
		parent2 = rouletteWheelSelection(pop, sumLengths)

		# partially-mapped crossover
		child = partiallyMappedCrossover(parent1, parent2)

		# mutation
		if (random.uniform(0,100) <= mutationRate):
			child = swapMutation(child)

		# newPop.tours[i] = child
		newPop.tours.append(child)

	return newPop

def readAdjMatrix(filename):
	""" Reads a TSPLIB format file and converts it to an adjacenty matrix.
		Returns the numpy adjacency matrix.
	"""

	with open(filename, 'r') as fobj:
		lines = [[int(num) for num in line.split()] for line in fobj]

	# the length of the first line is the number of cities - 1 in the TSP,
	# so I must add 1 to it
	numberOfCities = len(lines[0]) + 1

	matrix = [[0] * numberOfCities for i in range(numberOfCities)]

	# Since the matrix in the .tsp file is only diagonal, each line has 1 less
	# int than the previous one. So the offset starts at 1 and is incremented
	# by one after each row is processed.
	offset = 1
	for i in range(numberOfCities):
		for j in range(numberOfCities):
			if i < j:
				# lines[][] has offset fewer columns than matrix[][]
				matrix[i][j] = lines[i][j - offset]
			elif i > j:
				matrix[i][j] = matrix[j][i]
		offset += 1

	matrix = np.matrix(matrix)

	return matrix

def runEvolveModes(filename, popSize, maxGenerations, elitism, mutationRate):
	""" Try to find an optimal solution using each of the different evolve modes
		A, B, C and D.

		Returns a list with the the best solutions found in each mode.
	"""
	global A
	global G

	# whether to read from a file or use a previously generated graph
	if filename:
		A = readAdjMatrix(filename)
		G = nx.from_numpy_matrix(A)

	# free A
	A = []

	# generates initial population
	initialPop = Population(popSize, True)
	solutions = []

	pop = initialPop
	for g in range(2, maxGenerations + 1):
		pop = evolvePopulationA(pop, elitism, mutationRate)
	solutions.append(hq.heappop(pop.tours))

	pop = initialPop
	for g in range(2, maxGenerations + 1):
		pop = evolvePopulationB(pop, elitism, mutationRate)
	solutions.append(hq.heappop(pop.tours))

	pop = initialPop
	for g in range(2, maxGenerations + 1):
		pop = evolvePopulationC(pop, elitism, mutationRate)
	solutions.append(hq.heappop(pop.tours))

	pop = initialPop
	for g in range(2, maxGenerations + 1):
		pop = evolvePopulationD(pop, elitism, mutationRate)
	solutions.append(hq.heappop(pop.tours))

	return solutions

def main():
	global A
	global G
	runTimes = 20
	filename = "brazil58.tsp"
	optimal = 0

	popSize = 100
	maxGenerations = 500
	elitism = 10 # in %
	mutationRate = 1.5 # in %

	# generate random graph
	n = 40
	G = nx.complete_graph(n, None)
	for i in range(n):
		for j in range(n):
			if i < j:
				G[i][j]['weight'] = random.randint(10,500)
			elif i > j:
				G[i][j]['weight'] = G[j][i]['weight']

	solutions = []
	bestSolutions = [np.inf] * 4
	hits = [0] * 4 # number of times the solution was optimal
	solutionSum = [0] * 4 # sums of the lenghts of the best solutions on each run

	# runs a number of times and saves the statistics for each mode
	for r in range(runTimes):
		solutions = runEvolveModes(filename, popSize, maxGenerations, elitism, mutationRate)
		for i in range(4):
			if solutions[i].length < bestSolutions[i]:
				bestSolutions[i] = solutions[i].length

			# adds the length of the best solution found to the sum
			solutionSum[i] += tourLength(solutions[i].cities)

			# if the solution was optimal, register a hit
			if tourLength(solutions[i].cities) == optimal:
				hits[0] += 1
		if r % 2 == 0:
			print("ran ", r, " times!")

	# print results
	print("Mode A >> avg tour length: ", solutionSum[0]/runTimes, end='')
	print(", hits: ", hits[0], "/", runTimes, end='')
	print(", best solution: ", bestSolutions[0])

	print("Mode B >> avg tour length: ", solutionSum[1]/runTimes, end='')
	print(", hits: ", hits[1], "/", runTimes, end='')
	print(", best solution: ", bestSolutions[1])

	print("Mode C >> avg tour length: ", solutionSum[2]/runTimes, end='')
	print(", hits: ", hits[2], "/", runTimes, end='')
	print(", best solution: ", bestSolutions[2])

	print("Mode D >> avg tour length: ", solutionSum[3]/runTimes, end='')
	print(", hits: ", hits[3], "/", runTimes, end='')
	print(", best solution: ", bestSolutions[3])

	# # original loop, whithout run modes
	# global A
	# global B
	#
	# A = readAdjMatrix(filename)
	# G = nx.from_numpy_matrix(A)
	#
	# # free A
	# A = []
	#
	# elitism = 5 # in %
	# mutationRate = 1.5 # in %
	# maxGenerations = 1000
	# popSize = 100
	#
	# # generates initial population
	# pop = Population(popSize, True)
	#
	# bestSolution = pop.getShortest()
	# print("best random solution: ", tourLength(bestSolution.cities))
	#
	# # main loop. evolve the population maxGenerations times
	# for g in range(2, maxGenerations + 1):
	# 	pop = evolvePopulation(pop, elitism, mutationRate)
	#
	# 	if g % 100 == 0:
	# 		print("generation ", g)
	#
	# bestSolution = pop.getShortest()
	# print("best found solution: ", tourLength(bestSolution.cities))

	""" TESTS """
	# # testing the orderCrossover operator: OK
	# p1 = [1, 2, 5, 6, 4, 3, 8, 7]
	# p2 = [1, 4, 2, 3, 6, 5, 7, 8]
	#
	# parent1 = Tour(p1, tourLength(p1))
	# parent2 = Tour(p2, tourLength(p2))
	#
	# child = orderCrossover(parent1, parent2)
	#
	# print('p1 = ', p1)
	# print('p2 = ', p2)
	# print('child = ', child.cities)

	# # testing the partiallyMappedCrossover operator: OK
	# t1 = [5,6,7,2,3,4,1]
	# t2 = [2,5,6,3,4,1,7]
	#
	# parent1 = Tour(t1, tourLength(t1))
	# parent2 = Tour(t2, tourLength(t2))
	#
	# child = partiallyMappedCrossover(parent1, parent2)
	#
	# print('t1 = ', t1)
	# print('t2 = ', t2)
	# print('child = ', child.cities)

	# # testing the swapMutation operator: OK
	# swapMutation(child)
	# print('mutated child = ', child.cities)

	# # testing initial population generator: OK
	# pop = genRandomTours(5)
	# printPop(pop)
	#
	# # testing the > and < operators
	# for i in range(0, len(pop) - 1):
	# 	if pop[i] > pop[i+1]:
	# 		print("tour ", i, " is greater than tour ", i+1)
	# 	else:
	# 		print("tour ", i, " is lesser than tour ", i+1)

	# # testing the tourLength calculator and stuff related to it
	# t1 = [0,1,2,3,4] # length 10.103
	# t2 = [1,3,2,0,4] # length 10.345
	# t3 = [4,2,3,1,0] # length 10.307
	# t4 = [3,0,2,1,4] # length 8.954
	# t5 = [2,4,0,3,1] # length 7.693
	# print("t1 = ", t1, ", t1Length = ", tourLength(t1))
	# print("t2 = ", t2, ", t2Length = ", tourLength(t2))
	# print("t3 = ", t3, ", t3Length = ", tourLength(t3))
	# print("t4 = ", t4, ", t4Length = ", tourLength(t4))
	# print("t5 = ", t5, ", t5Length = ", tourLength(t5))
	#
	# t1 = Tour(t1, tourLength(t1))
	# t2 = Tour(t2, tourLength(t2))
	# t3 = Tour(t3, tourLength(t3))
	# t4 = Tour(t4, tourLength(t4))
	# t5 = Tour(t5, tourLength(t5))
	#
	# testPop = Population(5, False)
	#
	# testPop.tours.append(t1)
	# testPop.tours.append(t2)
	# testPop.tours.append(t3)
	# testPop.tours.append(t4)
	# testPop.tours.append(t5)
	#
	# best = hq.nsmallest(2, testPop.tours)
	# print("best tours: ", best[0].cities, " and ", best[1].cities)

if __name__ == '__main__':
	main()
