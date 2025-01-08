from deap import base, creator, tools, algorithms
from shapely import LineString
import numpy as np
import itertools
import math
from visualizer import visualize

def liczba_przeciec(edges, positions):
    reshaped = np.reshape(positions, (-1, 2))
    #print(reshaped, '\n')
    edge_pos = [LineString(reshaped[e]) for e in edges]
    intersections = 0.0
    for i1, i2 in itertools.combinations(range(len(edges)), 2):
        intersection = edge_pos[i1].intersection(edge_pos[i2])

        # brak przecięcia
        if intersection.is_empty:
            continue

        # linie na siebie nachodzą - nieskończoność przecięć
        if(intersection.geom_type != 'Point'):
            return 2137**3

        # pomijamy krawędzie, które mają wspólne wierzchołki
        # jeśli na siebie nie nachodzą to nie ma sensu ich sprawdzać
        if set(edges[i1]) & set(edges[i2]):
            continue

        intersections += 1

    return intersections*intersections

def liczba_przeciec2(edges, positions):
    return (liczba_przeciec(edges, positions),)

def cx_point_uniform(ind1, ind2, indpb):
    '''
    Krzyżowanie jednorodne, tylko dla punktów. Wymusza branie współrzędnych x i y razem.

    Aktualnie każdy osobnik jest postaci [x1, y1, x2, y2, ...]

    Zwykłe krzyżwoanie jednorodne pozwala na pożyczenie w potomku jednej współrzędnej od jednego rodzica, a drugiej od drugiego.
    Ta funkcja ma na celu restrykcję, żeby współrzędne x i y były zawsze razem.
    '''
    for i in range(0, len(ind1), 2):
        if np.random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
            ind1[i+1], ind2[i+1] = ind2[i+1], ind1[i+1]
    return ind1, ind2


def search_step(n_verts, fitness, iters=100, starting_point=None, min_fitness=0.0):
    '''
    # WSZYSTKIE ZAMIANY W ALGORYTMIE ZAPISUJCIE

    Główna część zadania. Za pomocą algorytmu genetycznego próbuje znaleźć rozwiązanie.

    Czy DEAP jest dobrym rozwiązaniem? Nie mam pojęcia. Jeśli tylko znajdziecie lepszą feel free żeby to zmienić.
    '''

    # przypisanie funkcji dopasowania
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # zawężenie dziedziny genów
    LOWER_BOUND = -1
    UPPER_BOUND = 1
    toolbox.register("attr_float", np.random.uniform, LOWER_BOUND, UPPER_BOUND)

    # przygotowanie początkowej populacji
    if starting_point is not None:
        tools.initIterate
        toolbox.register("individual", tools.initIterate, creator.Individual, starting_point)
    else:
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=(n_verts*2))
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # operator krzyżowania - krzyżowanie jednorodne tylko dla pełnych punktów
    toolbox.register("mate", cx_point_uniform, indpb=0.7)

    # operator mutacji - funkcja gaussa
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.33, indpb=0.75) 

    # operator selekcji - turniejowa
    toolbox.register("select", tools.selTournament, tournsize=3)

    # funkcja dopasowania
    toolbox.register("evaluate", fitness)

    # podstawowe statystyki
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1, similar=np.array_equal) # lista najlepszych osobników

    # uruchomienie algorytmu
    pop = toolbox.population(n=100)
    _, logbook = eaSimpleEarly(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=iters, stats=stats, halloffame=hof, verbose=True, min_fitness=min_fitness)

    return hof[0], hof[0].fitness.values[0]

def eaSimpleEarly(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, min_fitness = 0.0):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if record['min'] <= (min_fitness+0.5):
            print('early stopping')
            break

    return population, logbook

def search_best(n_verts, edges, step_size, iters_per_chunk):
    verts = np.zeros(n_verts*2)
    total_size = 0

    last_fit = 0.0
    for current_verts in itertools.batched(range(0, n_verts), step_size):

        relevant_verts = range(current_verts[-1]+1)
        relevant_edges = list(filter(lambda e: e[0] in relevant_verts and e[1] in relevant_verts, edges))

        print(f'{relevant_verts=} {relevant_edges=}')

        def fitness(x):
            verts[total_size:(total_size+len(x))] = x
            return liczba_przeciec2(relevant_edges, verts)
        
        best, fit = search_step(len(current_verts), fitness, iters_per_chunk, min_fitness=last_fit)
        last_fit = fit
        verts[total_size:total_size+len(best)] = best
        total_size += len(best)

    return verts

def gen_initial(verts, start, stop):
    v = np.copy(verts)
    v[:start] += np.random.randn() * 0.15
    v[start:stop] = np.random.random_sample()
    return v

def search_best_unfrozen(n_verts, edges, step_size, iters_per_chunk):
    verts = np.zeros(n_verts*2)

    total_size = 0
    for current_verts in itertools.batched(range(0, n_verts), step_size):
        slice_size = len(current_verts)*2
        relevant_verts = range(current_verts[-1]+1)
        relevant_edges = list(filter(lambda e: e[0] in relevant_verts and e[1] in relevant_verts, edges))

        print(f'{relevant_verts=} {relevant_edges=}')
        
        best, fit = search_step(len(current_verts), lambda x: liczba_przeciec2(relevant_edges, x), iters_per_chunk, starting_point=lambda: gen_initial(verts, total_size, total_size+slice_size), min_fitness=0.0)
        verts[:] = best
        total_size += slice_size

    return verts