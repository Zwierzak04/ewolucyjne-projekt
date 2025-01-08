from deap import base, creator, tools, algorithms
from shapely import LineString
import numpy as np
import itertools
import math

def liczba_przeciec(edges, positions, relevant_edges=None):
    reshaped = np.reshape(positions, (-1, 2))
    edge_pos = [LineString(reshaped[e]) for e in edges]
    intersections = 0.0
    for i1, i2 in itertools.combinations(range(len(edges)), 2):
        if relevant_edges is not None and (i1 not in relevant_edges or i2 not in relevant_edges):
            continue

        intersection = edge_pos[i1].intersection(edge_pos[i2])

        # brak przecięcia
        if intersection.is_empty:
            continue

        # linie na siebie nachodzą - nieskończoność przecięć
        if(intersection.geom_type != 'Point'):
            return 2137

        # pomijamy krawędzie, które mają wspólne wierzchołki
        # jeśli na siebie nie nachodzą to nie ma sensu ich sprawdzać
        if set(edges[i1]) & set(edges[i2]):
            continue

        intersections += 1

    return intersections

def liczba_przeciec2(edges, positions, relevant_edges=None):
    return (liczba_przeciec(edges, positions, relevant_edges),)

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


def search_step(n_verts, fitness, iters=100):
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
    # ustalenie dobrego początku może być też pomocne. W tym przypadku zaczynamy od losowych wartości.
    # inną opcją byłoby np. wyliczenie współrzędnych korzystając ze spring_layout z networkx czy cuś
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
    _, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=iters, stats=stats, halloffame=hof, verbose=True)

    return hof[0]

def search_best(n_verts, edges, step_size, iters_per_step):

    # zmieniamy pary na rzeczywistą listę sąsiedztwa w końcu
    # edges_dict = {}
    # for e in edges:
    #     if e[0] not in edges_dict:
    #         edges_dict[e[0]] = []
    #     edges_dict[e[0]].append(e[1])

    # potencjalna optymalizacja - kolejność breadth-first przeglądania wierzchołków
    # starting_vert = np.random.randint(0, n_verts)
    # order = [starting_vert]
    # stack = [starting_vert]
    # for i in range(n_verts-1):
    #     order.append(edges_dict[order[-1]])

    verts = np.random.rand(n_verts*2)
    for i in range(0, n_verts, step_size):
        view_boundaries = (i*2, (i+step_size)*2)
        verts_chunk = verts[view_boundaries[0]:view_boundaries[1]]
        
        print(f'Boundaries {view_boundaries}')
        print(f'Starting step {i+len(verts_chunk)//2}')
        print(f'Current verts: {verts[view_boundaries[0]:view_boundaries[1]]}')

        def fitness(x):
            v = np.copy(verts)
            v[view_boundaries[0]:view_boundaries[1]] = x
            return liczba_przeciec2(edges, v, relevant_edges=range(i+step_size))
        
        best = search_step(len(verts_chunk)//2, fitness, iters_per_step)
        verts_chunk[:] = best

    return verts