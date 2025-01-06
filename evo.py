from deap import base, creator, tools, algorithms
from shapely import LineString
import numpy as np
import itertools
import math

# FIXME: czasami z jakiegoś powodu nie działa
def liczba_przeciec(edges, positions):
    reshaped = np.reshape(positions, (-1, 2))
    edge_pos = [LineString(reshaped[e]) for e in edges]
    intersectins = 0.0
    seen_vertices = {}

    for i1, i2 in itertools.combinations(range(len(edges)), 2):
        # Sprawdzanie, czy krawędzie mają wspólny wierzchołek
        if set(edges[i1]) & set(edges[i2]):
            continue
        
        # Sprawdzanie, czy krawędzie się przecinają
        if edge_pos[i1].intersects(edge_pos[i2]):
            intersection = edge_pos[i1].intersection(edge_pos[i2])

            # linie na siebie nachodzą - nieskończoność przecięć
            if intersection.geom_type != 'Point':
                return math.inf

            intersectins += 1

    return (intersectins,)

def evaluate_indiviudal(edges, individual):
    '''
    Funkcja pomocnicza dla biblioteki deap.
    - edges: lista krawędzi grafu
    - individual: lista wszystkich obosników w generacji. Każdy element list to osobny zestaw wszystkich współrzędnych.
    '''
    return [liczba_przeciec(edges, pos) for pos in individual]

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


def search_best(n_verts, edges, start=None):
    '''
    # WSZYSTKIE ZAMIANY W ALGORYTMIE ZAPISUJCIE

    Główna część zadania. Za pomocą algorytmu genetycznego próbuje znaleźć rozwiązanie.

    Czy DEAP jest dobrym rozwiązaniem? Nie mam pojęcia. Jeśli tylko znajdziecie lepszą feel free żeby to zmienić.
    '''

    # PAMIĘTAJCIE KILKA RAZY URUCHAMIAĆ I ZAPISYWAĆ WSZYSTKO CO TYLKO PRÓBOWALIŚCIE
    # WSZYSTKIE EKSPERYMENTY MOGĄ SIĘ PRZYDAĆ
    # KILKA RAZY BO WYNIKI SĄ LOSOWE - JEDNO PRZEJŚCIE CHUJA MÓWI

    # przypisanie funkcji dopasowania
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # zawężenie dziedziny genów
    LOWER_BOUND = -n_verts
    UPPER_BOUND = n_verts
    toolbox.register("attr_float", np.random.uniform, LOWER_BOUND, UPPER_BOUND)

    # przygotowanie początkowej populacji
    # ustalenie dobrego początku może być też pomocne. W tym przypadku zaczynamy od losowych wartości.
    # inną opcją byłoby np. wyliczenie współrzędnych korzystając ze spring_layout z networkx czy cuś
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_verts*2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # operator krzyżowania - krzyżowanie jednorodne tylko dla pełnych punktów
    toolbox.register("mate", cx_point_uniform, indpb=0.6)

    # operator mutacji - funkcja gaussa
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=3, indpb=0.1) 

    # operator selekcji - turniejowa
    toolbox.register("select", tools.selTournament, tournsize=3)

    # funkcja dopasowania
    toolbox.register("evaluate", liczba_przeciec, edges)

    # podstawowe statystyki
    # TODO: zautomatyzować samodzielne zapisywanie statystyk i uruchamianie kilka razy
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1, similar=np.array_equal) # lista najlepszych osobników

    # uruchomienie algorytmu
    pop = toolbox.population(n=100)
    _, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=250, stats=stats, halloffame=hof, verbose=True)

    return hof[0]
