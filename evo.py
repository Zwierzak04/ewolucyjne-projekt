from deap import base, creator, tools, algorithms
from shapely import LineString
import numpy as np
import itertools
import math
from visualizer import visualize

# FIXME: czasami z jakiegoś powodu nie działa
def liczba_przeciec(edges, positions):
    reshaped = np.reshape(positions, (-1, 2))
    edge_pos = [LineString(reshaped[e]) for e in edges]
    intersections = 0.0
    for i1, i2 in itertools.combinations(range(len(edges)), 2):
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

def liczba_przeciec2(edges, positions):
    return (liczba_przeciec(edges, positions),)

def liczba_przeciec_inv(edges, positions):
    return ((1 / (liczba_przeciec(edges, positions) + 1)),)

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
    LOWER_BOUND = -1
    UPPER_BOUND = 1
    toolbox.register("attr_float", np.random.uniform, LOWER_BOUND, UPPER_BOUND)

    # przygotowanie początkowej populacji
    # ustalenie dobrego początku może być też pomocne. W tym przypadku zaczynamy od losowych wartości.
    # inną opcją byłoby np. wyliczenie współrzędnych korzystając ze spring_layout z networkx czy cuś
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_verts*2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # operator krzyżowania - krzyżowanie jednorodne tylko dla pełnych punktów
    toolbox.register("mate", cx_point_uniform, indpb=0.7)

    # operator mutacji - funkcja gaussa
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.33, indpb=0.75) 

    # operator selekcji - turniejowa
    toolbox.register("select", tools.selTournament, tournsize=3)

    # funkcja dopasowania
    toolbox.register("evaluate", liczba_przeciec2, edges)

    # podstawowe statystyki
    # TODO: zautomatyzować samodzielne zapisywanie statystyk i uruchamianie kilka razy
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1, similar=np.array_equal) # lista najlepszych osobników

    # uruchomienie algorytmu
    pop = toolbox.population(n=100)
    _, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=100, stats=stats, halloffame=hof, verbose=True)
    #_, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=100, lambda_=200, mutpb=0.1,cxpb=0.9, ngen=100, verbose=True, stats=stats, halloffame=hof)

    return hof[0]

def search_best_inv(n_verts, edges, start=None):
    '''
    # WSZYSTKIE ZAMIANY W ALGORYTMIE ZAPISUJCIE

    Główna część zadania. Za pomocą algorytmu genetycznego próbuje znaleźć rozwiązanie.

    Czy DEAP jest dobrym rozwiązaniem? Nie mam pojęcia. Jeśli tylko znajdziecie lepszą feel free żeby to zmienić.
    '''

    # PAMIĘTAJCIE KILKA RAZY URUCHAMIAĆ I ZAPISYWAĆ WSZYSTKO CO TYLKO PRÓBOWALIŚCIE
    # WSZYSTKIE EKSPERYMENTY MOGĄ SIĘ PRZYDAĆ
    # KILKA RAZY BO WYNIKI SĄ LOSOWE - JEDNO PRZEJŚCIE CHUJA MÓWI

    # przypisanie funkcji dopasowania
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # zawężenie dziedziny genów
    LOWER_BOUND = -1
    UPPER_BOUND = 1
    toolbox.register("attr_float", np.random.uniform, LOWER_BOUND, UPPER_BOUND)

    # przygotowanie początkowej populacji
    # ustalenie dobrego początku może być też pomocne. W tym przypadku zaczynamy od losowych wartości.
    # inną opcją byłoby np. wyliczenie współrzędnych korzystając ze spring_layout z networkx czy cuś
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_verts*2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # operator krzyżowania - krzyżowanie jednorodne tylko dla pełnych punktów
    toolbox.register("mate", cx_point_uniform, indpb=0.7)

    # operator mutacji - funkcja gaussa
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.25, indpb=0.6) 

    # operator selekcji - turniejowa
    #toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register('select', tools.selRoulette)

    # funkcja dopasowania
    toolbox.register("evaluate", liczba_przeciec_inv, edges)

    # podstawowe statystyki
    # TODO: zautomatyzować samodzielne zapisywanie statystyk i uruchamianie kilka razy
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1, similar=np.array_equal) # lista najlepszych osobników

    # uruchomienie algorytmu
    pop = toolbox.population(n=100)
    _, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=100, stats=stats, halloffame=hof, verbose=True)
    #_, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=100, lambda_=200, mutpb=0.1,cxpb=0.9, ngen=100, verbose=True, stats=stats, halloffame=hof)

    return hof[0]

def swap_intersections_mutation(individual, edges, verts): # , indpb
    #print('what')
    reshaped = np.reshape(individual, (-1, 2))
    edge_pos = [LineString(reshaped[e]) for e in edges]
    for i1, i2 in itertools.combinations(range(len(edges)), 2):
        intersection = edge_pos[i1].intersection(edge_pos[i2])

        # brak przecięcia
        if intersection.is_empty:
            continue

        # linie na siebie nachodzą - nieskończoność przecięć
        if(intersection.geom_type != 'Point'):
            visualize(verts, edges, individual)
            continue

        # pomijamy krawędzie, które mają wspólne wierzchołki
        # jeśli na siebie nie nachodzą to nie ma sensu ich sprawdzać
        if set(edges[i1]) & set(edges[i2]):
            continue

        # tam gdzie są przecięcia to próbujemy je odplątać
        prev = np.copy(individual)
        #if np.random.random_sample() < indpb:
        which_one = np.random.randint(0,2) # początek czy koniec krawędzi podmieniamy
        vert1 = 2*edges[i1][which_one]
        vert2 = 2*edges[i2][which_one]
        individual[vert1], individual[vert2] = individual[vert2], individual[vert1]
        individual[vert1+1], individual[vert2+1] = individual[vert2+1], individual[vert1+1]

        if(liczba_przeciec(edges, individual)-2137 < 1):
            visualize(verts, edges, prev)
            visualize(verts, edges, individual)

        break

    return individual,

def mieszane(individual, edges, verts):
    ind, = swap_intersections_mutation(individual, edges=edges, verts=verts)
    return tools.mutGaussian(ind, indpb=0.2, mu=0.0, sigma=0.33)

def search_fancy_mutation(n_verts, edges, start=None):

    bruh = list(range(n_verts))

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
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_verts*2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # operator krzyżowania - krzyżowanie jednorodne tylko dla pełnych punktów
    toolbox.register("mate", cx_point_uniform, indpb=0.6)

    # operator mutacji - funkcja gaussa
    toolbox.register("mutate", mieszane, edges=edges, verts=bruh) 

    # operator selekcji - turniejowa
    toolbox.register('select', tools.selTournament, tournsize=3)

    # funkcja dopasowania
    toolbox.register("evaluate", liczba_przeciec2, edges)

    # podstawowe statystyki
    # TODO: zautomatyzować samodzielne zapisywanie statystyk i uruchamianie kilka razy
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1, similar=np.array_equal) # lista najlepszych osobników

    # uruchomienie algorytmu
    pop = toolbox.population(n=100)
    _, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=200, stats=stats, halloffame=hof, verbose=True)
    #_, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=100, lambda_=200, mutpb=0.1,cxpb=0.9, ngen=100, verbose=True, stats=stats, halloffame=hof)

    return hof[0]