from deap import base, creator, tools
import random
import numpy as np
import sys
from visualizer import visualize_krzywe as visualize
from loader import load_file
from stats import all_statistics
from evo3 import eaSimpleEarly

def alt_crossing(vertices, pages, edges):
    position = {v: i for i, v in enumerate(vertices)}
    crossings = 0

    # Funkcja pomocnicza do liczenia przecięć na jednej stronie
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if pages[i] != pages[j]:
                continue

            u1, v1 = edges[i]
            u2, v2 = edges[j]

            # Zamień na pozycje w vertices
            u1_pos, v1_pos = position[u1], position[v1]
            u2_pos, v2_pos = position[u2], position[v2]

            # Upewnij się, że (u, v) jest w porządku rosnącym
            if u1_pos > v1_pos:
                u1_pos, v1_pos = v1_pos, u1_pos
            if u2_pos > v2_pos:
                u2_pos, v2_pos = v2_pos, u2_pos

            # Sprawdź, czy krawędzie się przecinają
            if (u1_pos < u2_pos < v1_pos < v2_pos) or (u2_pos < u1_pos < v2_pos < v1_pos):
                crossings += 1

    return crossings


def deap_crossings(individual, n_verts, edges):
    vertices, edges_page = individual[:n_verts], individual[n_verts:]
    return (alt_crossing(vertices, edges_page, edges),)

def dfs_ordering(vertices, edges, visited=None):
    """
    Kolejność wierzchołków uzyskana algorytmem DFS.

    Args:
        vertices (list): Lista wierzchołków grafu.
        edges (list): Lista krawędzi w postaci (u, v).

    Returns:
        list: Kolejność wierzchołków.
    """
    visited = {v: False for v in vertices} if visited is None else visited
    order = []

    vertices_shuffled = random.sample(vertices, len(vertices))
    edges_shuffled = random.sample(edges, len(edges))

    def dfs(v):
        visited[v] = True
        for u, w in edges_shuffled:
            if u == v and not visited[w]:
                dfs(w)
        order.append(v)

    for v in vertices_shuffled:
        if not visited[v]:
            dfs(v)

    return order[::-1]

def dfs_individual(vertices, edges):
    return [*dfs_ordering(vertices, edges), *np.random.choice([0, 1], len(edges))]

def other_crossover(parent1, parent2, n_verts):
    ord1, p1 = parent1[:n_verts], parent1[n_verts:]
    ord2, p2 = parent2[:n_verts], parent2[n_verts:]
    ord1,ord2 = tools.cxPartialyMatched(ord1, ord2)
    p1,p2 = tools.cxUniform(p1,p2, indpb=0.5)
    parent1[:n_verts] = ord1
    parent1[n_verts:] = p1
    parent2[:n_verts] = ord2
    parent2[n_verts:] = p2
    return (parent1, parent2)

def other_mutate(individual, n_verts, indpb=0.2):
    vertex_order, edges_page = individual[:n_verts], individual[n_verts:]
    vertex_order, = tools.mutShuffleIndexes(vertex_order, indpb=indpb*0.5)
    edges_page, = tools.mutUniformInt(edges_page, 0, 1, indpb=indpb)
    individual[:n_verts], individual[n_verts:] = vertex_order, edges_page
    return (individual,)

def deap_redef(vertices, edges, stfu=False):
    n_verts = len(vertices)

    # przypisanie funkcji dopasowania
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # przygotowanie początkowej populacji
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: dfs_individual(vertices, edges))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # operator krzyżowania - krzyżowanie jednorodne tylko dla pełnych punktów
    toolbox.register("mate", other_crossover, n_verts=n_verts)

    # operator mutacji - funkcja gaussa
    toolbox.register("mutate", other_mutate, n_verts=n_verts, indpb=0.1)

    # operator selekcji - turniejowa
    #toolbox.register('select', tools.selBest)
    toolbox.register("select", tools.selTournament, tournsize=3) #tools.selTournament, tournsize=3)

    # funkcja dopasowania
    toolbox.register("evaluate", lambda ind: deap_crossings(ind, n_verts, edges))

    # podstawowe statystyki
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    # uruchomienie algorytmu
    pop = toolbox.population(n=100)
    _, logbook = eaSimpleEarly(pop, toolbox, cxpb=0.9, mutpb=0.3, ngen=250, stats=stats, halloffame=hof, verbose=not stfu)
    #_, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=50, lambda_=200, mutpb=0.5,cxpb=0.9, ngen=150, verbose=not stfu, stats=stats if not stfu else None, halloffame=hof)

    best_verts, best_pages = hof[0][:n_verts], hof[0][n_verts:]
    return (best_verts, best_pages), hof[0].fitness.values[0]

def test_deap():
    # Przykład użycia:
    graph_file = sys.argv[1] if len(sys.argv) > 1 else 'examples/dececahedron.txt'
    vertices, edges = load_file(graph_file)

    # Uruchom algorytm ewolucyjny
    (best_vertices, best_edges_page), fit = deap_redef(vertices, edges)
    pages_diff = [page*2-1 for page in best_edges_page]

    # Oblicz liczbę przecięć dla najlepszego rozwiązania
    best_crossings = fit

    print(f"Najlepszy układ wierzchołków: {best_vertices}")
    print(f"Najlepsze przypisanie stron krawędzi: {pages_diff}")
    print(f"Liczba przecięć: {best_crossings}")
    visualize(len(vertices), edges, (best_vertices, pages_diff))

if __name__ == "__main__":
    #test_deap()
    all_statistics('deap_krzywe', deap_redef)