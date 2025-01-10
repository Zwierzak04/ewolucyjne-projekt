import random
from visualizer import visualize_krzywe as visualize
from loader import load_file
from deap import tools
from deap import  tools
import sys
import numpy as np
from stats import all_statistics

def count_edge_crossings(vertices, edges_page):
    """
    Oblicza liczbę przecięć krawędzi w grafie z uwzględnieniem stron (góra/dół).

    Args:
        vertices (list): Lista wierzchołków w ustalonej kolejności.
        edges_page (list of tuples): Lista krawędzi w postaci ((u, v), page), gdzie page to 0 (góra) lub 1 (dół).

    Returns:
        int: Liczba przecięć krawędzi.
    """
    # Mapa wierzchołków na ich pozycje w vertices
    position = {v: i for i, v in enumerate(vertices)}

    crossings = 0

    # Wyodrębnij krawędzie dla każdej strony
    edges_top = [edge for edge, page in edges_page if page == 0]
    edges_bottom = [edge for edge, page in edges_page if page == 1]

    # Funkcja pomocnicza do liczenia przecięć na jednej stronie
    def count_crossings_on_page(edges):
        count = 0
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
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
                    count += 1
        return count

    # Licz przecięcia na każdej stronie osobno
    crossings += count_crossings_on_page(edges_top)
    crossings += count_crossings_on_page(edges_bottom)

    return crossings

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
    return dfs_ordering(vertices, edges), [((u, v), random.randint(0, 1)) for u, v in edges]

def generate_individual(vertices, edges):
    # Losowa permutacja wierzchołków i przypisanie stron krawędzi
    vertex_order = random.sample(vertices, len(vertices))
    edges_page = [((u, v), random.randint(0, 1)) for u, v in edges]
    return vertex_order, edges_page

def crossover(parent1, parent2):
    # Krzyżowanie permutacji wierzchołków
    vertex_order1, edges_page1 = parent1
    vertex_order2, edges_page2 = parent2
    split = random.randint(1, len(vertex_order1) - 1)
    child_vertex_order = vertex_order1[:split] + [v for v in vertex_order2 if v not in vertex_order1[:split]]
    # Krzyżowanie przypisania stron krawędzi
    child_edges_page = [(edge, random.choice([p1, p2])) for ((edge, p1), (_, p2)) in zip(edges_page1, edges_page2)]
    return (child_vertex_order, child_edges_page)

def pmx_crossover(parent1, parent2):
    vertex_order1, edges_page1 = parent1
    vertex_order2, edges_page2 = parent2
    v1, v2 = tools.cxPartialyMatched(vertex_order1.copy(), vertex_order2.copy())
    p1, p2 = edges_page1.copy(), edges_page2.copy()
    for i in range(len(p1)):
        if random.random() < 0.5:
            p1[i], p2[i] = p2[i], p1[i]
            
    return (v1, p1), (v2, p2)

def mutate(individual, mutation_rate=0.1):
    vertex_order, edges_page = individual
    # Mutacja permutacji wierzchołków
    if random.random() < mutation_rate:
        i,j = random.sample(range(len(vertex_order)), 2)
        vertex_order[i], vertex_order[j] = vertex_order[j], vertex_order[i]
    # Mutacja przypisania stron krawędzi
    for k in range(len(edges_page)):
        if random.random() < mutation_rate:
            edge, page = edges_page[k]
            edges_page[k] = (edge, 1 - page)  # Zmień stronę
    return vertex_order, edges_page

def mutate_harder(individual, mutation_rate=0.1):
    vertex_order, edges_page = individual
    # Mutacja permutacji wierzchołków
    for i in range(len(vertex_order)):
        if random.random() < mutation_rate:
            j = np.random.randint(0, len(vertex_order))#random.sample(range(len(vertex_order)), 1)[0]
            vertex_order[i], vertex_order[j] = vertex_order[j], vertex_order[i]
    # Mutacja przypisania stron krawędzi
    for k in range(len(edges_page)):
        if random.random() < mutation_rate:
            edge, page = edges_page[k]
            edges_page[k] = (edge, 1 - page)  # Zmień stronę
    return vertex_order, edges_page

def evolutionary_algorithm(vertices, edges, population_size=105, generations=250, mutation_rate=0.1, stfu=False):
    """
    Algorytm ewolucyjny minimalizujący liczbę przecięć krawędzi.

    Args:
        vertices (list): Lista wierzchołków grafu.
        edges (list): Lista krawędzi w postaci (u, v).
        population_size (int): Liczba osobników w populacji.
        generations (int): Liczba pokoleń.
        mutation_rate (float): Prawdopodobieństwo mutacji.

    Returns:
        tuple: Najlepsze znalezione rozwiązanie (lista wierzchołków, przypisanie stron krawędzi).
    """
    evaluation_count = 0

    # Inicjalizacja populacji
    population = [generate_individual(vertices, edges) for _ in range(population_size)]

    for generation in range(generations):
        # Ocena populacji
        fitness = []
        for ind in population:
            fitness_value = count_edge_crossings(ind[0], ind[1])
            fitness.append(fitness_value)
            evaluation_count += 1

        sorted_population = sorted(zip(population, fitness), key=lambda x: x[1])
        population = [ind for ind, _ in sorted_population]

        # Wypisz wynik dla obecnej generacji
        best_fitness = sorted_population[0][1]

        if not stfu:
            print(f"Generacja {generation + 1}: Najlepsza liczba przecięć = {best_fitness}, Ewaluacje = {evaluation_count}")

        if abs(best_fitness) < 0.1:
            break

        # Selekcja rodziców (turniej)
        parents = population[:population_size // 2]

        # Tworzenie nowej populacji
        new_population = parents[:]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))

        population = new_population

    # Zwróć najlepsze rozwiązanie
    best_solution = min(population, key=lambda ind: count_edge_crossings(ind[0], ind[1]))
    return best_solution, best_fitness

def evolutionary_algorithm2(vertices, edges, population_size=125, generations=300, mutation_rate=0.05, stfu=False):
    evaluation_count = 0

    # Inicjalizacja populacji
    #population = [generate_individual(vertices, edges) for _ in range(population_size)]
    population = [dfs_individual(vertices, edges) for _ in range(population_size)]

    for generation in range(generations):
        # Ocena populacji
        fitness = []
        for ind in population:
            fitness_value = count_edge_crossings(ind[0], ind[1])
            fitness.append(fitness_value)
            evaluation_count += 1

        sorted_population = sorted(zip(population, fitness), key=lambda x: x[1])
        population = [ind for ind, _ in sorted_population]

        # Wypisz wynik dla obecnej generacji
        best_fitness = sorted_population[0][1]

        if not stfu:
            print(f"Generacja {generation + 1}: Najlepsza liczba przecięć = {best_fitness}, Ewaluacje = {evaluation_count}")

        if abs(best_fitness) < 0.1:
            break

        # Selekcja rodziców (turniej)
        parents = population[:population_size // 2]

        # Tworzenie nowej populacji
        new_population = parents[:]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            #child1 = crossover(parent1, parent2)
            child1, child2 = pmx_crossover(parent1, parent2)
            new_population.append(mutate_harder(child1, mutation_rate))
            new_population.append(mutate_harder(child2, mutation_rate))

        population = new_population

    # Zwróć najlepsze rozwiązanie
    best_solution = min(population, key=lambda ind: count_edge_crossings(ind[0], ind[1]))
    return best_solution, best_fitness

def test_algo(algo):
    # Przykład użycia:
    graph_file = sys.argv[1] if len(sys.argv) > 1 else 'examples/dececahedron.txt'
    vertices, edges = load_file(graph_file)

    # Uruchom algorytm ewolucyjny
    (best_vertices, best_edges_page), fit = algo(vertices, edges)
    edges2 = [(u-1, v-1) for (u, v) in edges]
    verts = [v-1 for v in best_vertices]
    pages_diff = [page*2-1 for _, page in best_edges_page]

    print(f"Najlepszy układ wierzchołków: {best_vertices}")
    print(f"Najlepsze przypisanie stron krawędzi: {best_edges_page}")
    print(f"Liczba przecięć: {fit}")
    visualize(len(verts), edges2, (verts, pages_diff))

if __name__ == "__main__":
    #test_algo(evolutionary_algorithm2)
    all_statistics('krzywe_pmx', evolutionary_algorithm2)
    all_statistics('krzywe_original', evolutionary_algorithm)