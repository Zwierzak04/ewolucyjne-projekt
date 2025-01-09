import random
from dokument1 import visualize

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

def evolutionary_algorithm(vertices, edges, population_size=100, generations=250, mutation_rate=0.1):
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

    def generate_individual():
        # Losowa permutacja wierzchołków i przypisanie stron krawędzi
        vertex_order = random.sample(vertices, len(vertices))
        edges_page = [((u, v), random.randint(0, 1)) for u, v in edges]
        return vertex_order, edges_page

    def mutate(individual):
        vertex_order, edges_page = individual
        # Mutacja permutacji wierzchołków
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(vertex_order)), 2)
            vertex_order[i], vertex_order[j] = vertex_order[j], vertex_order[i]
        # Mutacja przypisania stron krawędzi
        for k in range(len(edges_page)):
            if random.random() < mutation_rate:
                edge, page = edges_page[k]
                edges_page[k] = (edge, 1 - page)  # Zmień stronę
        return vertex_order, edges_page

    def crossover(parent1, parent2):
        # Krzyżowanie permutacji wierzchołków
        vertex_order1, edges_page1 = parent1
        vertex_order2, edges_page2 = parent2
        split = random.randint(1, len(vertex_order1) - 1)
        child_vertex_order = vertex_order1[:split] + [v for v in vertex_order2 if v not in vertex_order1[:split]]
        # Krzyżowanie przypisania stron krawędzi
        child_edges_page = [(edge, random.choice([p1, p2])) for ((edge, p1), (_, p2)) in zip(edges_page1, edges_page2)]
        return child_vertex_order, child_edges_page

    # Inicjalizacja populacji
    population = [generate_individual() for _ in range(population_size)]

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
        print(f"Generacja {generation + 1}: Najlepsza liczba przecięć = {best_fitness}, Ewaluacje = {evaluation_count}")

        # Selekcja rodziców (turniej)
        parents = population[:population_size // 2]

        # Tworzenie nowej populacji
        new_population = parents[:]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            new_population.append(mutate(child))

        population = new_population

    # Zwróć najlepsze rozwiązanie
    best_solution = min(population, key=lambda ind: count_edge_crossings(ind[0], ind[1]))
    return best_solution

# Przykład użycia:
if __name__ == "__main__":
    # Lista wierzchołków w ustalonej kolejności
    vertices = list(range(1, 21))

    # Lista krawędzi w postaci (u, v)
    edges = [
        (1, 2), (1, 6), (1, 5), (2, 8), (2, 3), (3, 10), (3, 4), (4, 12), (4, 5),
        (5, 14), (6, 7), (6, 15), (7, 16), (7, 8), (8, 17), (8, 9), (9, 17),
        (9, 10), (10, 18), (10, 11), (11, 18), (11, 12), (12, 19), (12, 13),
        (13, 19), (13, 14), (14, 20), (14, 15), (15, 20), (16, 20), (16, 17),
        (18, 19), (19, 20)
    ]

    # Uruchom algorytm ewolucyjny
    best_vertices, best_edges_page = evolutionary_algorithm(vertices, edges)
    #visualize(len(vertices), edges, (best_vertices, best_edges_page))
    edges2 = [(u-1, v-1) for (u, v) in edges]
    verts = [v-1 for v in best_vertices]
    pages_diff = [page*2-1 for _, page in best_edges_page]
    #assert all([orig == 0 and diff == -1 or orig == 1 and diff == 1 for orig, diff in zip([page for _, page in best_edges_page], pages_diff)])

    # Oblicz liczbę przecięć dla najlepszego rozwiązania
    best_crossings = count_edge_crossings(best_vertices, best_edges_page)

    print(f"Najlepszy układ wierzchołków: {best_vertices}")
    print(f"Najlepsze przypisanie stron krawędzi: {best_edges_page}")
    print(f"Liczba przecięć: {best_crossings}")
    visualize(len(verts), edges2, (verts, pages_diff))
