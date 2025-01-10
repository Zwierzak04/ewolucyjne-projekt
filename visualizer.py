import networkx as nx
import matplotlib.pyplot as plt
from shapely import LineString
import itertools
import math
import numpy as np

def set_invalid(graph, edge_start, edge_end):
    '''
    Ustawia kolor krawędzi na czerwony. W ten sposób oznaczam krawędzie które się na siebie nakładają.
    '''
    view = graph.get_edge_data(edge_start, edge_end)
    view['color'] = 'red'
    view['weight'] = 2

def visualize(vertices, edges, positions=None):
    '''
    Rysuje graf o podanych współrzędnych i zaznacza wszystkie przecięcia krawędzi.
    '''
    
    G: nx.graph.Graph = nx.Graph().to_undirected()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    # dla mojej własnej informacji. jeśli graf jest planarny to na pewno da się go narysować bez przecięć
    # jeden z testowych właśnie jest planarny
    # nie mówi to nic istotnego poza tym że nasz algorytm genetyczny jest do niczego zazwyczaj
    print("czy jest planarny: ", nx.is_planar(G))

    if positions is None:
        positions = nx.spring_layout(G)

    edge_pos = [LineString((positions[e[0]], positions[e[1]])) for e in edges]
    inte = [[],[]]
    intersection_count = 0.0
    
    # FIXME: coś w tych przecięciach się chyba jeszcze jebie
    for i1, i2 in itertools.combinations(range(len(edges)), 2):
        intersection = edge_pos[i1].intersection(edge_pos[i2])

        # brak przecięcia
        if intersection.is_empty:
            continue

        # linie na siebie nachodzą
        if(intersection.geom_type != 'Point'):
            set_invalid(G, *edges[i1])
            set_invalid(G, *edges[i2])
            intersection_count = math.inf
            continue

        # pomijamy krawędzie, które mają wspólne wierzchołki
        # jeśli na siebie nie nachodzą to nie ma sensu ich sprawdzać
        if set(edges[i1]) & set(edges[i2]):
            continue

        if not intersection.is_empty:
            intersection_count += 1
            inte[0].append(intersection.x)
            inte[1].append(intersection.y)
            #print(edges[i1], edges[i2], intersection)
    
    edge_colors = [G[u][v].get('color', 'black') for u,v in edges]
    edge_weights = [G[u][v].get('weight', 1) for u,v in edges]

    nx.draw(G, pos=positions, node_size=100, font_size=20, edge_color=edge_colors, width=edge_weights, labels={e: e+1 for e in vertices})
    plt.scatter(inte[0], inte[1], color='red', zorder=20)
    plt.legend([f'Liczba przecięć:  {intersection_count}'])
    plt.show()

def visualize_krzywe(n_verts, edges, solution):
    order, pages = solution
    rev_order = np.zeros(n_verts, dtype=int)

    _, ax = plt.subplots(figsize=(20, 10))
    for (i, vert) in enumerate(order):
        rev_order[vert] = i
        ax.plot(i, 0, 'o', markersize=15, color='cyan', zorder=2)
        ax.text(i, 0, f'{vert+1}', fontsize=10, ha='center', va='center', zorder=3)

    max_radius = 0
    total_crossings = 0

    other_edges = []
    for i, e in enumerate(edges):
        x1 = rev_order[e[0]]
        x2 = rev_order[e[1]]
        mid_x = (x1 + x2) / 2
        radius = abs(x2 - x1) / 2

        theta = np.linspace(0, np.pi, 100)
        max_radius = max(max_radius, radius)
        arc_x = mid_x + radius * np.cos(theta)
        arc_y = radius * np.sin(theta)

        for other_i, (other_mid, other_r) in enumerate(other_edges):
            if(pages[i] != pages[other_i]):
                continue

            if abs(mid_x - other_mid) < 0.1:
                continue
                
            x = (mid_x**2 - other_mid**2 - radius**2 + other_r**2) / (2*(mid_x - other_mid))
            y = radius**2 - (x - mid_x)**2
            if y <= 0:
                continue

            total_crossings += 1
            plt.plot(x, pages[i]*np.sqrt(y), 'o', markersize=5, color='red', zorder=5)

        other_edges.append((mid_x, radius))
        plt.plot(arc_x, pages[i]*arc_y, color='gray', zorder=1)

    plt.xlim(-1, n_verts+1)
    plt.ylim(-max_radius, max_radius)

    plt.legend([f'Liczba przecięć: {total_crossings}'])
    plt.axis('off')
    plt.show()