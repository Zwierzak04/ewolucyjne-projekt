import networkx as nx
import matplotlib.pyplot as plt
from shapely import LineString
import itertools
import math

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
            print(edges[i1], edges[i2], intersection)
    
    edge_colors = [G[u][v].get('color', 'black') for u,v in edges]
    edge_weights = [G[u][v].get('weight', 1) for u,v in edges]

    nx.draw(G, pos=positions, node_size=100, font_size=20, edge_color=edge_colors, width=edge_weights, labels={e: e+1 for e in vertices})
    plt.scatter(inte[0], inte[1], color='red', zorder=20)
    plt.legend([f'Liczba przecięć:  {intersection_count}'])
    plt.show()