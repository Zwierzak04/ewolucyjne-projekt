
from loader import load_file
from visualizer import visualize
#from evo import liczba_przeciec, search_best#search_fancy_mutation as search_best
#from evo2 import liczba_przeciec, search_best
from evo3 import search_best
import numpy as np
import sys

import networkx as nx

def main():
    '''
    Punkt wejscia. wczytuje graf, tworzy graf NetworkX, znajduje najlepsze rozwiazanie i wizualizuje je.
    '''

    graph_file = sys.argv[1] if len(sys.argv) > 1 else 'examples/dececahedron.txt'
    #'examples/dececahedron.txt'
    #'examples/z_dokumentu.txt'
    vertices, edges = load_file(graph_file)
    #print(vertices, edges)

    G: nx.graph.Graph = nx.Graph().to_undirected()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    
    # nieużywane ale generuje współrzędne które całkiem nieźle wyglądają
    # być może dałoby się użyć tego jako punkt startowy dla algorytmu
    positions = nx.spring_layout(G) 

    n_verts = len(vertices)
    szukane = search_best(n_verts, edges, 4, 50)#search_best(n_verts, edges)#, start=np.reshape([positions[i] for i in range(n_verts)], (1, n_verts))) # , start=[positions[i] for i in range(n_verts)] - na razie chyba to nie działa
    visualize(vertices, edges, positions=np.reshape(szukane, (-1, 2)))

if __name__ == '__main__':
    main()