def load_file(path):
    '''
    Wczytuje graf z moim formatem.
    
    TODO: ogarnąć wczytywanie i zapisywanie grafów w NetworkX
    '''

    with open(path, 'r') as f:
        vertices = list(range(int(f.readline().strip())))
        edges = set() # żeby nie powtarzały się krawędzie po 2 razy
        for line in f:
            start, *connections = map(lambda x: int(x)-1, line.strip().split(','))
            edges = edges.union({frozenset((start, end)) for end in connections})
    print(len(edges))
    return vertices, list(map(lambda e: list(e), edges))