import sys
from loader import load_file
import pandas as pd
import warnings
import os

RESULTS_PREFIX = 'test_results'

def statistics(graf, file_name, algo, actual_minimum=None):
    warnings.filterwarnings("ignore")
    print('Testowanie grafu: ', graf)

    graph_file = sys.argv[1] if len(sys.argv) > 1 else f'examples/{graf}.txt'
    vertices, edges = load_file(graph_file)

    results = []

    for i in range(100):
        # Uruchom algorytm ewolucyjny
        _, fitness = algo(vertices, edges, stfu=True)
        print(f'{i}: {fitness} przecięć')
        results.append(fitness)

    folder = f'{RESULTS_PREFIX}/{file_name}'
    os.makedirs(folder, exist_ok=True)

    file_prefix = f'{folder}/{graf}'
    df = pd.DataFrame(results, columns=['Liczba przecięć'])
    df.to_csv(f'{file_prefix}_results.csv')
    df.describe().to_csv(f'{file_prefix}_statistics.csv')
    df.value_counts().to_csv(f'{file_prefix}_counts.csv')
    print(df.describe())
    print(f'success_rate: { df.value_counts().get(actual_minimum, 0) / df.shape[0] * 100 } %')
    print()

grafy = [('dececahedron', 0), ('z_dokumentu',18)]

def all_statistics(file_name, algo):
    print('Testowanie algorytmu: ', file_name)
    for (graf, minimum) in grafy:
        statistics(graf, file_name, algo, minimum)
    print('\n')