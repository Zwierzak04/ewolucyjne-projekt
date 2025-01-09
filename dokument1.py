import numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as patches

# data = [vertex order | edge pages]
# edge positions = [-1 | 1]

def visualize(n_verts, edges, solution):
    order, pages = solution
    print(order)
    
    rev_order = np.zeros(n_verts, dtype=int)

    fig, ax = plt.subplots(figsize=(20, 10))
    for (i, vert) in enumerate(order):
        rev_order[vert] = i
        #print(i, vert)
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
            
            #print(mid_x, other_mid)
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

def gen_random_solution(n_verts, edges):
    order = np.random.permutation(n_verts)
    pages = np.random.choice([-1, 1], len(edges))
    return order, pages
