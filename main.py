import igraph as ig
import time
import plotly.io as pio
import os
import math
from utils.graphUtils import *
from csg import cost_seeds_greedy
from utils.goal_function import *
from utils.cost_function import *


def majority_cascade(G, S):
    """
    G: igraph.Graph
    S: insieme di nodi iniziali (seed set) come lista di indici
    ritorna: lista di insiemi Inf[S, r] per ogni r
    """
    V = set(range(G.vcount()))
    cascade = []  # lista Inf[S,0], Inf[S,1], ...

    prev_influenced = set(S)
    cascade.append(prev_influenced.copy())

    while True:
        new_influenced = prev_influenced.copy()

        for v in V - prev_influenced:
            neighbors = G.neighbors(v)
            active_neighbors = sum(
                1 for u in neighbors if u in prev_influenced)
            if active_neighbors >= math.ceil(G.degree(v) / 2):
                new_influenced.add(v)

        cascade.append(new_influenced.copy())

        if new_influenced == prev_influenced:
            break
        prev_influenced = new_influenced

    return cascade


def main():
    filepath = os.path.join(os.path.dirname(__file__),
                            'graphs/facebook_data/facebook_combined.txt')
    g = ig.Graph.Read_Edgelist(filepath, directed=False)

    print(f"Grafo caricato: {g.vcount()} nodi, {g.ecount()} archi.")

    seed_set = cost_seeds_greedy(get_subgraph(g, 300), 300, cost1, f1)
    seed_set.sort()
    print(f"Seed set: {seed_set}")

    cascade = majority_cascade(get_subgraph(g, 300), seed_set)
    # print(cascade)
    # print(type(cascade))

    for t, influenced in enumerate(cascade):
        print(f"Inf[S, {t}] = {sorted(influenced)}")

    # save_plot(g, "graphs/images/test2.png")


if __name__ == '__main__':
    main()
