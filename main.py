import igraph as ig
import os
from utils.graphUtils import *
from csg import cost_seeds_greedy
from utils.goal_function import *
from utils.cost_function import *
from utils.plotUtils import *
from utils.majority_cascade import *

def main():
    filepath = os.path.join(os.path.dirname(__file__),
                            'graphs/facebook_data/facebook_combined.txt')
    g = ig.Graph.Read_Edgelist(filepath, directed=False)

    print(f"Grafo caricato: {g.vcount()} nodi, {g.ecount()} archi.")

    seed_set = cost_seeds_greedy(get_subgraph(g, 300), 300, cost1, f1)
    seed_set.sort()
    print(f"Seed set: {seed_set}")

    cascade = majority_cascade(get_subgraph(g, 300), seed_set)
    print(cascade)
    print(type(cascade))
    for t, influenced in enumerate(cascade):
        print(f"Inf[S, {t}] = {sorted(influenced)}")

    # save_plot(g, "graphs/images/test2.png")


if __name__ == '__main__':
    main()
