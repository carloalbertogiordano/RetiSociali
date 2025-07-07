import igraph as ig
import time
import plotly.io as pio
import os
from utils.graphUtils import *
from csg import cost_seeds_greedy
from utils.goal_function import *


def main():
    filepath = os.path.join(os.path.dirname(__file__), 'graphs/facebook_data/facebook_combined.txt')
    g = ig.Graph.Read_Edgelist(filepath, directed=False)

    print(f"Grafo caricato: {g.vcount()} nodi, {g.ecount()} archi.")
        
    cost_seeds_greedy(get_subgraph(g, 100), 4039, cost1, f1)
    
    save_plot(g, "graphs/images/test2.png")
    
if __name__ == '__main__':
    main()
