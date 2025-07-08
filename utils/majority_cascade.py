import math
import igraph as ig

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
