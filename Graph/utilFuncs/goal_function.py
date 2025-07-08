import math


def f1(G, S):
    total = 0
    for v in range(G.vcount()):
        neighbors_in_S = set(G.neighbors(v)).intersection(S)
        threshold = math.ceil(G.degree(v) / 2)
        total += min(len(neighbors_in_S), threshold)
    return total


def f2(G, S):
    total = 0
    for v in range(G.vcount()):
        neighbors_in_S = list(set(G.neighbors(v)).intersection(S))
        d_v = G.degree(v)
        threshold = math.ceil(d_v / 2)
        for i in range(1, len(neighbors_in_S) + 1):
            total += max(threshold - i + 1, 0)
    return total


def f3(G, S):
    total = 0
    for v in range(G.vcount()):
        neighbors_in_S = list(set(G.neighbors(v)).intersection(S))
        d_v = G.degree(v)
        threshold = math.ceil(d_v / 2)
        for i in range(1, len(neighbors_in_S) + 1):
            denom = d_v - i + 1
            term = (threshold - i + 1) / denom if denom != 0 else 0
            total += max(term, 0)
    return total
