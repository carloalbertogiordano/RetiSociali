import igraph as ig

# Calcola il costo del seed set sommando il valore di c(u) per tutti i nodi
# nell'insieme S
def cost_seed_set(S, cost):
    return sum(cost(u) for u in S)

# Calcola la differenza tra la funzione obiettivo calcolata sull'insieme contenente v
# e la funzione obiettivo calcolata sull'insieme non contenente v
def marginal_gain(v, S, fi, G):
    S_with_v = S.union({v})
    return fi(G, S_with_v) - fi(G, S)

# Seleziona il nodo che ha un rapporto marginal gain / costo del nodo  migliore
def argmax(V, S, f, cost_function, g):
    return max(set(V)- S, key=lambda v: marginal_gain(v,S, f, g) / cost_function(v))



def cost_seeds_greedy(G, k, cost_fn, fi):
    #print(f"G: {G}, k: {k}, cost function: {cost_fn}, goal function: {fi}")
    # Insieme V dei nodi del grafo
    V = list(range(G.vcount()))
    
    # Insieme S_p=S_d=Empty
    S_p = set()
    S_d = set()

    # Continua fino a quando il costo del seed set S_d non è maggiore del budget k
    while cost_seed_set(S_d, cost_fn) <= k:
        #print(f"Costo di S {cost_seed_set(S_d, cost_fn)}, k={k}")
        u = argmax(V, S_d, fi, cost_fn, G)
        
        S_p = S_d.copy()
        S_d.add(u)

        print(f"Dimensione S_p: {len(S_p)}")
    return list(S_p)