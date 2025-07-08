import os
from Graph.utilFuncs.goal_function import *
from Graph.utilFuncs.cost_function import *
import matplotlib.pyplot as plt


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
    return max(set(V) - S, key=lambda v: marginal_gain(v, S, f, g) / cost_function(v))

def get_subgraph(graph: ig.Graph, number: int):
    return graph.induced_subgraph(list(range(number)))


class Graph:

    def __init__(self, filePath: str, budget: int, save_path:str, is_sub_graph=True, sub_graph_dim=100):
        self.full_graph = ig.Graph.Read_Edgelist(filePath, directed=False)
        self.graph = self.full_graph
        
        if is_sub_graph :
            self.graph = get_subgraph(self.full_graph, sub_graph_dim)

        if budget > self.graph.vcount():
            budget = self.graph.vcount()

        self.budget = budget
        self.save_path = save_path
        self.seedSet = None
        self.cascade = None

    def calc_seed_set(self, select_cost=1, select_goal_fun=1):
        if select_cost > 3 or select_cost < 0 or select_goal_fun > 3 or select_goal_fun < 0:
            return

        match select_cost:
            case 1:
                cost_fun = cost1
            case 2:
                cost_fun = cost2
            case 3:
                cost_fun = cost3
            case _:
                return

        match select_goal_fun:
            case 1:
                obj_fun = f1
            case 2:
                obj_fun = f2
            case 3:
                obj_fun = f3
            case _:
                return


        #Calculate the seed set
        V = list(range(self.graph.vcount()))

        # Insieme S_p=S_d=Empty
        S_p = set()
        S_d = set()

        # Continua fino a quando il costo del seed set S_d non è maggiore del budget k
        while cost_seed_set(S_d, cost_fun) <= self.budget:
            # print(f"Costo di S {cost_seed_set(S_d, cost_fn)}, k={k}")
            u = argmax(V, S_d, obj_fun, cost_fun, self.graph)

            S_p = S_d.copy()
            S_d.add(u)
            print(f"Dimensione S_p: {len(S_p)}")

        self.seedSet = list(S_p)

    def get_seed_set(self): return self.seedSet

    def calc_majority_cascade(self):
        if self.seedSet is None: return #Error

        # Calc majority cascade
        V = set(range(self.graph.vcount()))
        cascade = []  # lista Inf[S,0], Inf[S,1], ...

        prev_influenced = set(self.seedSet)
        cascade.append(prev_influenced.copy())

        while True:
            new_influenced = prev_influenced.copy()

            for v in V - prev_influenced:
                neighbors = self.graph.neighbors(v)
                active_neighbors = sum(
                    1 for u in neighbors if u in prev_influenced)
                if active_neighbors >= math.ceil(self.graph.degree(v) / 2):
                    new_influenced.add(v)

            cascade.append(new_influenced.copy())

            if new_influenced == prev_influenced:
                break
            prev_influenced = new_influenced

        self.cascade = cascade

    def get_majority_cascade(self): return self.cascade

    def print_majority_cascade(self):
        for t, influenced in enumerate(self.cascade):
            print(f"Inf[S, {t}] = {sorted(influenced)}")


    def save_plot(self, filename:str):

        layout = self.graph.layout("fr")  # Fruchterman-Reingold

        ig.plot(
            self.graph,
            target=self.save_path+'images/'+filename,
            layout=layout,
            vertex_size=10,
            vertex_label=None,
            bbox=(2000, 2000),
        )

    def plot_majority_cascade(self):
        x = list(range(len(self.cascade)))  # Indici: 0, 1, 2, ...
        y = [len(s) for s in self.cascade]  # Cardinalità di ogni set

        plt.plot(x, y, marker='o')
        plt.xlabel("Indice del passo nella cascata")
        plt.ylabel("Numero di nodi (len del set)")
        plt.title("Evoluzione della cascata")
        plt.grid(True)
        plt.show()


