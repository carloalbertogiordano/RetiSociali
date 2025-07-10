import igraph as ig
import random
from enum import Enum
import math
from cost_functions import base as cost_func_base
from Graph.utils import plot_utils


def marginal_gain(v, S, fi):
    """
    Compute the marginal gain of adding node v to the seed set S.

    :param v: The node to be added.
    :param S: The current seed set.
    :param fi: The objective function to evaluate the seed set.
    :param G: The graph on which the computation is performed.
    :return: The marginal gain, i.e., fi(S ∪ {v}) - fi(S).
    """
    S_with_v = S.union({v})
    return fi(S_with_v) - fi(S)

def get_subgraph(graph: ig.Graph, number: int):
    """
    Extract a subgraph with a specified number of nodes using BFS.

    :param graph: The original graph.
    :param number: The desired number of nodes in the subgraph.
    :return: An induced subgraph containing the selected nodes.
    """
    start_node = random.choice(range(graph.vcount()))
    visited = set()
    queue = [start_node]

    while queue and len(visited) < number:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            neighbors = graph.neighbors(current)
            for n in neighbors:
                if n not in visited and n not in queue:
                    queue.append(n)

    return graph.induced_subgraph(list(visited))

class Graph:

    class GoalFuncType(Enum):
        F1 = 1
        F2 = 2
        F3 = 3

    def __init__(self, file_path: str, save_path: str, cost_func :cost_func_base.CostFunction.calculate_cost, is_sub_graph=True, sub_graph_dim=100, info_name=""):
        """
        Initialize a Graph object with a graph loaded from a file.

        :param file_path: Path to the edge list file defining the graph.
        :param save_path: Directory path to save output plots.
        :param cost_func: The cost function to be applied
        :param is_sub_graph: If True, use a subgraph instead of the full graph.
        :param sub_graph_dim: Number of nodes in the subgraph if is_sub_graph is True.
        """
        self.full_graph = ig.Graph.Read_Edgelist(file_path, directed=False)
        self.full_graph.vs["name"] = list(range(self.full_graph.vcount()))
        self.graph = self.full_graph

        # Cache function results for f2 and f3
        self.fun_cache = {}
        # Cache neighborhood results
        self._neighbor_cache = {}

        if is_sub_graph:
            self.graph = get_subgraph(self.full_graph, sub_graph_dim)

        if not isinstance(cost_func, cost_func_base.CostFunction):
            raise ValueError(f"Cost func is {type(cost_func)}, expected subclass of CostFunction")
        self.cost_fun = cost_func.calculate_cost

        self.budget = self.calculate_budget(self.cost_fun)
        if self.budget > self.graph.vcount():
            print(f"WARN: BUDGET HIGH GOT {self.budget}, NODE NUMBER IS {self.graph.vcount()}")

        #self.budget = budget
        self.save_path = save_path
        self.info_name=info_name
        self.seedSet = None
        self.cascade = None

    def calculate_budget(self, cost_fun):
        budget = 0

        degrees = self.graph.degree()
        degrees_dict = {v.index: degree for v, degree in zip(self.graph.vs, degrees)}
        top5 = sorted(degrees_dict.items(), key=lambda item: item[1], reverse=True)[:5]
        top5_nomi = [(self.graph.vs[i]["name"], degree) for i, degree in top5]
        for name, val in top5_nomi:
            budget += cost_fun(node_label=name, igraph=self.graph, graph=self)
        return budget

    def get_neighbors(self, v):
        """
        Returns cached neighbors of node v by name.
        """
        if v not in self._neighbor_cache:
            vertex = self.graph.vs.find(name=v)
            self._neighbor_cache[v] = [nbr["name"] for nbr in vertex.neighbors()]
        return self._neighbor_cache[v]

    def get_graf(self):
        return self.graph

    def set_graph(self, graph: ig.Graph):
        self.graph = graph
        self.full_graph = graph
        self.fun_cache = {}
        self._neighbor_cache = {}

    def cost_seed_set(self, S, cost):
        """
        Calculate the total cost of the seed set by summing the cost of each node.

        :param S: The seed set, a collection of nodes.
        :param cost: A function that returns the cost of a node.
        :return: The total cost of the seed set.
        """
        return sum(cost(node_label=u, igraph=self.graph, graph=self) for u in S)

    # Seleziona il nodo che ha un rapporto marginal gain / costo del nodo  migliore
    def argmax(self, V, S, f, **kwargs):
        """
        Select the node with the highest marginal gain to cost ratio.

        :param V: The set of all nodes in the graph.
        :param S: The current seed set.
        :param f: The objective function to evaluate the seed set.
        :param cost_function: The function to compute the cost of a node.
        :return: The node with the best marginal gain / cost ratio.
        """
        return max(set(V) - S, key=lambda v: marginal_gain(v, S, f) / self.cost_fun(node_label=v, igraph=self.graph, graph=self))

    def f1(self, S):
        """
                Objective function F1: Sum of minimum between neighbors in S and threshold.

                :param S: The seed set.
                :return: The total value of the objective function across all nodes.
                """

        key = frozenset(S)
        if key in self.fun_cache and False:
            return self.fun_cache[key]

        total = 0
        for v in self.graph.vs["name"]:
            neighbors_in_S = set(self.get_neighbors(v)).intersection(S) # set(nbr["name"] for nbr in self.graph.vs.find(name=v).neighbors()).intersection(S)
            threshold = math.ceil(self.graph.vs.find(name=v).degree() / 2)
            total += min(len(neighbors_in_S), threshold)
        return total

    def f2(self, S):
        """
        Objective function F2: Sum of decremented thresholds based on neighbors in S.

        :param S: The seed set.
        :return: The total value of the objective function across all nodes.
        """
        key = frozenset(S)
        if key in self.fun_cache and False:
            return self.fun_cache[key]

        total = 0
        for v in self.graph.vs["name"]:
            neighbors_in_S = list(set(self.get_neighbors(v)).intersection(S)) #list(set(nbr["name"] for nbr in self.graph.vs.find(name=v).neighbors()).intersection(S))
            d_v = self.graph.vs.find(name=v).degree()
            threshold = math.ceil(d_v / 2)
            for i in range(1, len(neighbors_in_S) + 1):
                total += max(threshold - i + 1, 0)

        self.fun_cache[key] = total
        return total

    def f3(self, S):
        """
        Objective function F3: Weighted sum based on neighbors in S and degree.

        :param S: The seed set.
        :return: The total value of the objective function across all nodes.
        """

        key = frozenset(S)
        if key in self.fun_cache and False:
            return self.fun_cache[key]

        total = 0
        for v in self.graph.vs["name"]:
            neighbors_in_S = list(set(self.get_neighbors(v)).intersection(S)) # list(set(nbr["name"] for nbr in self.graph.vs.find(name=v).neighbors()).intersection(S))
            d_v = self.graph.vs.find(name=v).degree()
            threshold = math.ceil(d_v / 2)
            for i in range(1, len(neighbors_in_S) + 1):
                denom = d_v - i + 1
                term = (threshold - i + 1) / denom if denom != 0 else 0
                total += max(term, 0)
        return total

    def csg(self, select_goal_fun=GoalFuncType.F1):
        """
        Compute the seed set using the Cost-Sensitive Greedy (CSG) algorithm.

        :param select_goal_fun: The objective function type (F1, F2, F3).
        """

        match select_goal_fun:
            case Graph.GoalFuncType.F1:
                obj_fun = self.f1
            case Graph.GoalFuncType.F2:
                obj_fun = self.f2
            case Graph.GoalFuncType.F3:
                obj_fun = self.f3
            case _:
                return

        # Calculate the seed set
        V = self.graph.vs["name"]
        # print(f"Node set: {V}")
        # Insieme S_p=S_d=Empty
        S_p = set()
        S_d = set()

        # Continua fino a quando il costo del seed set S_d non è maggiore del budget k
        while self.cost_seed_set(S_d, self.cost_fun) <= self.budget:
            print(f" Cost seed set: {self.cost_seed_set(S_d, self.cost_fun)}, max is {self.budget}")
            # print(f"Costo di S {self.cost_seed_set(S_d, cost_fn)}, k={k}")
            u = self.argmax(V, S_d, obj_fun)
            print(f"Selected node: {u}")

            S_p = S_d.copy()
            S_d.add(u)
            print(f"Seed set (d): {S_d}, DIM{len(S_d)}")
        print(f"Seed set (p): {S_p}")
        self.seedSet = list(S_p)

    def wtss(self):
        """
        Compute the seed set using the Weak-Tie Seed Selection (WTSS) algorithm.
        """

        # select_threshold = 1 #TODO: Define functions

        # thresholds = {v.index: max(1, self.graph.degree(v) // 2) for v in self.graph.vs}

        V = self.graph.vs["name"]

        U = set(V)
        S = set()
        thresholds = {v: max(1, self.graph.vs.find(
            name=v).degree() // 2) for v in V}

        # Stato dinamico
        delta = {v: self.graph.vs.find(name=v).degree()
                 for v in V}  # δ(v) -> grado
        k = {v: thresholds[v] for v in V}  # k(v) -> threshold
        neighbors = {v: set(x["name"] for x in self.graph.vs.find(
            name=v).neighbors()) for v in V}  # neighbors(v) -> Vicini

        while U:
            # Case 1: nodo già attivabile
            activated = [v for v in U if k[v] == 0]
            if activated:
                v = activated[0]
                for u in neighbors[v]:
                    if u in U:
                        k[u] = max(0, k[u] - 1)
            else:
                # Case 2: nodo non attivabile, ma troppo debole
                weak = [v for v in U if delta[v] < k[v]]
                if weak:
                    v = weak[0]
                    S.add(v)
                    for u in neighbors[v]:
                        if u in U:
                            k[u] = max(0, k[u] - 1)
                else:
                    # Case 3: scegli nodo ottimale da eliminare
                    def score(u):
                        if delta[u] == 0:
                            return float('inf')
                        return self.cost_fun(label=u, igraph=self.graph, graph=self) * k[u] / (delta[u] * (delta[u] + 1))

                    v = min(U, key=score)

            # Rimuovi v da U e aggiorna grafo dinamico
            for u in neighbors[v]:
                if u in U:
                    delta[u] -= 1
                    neighbors[u].discard(v)
            U.remove(v)
        self.seedSet = S
        print(f"Len seed set {len(self.seedSet)}")

    def calc_seed_set(self, method, *args, **kwargs):
        """
        Calculate the seed set using the specified method.

        :param method: The algorithm to use ('csg' or 'wtss').
        :param args: Positional arguments for the method.
        :param kwargs: Keyword arguments for the method.
        """
        if method == 'csg':
            self.csg(**kwargs)
        elif method == 'wtss':
            self.wtss(**kwargs)
        else:
            raise ValueError("Method must be csg or wtss")

    def get_seed_set(self):
        """
        Retrieve the computed seed set.

        :return: The seed set as a list or set, or None if not computed.
        """
        return self.seedSet

    def calc_majority_cascade(self):
        """
        Compute the majority cascade starting from the seed set.

        Requires the seed set to be computed first.
        """
        if self.seedSet is None:
            return

        V = set(self.graph.vs["name"])
        cascade = []

        prev_influenced = set(self.seedSet)
        cascade.append(prev_influenced.copy())

        while True:
            new_influenced = prev_influenced.copy()

            for v in V - prev_influenced:
                vertex = self.graph.vs.find(name=v)
                # prendo la lista di name dei vicini
                neighbor_names = self.get_neighbors(v) # [nbr["name"] for nbr in vertex.neighbors()]

                # conto quanti di questi sono già influenzati
                #active_neighbors = sum(1 for nbr in neighbor_names if nbr in prev_influenced)
                active_neighbors = sum(1 for nbr in self.get_neighbors(v) if nbr in prev_influenced)

                threshold = math.ceil(vertex.degree() / 2)
                if active_neighbors >= threshold:
                    new_influenced.add(v)

            cascade.append(new_influenced.copy())

            if new_influenced == prev_influenced:
                break
            prev_influenced = new_influenced

        self.cascade = cascade

    def get_majority_cascade(self):
        """
        Retrieve the computed majority cascade.

        :return: A list of sets representing the cascade steps, or None if not computed.
        """
        return self.cascade

    def print_majority_cascade(self):
        """
        Print the majority cascade steps with sorted node lists.
        """
        for t, influenced in enumerate(self.cascade):
            print(
                f"Inf[S, {t}] = {sorted(influenced)}; |Inf[S, {t}]| = {len(influenced)}\n")

    def save_plot(self, filename: str):
        """
        Save a visual representation of the graph to a file.

        :param filename: The name of the file to save the plot (without path).
        """
        plot_utils.save_plot(self, filename)


    def plot_majority_cascade(self):
        """
        Plot the evolution of the majority cascade over time.
        """
        plot_utils.plot_majority_cascade(self)


    def dyn_plot_cascade(self):
        """
        Generates a sequence of network plots representing the cumulative activation of nodes
        over the steps of a cascade process, and compiles them into an animated GIF.

        The cascade is visualized using color intensity to indicate activation time, and
        non-activated nodes are shown in gray. All output images and the final GIF are saved
        under a structured directory inside `self.save_path`.

        Output structure:
        └── self.save_path/
            └── plots/
                └── plot_cascade/
                    ├── images/
                    │   ├── step_00.png
                    │   ├── step_01.png
                    │   └── ...
                    └── diffusione.gif
        """
        plot_utils.dyn_plot_cascade(self)
