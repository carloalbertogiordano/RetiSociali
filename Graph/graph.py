import igraph as ig
import random
from enum import Enum
import math
from cost_functions import base as cost_func_base
from Graph.utils import plot_utils
import json
import os

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

class GoalFuncType(Enum):
    F1 = 1
    F2 = 2
    F3 = 3

class Graph:

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
        self._fun_cache = {}
        # Cache neighborhood results
        self._neighbor_cache = {}
        # Cache degree of a node
        self._degree_cache = {}
        # Cache node list
        self._node_list_cache = ()
        # Define if cache is to be used
        self._is_cache = True

        if is_sub_graph:
            self.graph = get_subgraph(self.full_graph, sub_graph_dim)

        if not isinstance(cost_func, cost_func_base.CostFunction):
            raise ValueError(f"Cost func is {type(cost_func)}, expected subclass of CostFunction")
        self.cost_fun = cost_func.calculate_cost

        self.budget = 0

        #self.budget = budget
        self.save_path = save_path
        self.info_name=info_name
        self.seedSet = None
        self.cascade = None

    def get_budget(self):
        if self.budget == 0 : self.budget = self.calculate_budget(self.cost_fun)
        return self.budget

    def disable_cache(self):
        self._is_cache = False

    def enable_cache(self):
        self._is_cache = True

    def is_cache(self):
        return self._is_cache

    def get_node_num(self):
        """
        Returns number of vertexes in the graph

        :return: The number of nodes in the graph
        """
        return self.graph.vcount()

    def calculate_budget(self, cost_fun):
        budget = 0
        node_cost = [cost_fun(node_label=n, igraph=self.graph, graph=self) for n in self.get_nodes_list()]
        node_cost = sorted(node_cost, reverse=True)
        print(f"First 5 nodes with higher cost: {node_cost[:5]}; {self.info_name}")
        for i in range(0,5):
            budget += node_cost[i]

        return budget

    def get_neighbors(self, v):
        """
        Returns cached neighbors of node v by name.
        """
        if v not in self._neighbor_cache or not self.is_cache():
            vertex = self.graph.vs.find(name=v)
            self._neighbor_cache[v] = [nbr["name"] for nbr in vertex.neighbors()]
        return self._neighbor_cache[v]

    def get_nodes_list(self):
        """
        Return a list of all node names in the graph, in order
        """
        if len(self._node_list_cache) == 0 or not self.is_cache():
            nl = []
            for v in self.graph.vs:
                nl.append(v["name"])
            self._node_list_cache = nl

        return self._node_list_cache

    def get_degree(self, v):
        if v not in self._degree_cache or not self.is_cache():
            self._degree_cache[v] = self.graph.vs.find(name=v).degree()
        return self._degree_cache[v]

    def get_graf(self):
        return self.graph

    def set_graph(self, graph: ig.Graph):
        self.graph = graph
        self.full_graph = graph
        # Invalidate caches
        self._fun_cache = {}
        self._neighbor_cache = {}
        self._degree_cache = {}
        self._node_list_cache = ()
        self.budget = 0

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
        if key in self._fun_cache:
            return self._fun_cache[key]

        total = 0
        for v in self.get_nodes_list():
            neighbors_in_S = set(self.get_neighbors(v)).intersection(S)
            threshold = math.ceil(self.get_degree(v) / 2)
            total += min(len(neighbors_in_S), threshold)
        self._fun_cache[key] = total
        return total

    def f2(self, S):
        """
        Objective function F2: Sum of decremented thresholds based on neighbors in S.

        :param S: The seed set.
        :return: The total value of the objective function across all nodes.
        """
        key = frozenset(S)
        if key in self._fun_cache:
            return self._fun_cache[key]

        total = 0
        for v in self.get_nodes_list():
            neighbors_in_S = list(set(self.get_neighbors(v)).intersection(S))
            d_v = self.get_degree(v)
            threshold = math.ceil(d_v / 2)
            for i in range(1, len(neighbors_in_S) + 1):
                total += max(threshold - i + 1, 0)

        self._fun_cache[key] = total
        return total

    def f3(self, S):
        """
        Objective function F3: Weighted sum based on neighbors in S and degree.

        :param S: The seed set.
        :return: The total value of the objective function across all nodes.
        """

        key = frozenset(S)
        if key in self._fun_cache:
            return self._fun_cache[key]

        total = 0
        for v in self.get_nodes_list():
            neighbors_in_S = list(set(self.get_neighbors(v)).intersection(S))
            d_v = self.get_degree(v)
            threshold = math.ceil(d_v / 2)
            for i in range(1, len(neighbors_in_S) + 1):
                denom = d_v - i + 1
                term = (threshold - i + 1) / denom if denom != 0 else 0
                total += max(term, 0)
        self._fun_cache[key] = total
        return total

    def csg(self, select_goal_fun=GoalFuncType.F1):
        """
        Compute the seed set using the Cost-Sensitive Greedy (CSG) algorithm.

        :param select_goal_fun: The objective function type (F1, F2, F3).
        """

        match select_goal_fun:
            case GoalFuncType.F1:
                obj_fun = self.f1
            case GoalFuncType.F2:
                obj_fun = self.f2
            case GoalFuncType.F3:
                obj_fun = self.f3
            case _:
                return

        # Calculate the seed set
        V = self.get_nodes_list()
        # print(f"Node set: {V}")
        # Insieme S_p=S_d=Empty
        S_p = set()
        S_d = set()

        # Continua fino a quando il costo del seed set S_d non è maggiore del budget k
        seed_set_cost = self.cost_seed_set(S_d, self.cost_fun)
        while seed_set_cost <= self.get_budget():
            print(f"Costo di S {seed_set_cost}, budget={self.get_budget()}")
            u = self.argmax(V, S_d, obj_fun)

            S_p = S_d.copy()
            S_d.add(u)
            seed_set_cost = self.cost_seed_set(S_d, self.cost_fun)
        self.seedSet = list(S_p)

    def wtss(self, **kwargs):
        """
        Compute the seed set using the Weak-Tie Seed Selection (WTSS) algorithm.

        :param kwargs are ignored
        """
        V = self.get_nodes_list()
        U = set(V)
        S = set()
        thresholds = {v: max(1, self.get_degree(v) // 2) for v in V}

        # Stato dinamico
        delta = {v: self.get_degree(v) for v in V}  # δ(v) -> grado
        k = {v: thresholds[v] for v in V}  # k(v) -> threshold
        neighbors = {v: set(x for x in self.get_neighbors(v)) for v in V}  # neighbors(v) -> Vicini

        iteration = 0
        while U:
            iteration += 1

            # Caso 1: nodo già attivabile
            activated = [v for v in U if k[v] == 0]
            if activated:
                v = activated[0]
                for u in neighbors[v]:
                    if u in U:
                        old_k = k[u]
                        k[u] = max(0, k[u] - 1)
            else:
                # Caso 2: nodo non attivabile, ma troppo debole
                weak = [v for v in U if delta[v] < k[v]]
                if weak:
                    v = weak[0]
                    S.add(v)
                    for u in neighbors[v]:
                        if u in U:
                            old_k = k[u]
                            k[u] = max(0, k[u] - 1)
                else:
                    # Caso 3: scegli nodo ottimale da eliminare
                    def score(u_in):
                        if delta[u_in] == 0:
                            return float('inf')
                        cost = self.cost_fun(node_label=u_in, igraph=self.graph, graph=self)
                        result = cost * k[u_in] / (delta[u_in] * (delta[u_in] + 1))
                        return result

                    v = max(U, key=score)

            # Aggiornamento del grafo
            for u in neighbors[v]:
                if u in U:
                    old_delta = delta[u]
                    delta[u] -= 1
                    neighbors[u].discard(v)
            U.remove(v)

        self.seedSet = S

    def genetic_search(self, select_goal_fun=GoalFuncType.F1, **genetic_params):
        """
        Run the genetic algorithm with customizable parameters.

        :param select_goal_fun: Which objective to use (F1, F2, F3).
        :param genetic_params: dictionary with keys
            - crossover_probability
            - mutation_probability
            - gene_swap_probability
            - bit_flip_probability
            - population_size
            - number_of_generations
            - verbose
            - new_individual_fraction
        """
        from Graph.utils.genetic import GeneticAlgo

        # Selezione della funzione obiettivo
        if select_goal_fun == GoalFuncType.F1:
            obj_fun = self.f1
        elif select_goal_fun == GoalFuncType.F2:
            obj_fun = self.f2
        elif select_goal_fun == GoalFuncType.F3:
            obj_fun = self.f3
        else:
            obj_fun = self.calc_majority_cascade_on_seed_set

        # Estrai i parametri dal dict, con valori di default identici a prima
        cxpb = genetic_params.get('crossover_probability', 0.8)
        mutpb = genetic_params.get('mutation_probability', 0.1)
        indpb_crossover = genetic_params.get('gene_swap_probability', 0.2)
        indpb_mutation = genetic_params.get('bit_flip_probability', 0.01)
        population_size = genetic_params.get('population_size', 200)
        num_generations = genetic_params.get('number_of_generations', 300)
        verbose = genetic_params.get('verbose', True)
        new_ind_fraction = genetic_params.get('new_individual_fraction', 0.1)

        alg = GeneticAlgo(
            self,
            cxpb=cxpb,
            mutpb=mutpb,
            indpb_crossover=indpb_crossover,
            indpb_mutation=indpb_mutation,
            population_size=population_size,
            num_generations=num_generations,
            cost_function=self.cost_fun,
            fitness_function=obj_fun,
            verbose=verbose,
            new_ind_fraction=new_ind_fraction
        )

        self.seedSet = alg.run()


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
        elif method == 'genetic':
            self.genetic_search(**kwargs)
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

        V = set(self.get_nodes_list())
        cascade = []

        prev_influenced = set(self.seedSet)
        cascade.append(prev_influenced.copy())

        while True:
            new_influenced = prev_influenced.copy()

            for v in V - prev_influenced:
                active_neighbors = sum(1 for nbr in self.get_neighbors(v) if nbr in prev_influenced)
                threshold = math.ceil(self.get_degree(v) / 2)

                if active_neighbors >= threshold:
                    new_influenced.add(v)

            cascade.append(new_influenced.copy())

            if new_influenced == prev_influenced:
                break
            prev_influenced = new_influenced

        self.cascade = cascade

    def calc_majority_cascade_on_seed_set(self, s: set):
        """
        Calculate the majority cascade on a given seed set
        Note: This method saves the previous seed set and majority cascade,
            then restores it after completion
        :param s: The seed set
        :return: List of influenced nodes from the seed set
        """
        prev_ss = self.seedSet
        prev_mjc = self.cascade
        self.seedSet = s # Use new seed set
        self.cascade = None

        self.calc_majority_cascade() # Majority cascade
        result = self.cascade

        self.seedSet = prev_ss  # Restore ss
        self.cascade = prev_mjc # Restore cascade

        return result

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

    def save_cascade_as_json(self, filepath=None):
        """
        Saves the current cascade (self.cascade) to a JSON file.
        Each cascade is assumed to be a list of sets of integers (e.g., a list of diffusion steps).
        If the file already exists, the cascade will be appended only if it’s not already present.

        Parameters:
            filepath (str): Optional path to the JSON file. If None, defaults to self.save_path + self.info_name.
        """

        def make_json_serializable(obj):
            """
            Recursively converts any sets in the object to sorted lists
            so they can be safely serialized to JSON.
            """
            if isinstance(obj, set):
                return sorted(make_json_serializable(e) for e in obj)
            elif isinstance(obj, list):
                return [make_json_serializable(e) for e in obj]
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            else:
                return obj

        # Determine file path
        if filepath is None:
            filepath = self.save_path + "txt/" + self.info_name + ".json"

        # Load existing cascades if the file exists
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                try:
                    cascades = json.load(f)
                except json.JSONDecodeError:
                    cascades = []
        else:
            cascades = []

        # Convert self.cascade into a JSON-serializable format
        cascade_serializable = make_json_serializable(self.cascade)

        # Add only if not already present
        if cascade_serializable not in cascades:
            cascades.append(cascade_serializable)

        # Save updated list to file
        with open(filepath, 'w') as f:
            json.dump(cascades, f, indent=2)

