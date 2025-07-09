import matplotlib.pyplot as plt
from matplotlib import cm
import igraph as ig
import random
from enum import Enum
import math
import os
from PIL import Image
import glob
from pyvis.network import Network
from cost_functions import base as cost_func_base


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


def calculate_budget(graph: ig.Graph, cost_fun):
    budget = 0

    degrees = graph.degree()
    degrees_dict = {v.index: degree for v, degree in zip(graph.vs, degrees)}
    top5 = sorted(degrees_dict.items(), key=lambda item: item[1], reverse=True)[:5]
    top5_nomi = [(graph.vs[i]["name"], degree) for i, degree in top5]
    for name, val in top5_nomi:
        budget += cost_fun(node_label=name, graph=graph)
    return budget

class Graph:

    class GoalFuncType(Enum):
        F1 = 1
        F2 = 2
        F3 = 3

    def __init__(self, file_path: str, save_path: str, cost_func :cost_func_base.CostFunction.calculate_cost, is_sub_graph=True, sub_graph_dim=100):
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

        if is_sub_graph:
            self.graph = get_subgraph(self.full_graph, sub_graph_dim)

        if not isinstance(cost_func, cost_func_base.CostFunction):
            raise ValueError(f"Cost func is {type(cost_func)}, expected subclass of CostFunction")
        self.cost_fun = cost_func.calculate_cost

        self.budget = calculate_budget(self.graph, self.cost_fun)
        if self.budget > self.graph.vcount():
            print(f"WARN: BUDGET HIGH GOT {self.budget}, NODE NUMBER IS {self.graph.vcount()}")

        #self.budget = budget
        self.save_path = save_path
        self.seedSet = None
        self.cascade = None

    def cost_seed_set(self, S, cost):
        """
        Calculate the total cost of the seed set by summing the cost of each node.

        :param S: The seed set, a collection of nodes.
        :param cost: A function that returns the cost of a node.
        :return: The total cost of the seed set.
        """
        return sum(cost(node_label=u, graph=self.graph) for u in S)

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
        return max(set(V) - S, key=lambda v: marginal_gain(v, S, f) / self.cost_fun(node_label=v, graph=self.graph))

    def f1(self, S):
        """
                Objective function F1: Sum of minimum between neighbors in S and threshold.

                :param S: The seed set.
                :return: The total value of the objective function across all nodes.
                """
        total = 0
        for v in range(self.graph.vcount()):
            neighbors_in_S = set(self.graph.neighbors(v)).intersection(S)
            threshold = math.ceil(self.graph.degree(v) / 2)
            total += min(len(neighbors_in_S), threshold)
        return total

    def f2(self, S):
        """
        Objective function F2: Sum of decremented thresholds based on neighbors in S.

        :param S: The seed set.
        :return: The total value of the objective function across all nodes.
        """
        total = 0
        for v in range(self.graph.vcount()):
            neighbors_in_S = list(set(self.graph.neighbors(v)).intersection(S))
            d_v = self.graph.degree(v)
            threshold = math.ceil(d_v / 2)
            for i in range(1, len(neighbors_in_S) + 1):
                total += max(threshold - i + 1, 0)
        return total

    def f3(self, S):
        """
        Objective function F3: Weighted sum based on neighbors in S and degree.

        :param S: The seed set.
        :return: The total value of the objective function across all nodes.
        """
        total = 0
        for v in range(self.graph.vcount()):
            neighbors_in_S = list(set(self.graph.neighbors(v)).intersection(S))
            d_v = self.graph.degree(v)
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
            print(f" Cost seed set: {self.cost_seed_set(S_d, self.cost_fun)}")
            # print(f"Costo di S {self.cost_seed_set(S_d, cost_fn)}, k={k}")
            u = self.argmax(V, S_d, obj_fun)
            print(f"Selected node: {u}")

            S_p = S_d.copy()
            S_d.add(u)
            print(f"Seed set (d): {S_d}")
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
                        return self.cost_fun(label=u, graph=self.graph) * k[u] / (delta[u] * (delta[u] + 1))

                    v = min(U, key=score)

            # Rimuovi v da U e aggiorna grafo dinamico
            for u in neighbors[v]:
                if u in U:
                    delta[u] -= 1
                    neighbors[u].discard(v)
            U.remove(v)
        self.seedSet = S

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
            return  # Error

        # Calc majority cascade
        #V = set(range(self.graph.vcount()))
        V = set(self.graph.vs["name"])
        print(f"Vertex set: {V}")        
        cascade = []  # lista Inf[S,0], Inf[S,1], ...
        
        prev_influenced = set([x for x in self.seedSet])
        print(f"Prev influenced (Seed set): {prev_influenced}")
        cascade.append(prev_influenced.copy())

        print(f"V-Seedset = {V-prev_influenced}")
        while True:
            new_influenced = prev_influenced.copy()

            for v in V - prev_influenced:
                neighbors = self.graph.vs.find(name=v).neighbors()
                active_neighbors = sum(
                    1 for u in neighbors if u in prev_influenced)
                if active_neighbors >= math.ceil(self.graph.vs.find(name=v).degree() / 2):
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
        """
        Plot the evolution of the majority cascade over time.
        """
        x = list(range(len(self.cascade)))  # Indici: 0, 1, 2, ...
        y = [len(s) for s in self.cascade]  # Cardinalità di ogni set

        plt.plot(x, y, marker='o')
        plt.xlabel("Indice del passo nella cascata")
        plt.ylabel("Numero di nodi (len del set)")
        plt.title("Evoluzione della cascata")
        plt.grid(True)
        plt.show()

    def dyn_plot_cascade(self, test_name: str):
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
        return 
        layout = self.graph.layout("fr")  # Use force-directed layout for graph positioning
        max_step = len(self.cascade)
        colormap = cm.get_cmap("plasma", max_step + 1)

        def rgba_to_hex(rgba):
            """Convert RGBA color to HEX string (ignore alpha)."""
            r, g, b, _ = rgba
            return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))

        # === Centralized directory structure ===
        base_output_dir = os.path.join(self.save_path, "plots")
        cascade_dir = os.path.join(base_output_dir, "plot_cascade")
        images_dir = os.path.join(cascade_dir, "images")
        gif_path = os.path.join(cascade_dir, f"diffusione_{test_name}.gif")

        # Create output directories if they do not exist
        os.makedirs(images_dir, exist_ok=True)

        # Compute cumulative activated nodes at each cascade step
        cumulative_cascade = []
        active_nodes = set()
        for step in self.cascade:
            active_nodes |= step
            cumulative_cascade.append(active_nodes.copy())

        # Generate one image per step
        for t, active in enumerate(cumulative_cascade):
            colors = []
            for v in range(self.graph.vcount()):
                if v in active:
                    c = colormap(t)               # Color based on activation time
                    c_hex = rgba_to_hex(c)
                else:
                    c_hex = "#dddddd"             # Gray for inactive nodes
                colors.append(c_hex)

            ig.plot(
                self.graph,
                target=os.path.join(images_dir, f"step_{t:02d}_{test_name}.png"),
                layout=layout,
                vertex_color=colors,
                vertex_size=8,
                bbox=(3000, 3000),
                margin=40,
            )

        # Load all generated images and compile into an animated GIF
        images = [Image.open(f) for f in sorted(glob.glob(os.path.join(images_dir, "step_*.png")))]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=1000,  # Frame duration in milliseconds
            loop=0
        )
