import random

from cost_functions.base import CostFunction
from math import ceil


class CustomCostFunction(CostFunction):
    def calculate_cost(self, **kwargs):
        """Placeholder for a custom cost function (currently random-based)."""

        graph = kwargs.get("graph")
        igraph = kwargs.get("igraph")
        node_label = kwargs.get("node_label")
        total = 0
        neighbors_of_u = set(graph.get_neighbors(node_label))
        
        for v in neighbors_of_u:
            neighbors_of_v = set(graph.get_neighbors(v))
            degree_of_v = igraph.vs.find(name=v).degree()
            total += (degree_of_v / (len(neighbors_of_v.intersection(neighbors_of_u)) + 1))
        
        return ceil(total / 2)