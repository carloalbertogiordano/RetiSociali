import random

from cost_functions.base import CostFunction
from math import ceil


class CustomCostFunction(CostFunction):
    cache = {}
    def calculate_cost(self, **kwargs):

        graph = kwargs.get("graph")
        node_label = kwargs.get("node_label")

        if node_label in self.cache.keys() and graph.is_cache():
            return self.cache[node_label]
        
        total = 0
        neighbors_of_u = set(graph.get_neighbors(node_label))
        
        for v in neighbors_of_u:
            neighbors_of_v = set(graph.get_neighbors(v))
            degree_of_v = graph.get_degree(v)
            total += (degree_of_v / (len(neighbors_of_v.intersection(neighbors_of_u)) + 1))

        result = ceil(total / 2)
        self.cache[node_label] = result

        return result

    def print_name(self):
        return "CUSTOM"