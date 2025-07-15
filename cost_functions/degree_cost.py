from cost_functions.base import CostFunction
from math import ceil

class DegreeCostFunction(CostFunction):
    cache = {}
    def calculate_cost(self, **kwargs):
        """Compute the cost based on the node's degree."""
        node_label = kwargs.get("node_label")
        if node_label in self.cache.keys():
            return self.cache[node_label]
        
        graph = kwargs.get("graph")
        result = ceil(graph.get_degree(node_label) / 2)
        self.cache[node_label] = result
        return result