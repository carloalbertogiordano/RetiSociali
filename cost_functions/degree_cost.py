from cost_functions.base import CostFunction
from math import ceil

class DegreeCostFunction(CostFunction):
    def calculate_cost(self, **kwargs):
        """Compute the cost based on the node's degree."""
        node_label = kwargs.get("node_label")
        graph = kwargs.get("graph")
        return ceil(graph.get_degree(node_label) / 2)