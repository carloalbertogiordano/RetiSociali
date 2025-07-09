from cost_functions.base import CostFunction


class DegreeCostFunction(CostFunction):
    def __init__(self, graph):
        """Initialize with a graph to access node degrees."""
        self.graph = graph

    def calculate_cost(self, node_label, **kwargs):
        """Compute the cost based on the node's degree."""
        if not isinstance(node_label, str):
            raise TypeError(f"node_label must be a string, got {type(node_label)}")
        node = self.graph.vs.find(name=node_label)
        return node.degree() / 2