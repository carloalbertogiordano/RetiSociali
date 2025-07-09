from cost_functions.base import CostFunction


class DegreeCostFunction(CostFunction):
    def calculate_cost(self, **kwargs):
        """Compute the cost based on the node's degree."""
        node_label = kwargs.get("node_label")
        graph = kwargs.get("graph")
        #node_label = f"{node_label}"
        print(f"label:{node_label}")
        print(graph)
        node = graph.vs.find(name=node_label)
        return node.degree() / 2