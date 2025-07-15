import random

from cost_functions.base import CostFunction
from math import ceil


class RandomCostFunction(CostFunction):
    cache = {}

    def calculate_cost(self, **kwargs):
        """
        Parameters in kwargs:
            - rangeLow (int): default 1
            - rangeMax (int): default 10
        """
        node_label = kwargs.get("node_label")

        if node_label in self.cache.keys():
            return self.cache[node_label]

        rangeLow = kwargs.get("rangeLow", 1)
        graph = kwargs.get("graph")
        degree = graph.get_degree(node_label) 
        rangeMax = kwargs.get("rangeMax", ceil(degree/2))
        result = random.randint(rangeLow, rangeMax)
        self.cache[node_label] = result
        return result
