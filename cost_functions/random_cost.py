import random

from cost_functions.base import CostFunction
from math import ceil


class RandomCostFunction(CostFunction):
    def calculate_cost(self, **kwargs):
        """
        Parameters in kwargs:
            - rangeLow (int): default 1
            - rangeMax (int): default 10
        """
        rangeLow = kwargs.get("rangeLow", 1)
        node_label = kwargs.get("node_label")
        graph = kwargs.get("graph")
        degree = graph.get_degree(node_label) 
        rangeMax = kwargs.get("rangeMax", ceil(degree/2))
        return random.randint(rangeLow, rangeMax)
