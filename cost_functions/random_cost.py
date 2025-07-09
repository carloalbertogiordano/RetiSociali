import random

from cost_functions.base import CostFunction


class RandomCostFunction(CostFunction):
    def calculate_cost(self, **kwargs):
        """
        Parameters in kwargs:
            - rangeLow (int): default 1
            - rangeMax (int): default 10
        """
        rangeLow = kwargs.get("rangeLow", 1)
        rangeMax = kwargs.get("rangeMax", 10)
        return random.randint(rangeLow, rangeMax)
