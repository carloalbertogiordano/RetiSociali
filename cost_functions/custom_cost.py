import random

from cost_functions.base import CostFunction


class CustomCostFunction(CostFunction):
    def calculate_cost(self, **kwargs):
        """Placeholder for a custom cost function (currently random-based)."""
        rangeLow = kwargs.get("rangeLow", 10)
        rangeMax = kwargs.get("rangeMax", 1000)
        return random.randint(rangeLow, rangeMax)