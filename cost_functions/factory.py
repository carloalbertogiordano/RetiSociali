from Graph.graph import Graph
from cost_functions.custom_cost import CustomCostFunction
from cost_functions.degree_cost import DegreeCostFunction
from cost_functions.random_cost import RandomCostFunction
from enum import Enum


class CostFuncType(Enum):
    RANDOM = 1
    DEGREE = 2
    CUSTOM = 3

class CostFunctionFactory:

    @staticmethod
    def create_cost_function(cost_type):
        """Create a cost function instance based on the specified type."""
        if cost_type == CostFuncType.RANDOM:
            return RandomCostFunction()
        elif cost_type == CostFuncType.DEGREE:
            return DegreeCostFunction()
        elif cost_type == CostFuncType.CUSTOM:
            return CustomCostFunction()
        else:
            raise ValueError(f"Unknown cost function type: {cost_type}")
