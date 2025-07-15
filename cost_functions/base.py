from abc import ABC, abstractmethod

class CostFunction(ABC):
    @abstractmethod
    def calculate_cost(self, **kwargs):
        """Calculate the cost of a node given its label and optional arguments."""
        pass

    def print_name(self):
        pass