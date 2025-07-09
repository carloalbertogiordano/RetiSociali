import unittest
import igraph as ig

from cost_functions.factory import CostFunctionFactory
from cost_functions.random_cost import RandomCostFunction
from cost_functions.degree_cost import DegreeCostFunction
from cost_functions.custom_cost import CustomCostFunction
from cost_functions.factory import CostFuncType as Cft

class TestFactoryWithIgraph(unittest.TestCase):

    def setUp(self):
        # Crea un grafo con 3 nodi e 2 archi
        self.graph = ig.Graph()
        self.graph.add_vertices(3)
        self.graph.add_edges([(0, 1), (1, 2)])
        # Aggiungi nomi ai nodi (richiesto da .vs.find(name=...))
        self.graph.vs["name"] = ["1", "2", "3"]

    def test_factory_random(self):
        func = CostFunctionFactory.create_cost_function(Cft.RANDOM)
        self.assertIsInstance(func, RandomCostFunction)

    def test_factory_custom(self):
        func = CostFunctionFactory.create_cost_function(Cft.CUSTOM)
        self.assertIsInstance(func, CustomCostFunction)

    def test_factory_degree_with_graph(self):
        func = CostFunctionFactory.create_cost_function(Cft.DEGREE)
        self.assertIsInstance(func, DegreeCostFunction)

    def test_factory_unknown_type(self):
        class FakeType:
            pass
        with self.assertRaises(ValueError):
            CostFunctionFactory.create_cost_function(FakeType())
