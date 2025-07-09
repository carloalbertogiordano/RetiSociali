import unittest
import igraph as ig

from cost_functions.random_cost import RandomCostFunction
from cost_functions.degree_cost import DegreeCostFunction
from cost_functions.custom_cost import CustomCostFunction


class TestCostFunctionsWithIgraph(unittest.TestCase):

    def setUp(self):
        # Crea un grafo con 3 nodi e 2 archi
        self.graph = ig.Graph()
        self.graph.add_vertices(3)
        self.graph.add_edges([(0, 1), (1, 2)])
        # Aggiungi nomi ai nodi (richiesto da .vs.find(name=...))
        self.graph.vs["name"] = ["1", "2", "3"]

    def test_random_cost_function(self):
        func = RandomCostFunction()
        cost = func.calculate_cost(rangeLow=5, rangeMax=10)
        print(f"Random cost function got cost = {cost}")
        self.assertTrue(5 <= cost <= 10)

    def test_custom_cost_function(self):
        func = CustomCostFunction()
        cost = func.calculate_cost(rangeLow=100, rangeMax=200)
        self.assertTrue(100 <= cost <= 200)

    def test_degree_cost_function(self):
        func = DegreeCostFunction(self.graph)
        cost = func.calculate_cost("2")  # Nodo 2 ha grado 2
        self.assertEqual(cost, 1.0)      # degree / 2 = 2 / 2 = 1.0

    def test_degree_invalid_node_label(self):
        func = DegreeCostFunction(self.graph)
        with self.assertRaises(ValueError):
            func.calculate_cost("6")  # Nodo inesistente

    def test_degree_invalid_type_node_label(self):
        func = DegreeCostFunction(self.graph)
        with self.assertRaises(TypeError):
            func.calculate_cost(42)  # Non è una stringa

if __name__ == "__main__":
    unittest.main()
