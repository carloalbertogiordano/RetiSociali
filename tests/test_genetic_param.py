import unittest
import igraph as ig

from cost_functions.factory import CostFunctionFactory
from cost_functions.random_cost import RandomCostFunction
from cost_functions.degree_cost import DegreeCostFunction
from cost_functions.custom_cost import CustomCostFunction
from cost_functions.factory import CostFuncType as Cft

import itertools
import math
import random
from Graph.graph import Graph

from cost_functions.factory import CostFunctionFactory as Cff
from cost_functions.factory import CostFuncType as Cft
from Graph.utils.genetic import GeneticAlgo
from Graph.utils.genetic import evaluate_individual

def run_once(params, graph, cost_fn, fitness_fn):
    """
    Avvia una singola run del GA con i parametri specificati e
    restituisce il best_result ('seed_set', 'fitness') della classe GeneticAlgo.
    """
    ga = GeneticAlgo(
        graph=graph,
        cxpb=params['cxpb'],
        mutpb=params['mutpb'],
        indpb_crossover=params['indpb_crossover'],
        indpb_mutation=params['indpb_mutation'],
        population_size=params['pop_size'],
        num_generations=params['ngen'],
        cost_function=cost_fn,
        fitness_function=fitness_fn,
        verbose=False,
        new_ind_fraction=params['new_ind_frac'],
    )
    best_seed_set = ga.run()        # già restituisce il migliore
    size, cost = ga.toolbox.evaluate(
        [1 if node in best_seed_set else 0 for node in ga.node_list]
    )
    return {'seed_set': best_seed_set, 'size': size, 'cost': cost}


def grid_search(graph, cost_fn, fitness_fn, n_runs=3, random_seed=42):
    random.seed(random_seed)

    # ⇩  definisci qui il tuo spazio degli iper-parametri
    param_grid = {
        'cxpb':             [0.6, 0.8],
        'mutpb':            [0.2, 0.4],
        'indpb_crossover':  [0.4, 0.6],
        'indpb_mutation':   [0.05, 0.1],
        'pop_size':         [100, 200],
        'ngen':             [100, 150],
        'new_ind_frac':     [0.05, 0.1],
    }

    keys = list(param_grid.keys())
    best_conf, best_score = None, (-math.inf, math.inf)   # (size,-cost)

    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        # media dei punteggi su n_runs
        agg_size, agg_cost = 0, 0
        for i in range(n_runs):
            res = run_once(params, graph, cost_fn, fitness_fn)
            agg_size += res['size']
            agg_cost += res['cost']
        avg_size = agg_size / n_runs
        avg_cost = agg_cost / n_runs

        score = (avg_size, -avg_cost)
        print(f"Test {params}  →  size={avg_size:.2f}, cost={avg_cost:.2f}")

        if score > best_score:
            best_score = score
            best_conf = params

    print("\n=== Migliori parametri trovati ===")
    print(best_conf, "→", best_score)
    return best_conf

class TestGeneticParam(unittest.TestCase):

    def setUp(self):
        data_file = 'sourceData/facebook_data/facebook_combined.txt'
        output_dir = 'results/'
        sub_graph_dim = 200
        base_cost = Cff.create_cost_function(Cft.CUSTOM)
        name = "GENETIC_test"#f"GENETIC_{base_cost.name.lower()}_{""}"
        self.graph = Graph(data_file, output_dir, base_cost, is_sub_graph=True, sub_graph_dim=sub_graph_dim, info_name=name)


    def test_gen_matrix(self):
        grid_search(graph=self.graph, cost_fn=self.graph.cost_fun, fitness_fn=evaluate_individual)
        assert(True)

