"""
Genetic Algorithm Multi-Objective (Seed Size and Cost) with Budget Constraint
– NSGA-II selection filtered by budget
– Custom valid individual initializer
– Random injection per generation
– Detailed debug printing
– Proper error handling
– Parallelizzazione con Dask Distributed
"""

from deap import base, creator, tools
from Graph.graph import Graph
import random
from functools import partial
from dask.distributed import Client

#===============================================================================
# Evaluation function
#===============================================================================
def evaluate_individual(individual, node_list, fitness_function, cost_function, graph):
    active = individual.count(1)
    seed_set = {node_list[i] for i, g in enumerate(individual) if g == 1}

    # Compute spread
    try:
        spread_res = fitness_function(seed_set)
    except Exception:
        spread_res = 0

    if isinstance(spread_res, (set, list)):
        spread = len(spread_res)
    elif isinstance(spread_res, (int, float)):
        spread = spread_res
    else:
        spread = 0

    # Compute cost
    cost = graph.cost_seed_set(seed_set, cost_function)

    # Penalty for invalid
    if cost <= 0 or cost > graph.budget:
        return (0, graph.budget * 2)

    # Valid individual
    size = len(seed_set)
    return (size, cost)

#===============================================================================
# Custom initializer within budget
#===============================================================================
def init_valid_individual(node_list, graph, cost_function):
    individual = [0] * len(node_list)
    budget = graph.budget
    total_cost = 0

    indices = list(range(len(node_list)))
    random.shuffle(indices)
    for idx in indices:
        node = node_list[idx]
        node_cost = graph.cost_seed_set({node}, cost_function)
        if node_cost <= 0:
            continue
        if total_cost + node_cost <= budget:
            individual[idx] = 1
            total_cost += node_cost

    return creator.Individual(individual)

#===============================================================================
# Selection: NSGA-II filtered by budget
#===============================================================================
def sel_nsga2_filtered(population, k, budget):
    valid = [ind for ind in population if ind.fitness.values[1] <= budget]
    if len(valid) < k:
        return tools.selNSGA2(population, k)
    else:
        return tools.selNSGA2(valid, k)

#===============================================================================
# Genetic Algorithm Class
#===============================================================================
class GeneticAlgo:
    def __init__(self,
                 graph: Graph,
                 cxpb, mutpb, indpb_crossover, indpb_mutation,
                 population_size, num_generations,
                 cost_function,
                 fitness_function=Graph.calc_majority_cascade_on_seed_set,
                 verbose=True,
                 new_ind_fraction=0.1,
                 dask_scheduler='tcp://192.168.1.78:8786'):
        self.graph = graph
        self.node_list = graph.get_nodes_list()
        self.node_num = len(self.node_list)
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.indpb_crossover = indpb_crossover
        self.indpb_mutation = indpb_mutation
        self.pop_size = population_size
        self.num_generations = num_generations
        self.cost_function = cost_function
        self.fitness_function = fitness_function
        self.verbose = verbose
        self.new_ind_fraction = new_ind_fraction

        self.client = Client(dask_scheduler)

        self._setup_deap()

    def _setup_deap(self):
        # Multi-objective: maximize size, minimize cost
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        # Operatori di base (non usati quando si usa init_valid_ind)
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, self.node_num)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Inizializzatore custom
        toolbox.register("init_valid_ind", init_valid_individual,
                         self.node_list, self.graph, self.cost_function)
        toolbox.register("mate", tools.cxUniform, indpb=self.indpb_crossover)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.indpb_mutation)
        toolbox.register("select", partial(sel_nsga2_filtered, budget=self.graph.budget))

        self.toolbox = toolbox

    def run(self):
        # Registra la funzione di valutazione
        eval_fn = partial(evaluate_individual,
                          node_list=self.node_list,
                          fitness_function=self.fitness_function,
                          cost_function=self.cost_function,
                          graph=self.graph)
        self.toolbox.register("evaluate", eval_fn)

        # Inizializza popolazione
        pop = self.toolbox.population(n=self.pop_size)

        # Prima valutazione
        futures = self.client.map(self.toolbox.evaluate, pop)
        fits = self.client.gather(futures)
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

        # Ciclo evolutivo
        for gen in range(1, self.num_generations + 1):
            print(f"\n[RUN DEBUG] === Generation {gen} ===")

            # Selezione
            offspring = self.toolbox.select(pop, len(pop))
            # Clonazione
            offspring = [creator.Individual(ind[:]) for ind in offspring]

            # Crossover
            for i in range(1, len(offspring), 2):
                if random.random() < self.cxpb:
                    self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            # Mutazione
            for ind in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values

            # Iniezione nuovi individui validi
            num_new = max(1, int(self.pop_size * self.new_ind_fraction))
            new_inds = [self.toolbox.init_valid_ind() for _ in range(num_new)]
            offspring[-num_new:] = new_inds

            # Valuta solo gli invalid
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            if invalid:
                futures = self.client.map(self.toolbox.evaluate, invalid)
                fits = self.client.gather(futures)
                for ind, fit in zip(invalid, fits):
                    ind.fitness.values = fit

            pop[:] = offspring

            # Statistiche a video
            sizes = [ind.fitness.values[0] for ind in pop]
            costs = [ind.fitness.values[1] for ind in pop]
            if self.verbose:
                print(f"[STATS] Gen {gen}: max size={max(sizes)}, min cost={min(costs)}")

        # Estrai Pareto front
        front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        valid_front = [ind for ind in front if ind.fitness.values[1] <= self.graph.budget]
        print(f"[FINAL] Pareto solutions within budget: {len(valid_front)}")

        # Costruisci risultati
        results = []
        for ind in valid_front:
            seed_set = [self.node_list[i] for i, g in enumerate(ind) if g == 1]
            results.append({
                'seed_set': seed_set,
                'fitness': ind.fitness.values
            })

        # Scegli il migliore
        best_result = max(results, key=lambda r: (r['fitness'][0], -r['fitness'][1]))
        print(f"[RESULT] Best seed set: {best_result['seed_set']} "
              f"with fitness {best_result['fitness']} (budget={self.graph.budget})")

        return best_result['seed_set']
