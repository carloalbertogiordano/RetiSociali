"""
Genetic Algorithm Multi-Objective (Seed Size and Cost) with Budget Constraint
- NSGA-II selection filtered by budget
- Custom valid individual initializer
- Random injection per generation
- Detailed debug printing
- Proper error handling
"""
from deap import base, creator, tools, algorithms
from Graph.graph import Graph
import random
import multiprocessing
from functools import partial

#===============================================================================
# Evaluation function
#===============================================================================
def evaluate_individual(individual, node_list, cost_function, graph, individual_cache):
    """
    Multi-objective evaluation:
      1) maximize seed set size
      2) minimize cost
    Constraint: cost must be <= budget to be admissible.

    Returns:
      (size, cost) for valid individuals
      (0, budget*2) penalty for invalid individuals
    """
    key = tuple(individual)
    if key in individual_cache:
        return individual_cache[key]

    # Count active genes
    active = individual.count(1)
    #print(f"[EVAL DEBUG] Active genes: {active}/{len(individual)}")

    # Build seed set
    seed_set = {node_list[i] for i, g in enumerate(individual) if g == 1}
    #print(f"[EVAL DEBUG] Seed set: size={len(seed_set)}, ids(sample)={list(seed_set)[:5]}...")

    # Compute cost
    cost = graph.cost_seed_set(seed_set, cost_function)
    #print(f"[EVAL DEBUG] Cost: {cost} (Budget: {graph.budget})")

    # Penalty for invalid
    if cost <= 0 or cost > graph.budget:
        result = (0, graph.budget * 2)
        individual_cache[key] = result
        return result

    # Valid individual
    size = len(seed_set)
    result = (size, cost)
    individual_cache[key] = result
    return result

#===============================================================================
# Custom initializer within budget
#===============================================================================
def init_valid_individual(node_list, graph, cost_function):
    """
    Create an individual with random greedy selection under budget.
    """
    individual = [0] * len(node_list)
    budget = graph.budget
    total_cost = 0

    # Shuffle nodes and try include greedily
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
    #print(f"[INIT DEBUG] Generated valid individual with cost={total_cost}")
    return creator.Individual(individual)

#===============================================================================
# Selection: NSGA-II filtered by budget
#===============================================================================
def sel_nsga2_filtered(population, k, budget):
    """
    Apply NSGA-II selection on individuals within budget.
    If not enough, fallback to full population.
    """
    # Filter by budget
    valid = [ind for ind in population if ind.fitness.values[1] <= budget]
    #print(f"[SEL DEBUG] Total pop={len(population)}, valid={len(valid)}, required={k}")
    if len(valid) < k:
        #print("[SEL DEBUG] Not enough valid, applying NSGA-II on full pop")
        return tools.selNSGA2(population, k)
    else:
        #print("[SEL DEBUG] Applying NSGA-II on filtered valid subset")
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
                 new_ind_fraction=0.1):
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
        self._setup_deap()
        self.individual_cache = {}

    def _setup_deap(self):
        # Multi-objective: maximize size, minimize cost
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        # Operators
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, self.node_num)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("init_valid_ind", init_valid_individual,
                         self.node_list, self.graph, self.cost_function)
        toolbox.register("mate", tools.cxUniform, indpb=self.indpb_crossover)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.indpb_mutation)
        # Selection with budget
        toolbox.register("select", partial(sel_nsga2_filtered, budget=self.graph.budget))

        self.toolbox = toolbox

    def run(self):
        # Parallel evaluation
        with multiprocessing.Pool() as pool:
            self.toolbox.register("map", pool.map)
            eval_fn = partial(evaluate_individual,
                             node_list=self.node_list,
                             cost_function=self.cost_function,
                             graph=self.graph,
                              individual_cache = self.individual_cache)
            self.toolbox.register("evaluate", eval_fn)

            # Initialize population
            pop = self.toolbox.population(n=self.pop_size)
            # Evaluate
            fits = list(self.toolbox.map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fits):
                ind.fitness.values = fit

            # Evolution
            for gen in range(1, self.num_generations + 1):
                print(f"\n[RUN DEBUG] === Generation {gen} ===")
                # Select
                offspring = self.toolbox.select(pop, len(pop))
                # Clone
                offspring = [creator.Individual(ind[:]) for ind in offspring]

                # Crossover
                for i in range(1, len(offspring), 2):
                    if random.random() < self.cxpb:
                        self.toolbox.mate(offspring[i-1], offspring[i])
                        del offspring[i-1].fitness.values
                        del offspring[i].fitness.values

                # Mutation
                for ind in offspring:
                    if random.random() < self.mutpb:
                        self.toolbox.mutate(ind)
                        del ind.fitness.values

                # Inject new valid individuals
                num_new = max(1, int(self.pop_size * self.new_ind_fraction))
                new_inds = [self.toolbox.init_valid_ind() for _ in range(num_new)]
                #if self.verbose: print(f"[RUN DEBUG] Injecting {num_new} new individuals")
                # Replace worst by cost ascending
                offspring[-num_new:] = new_inds

                # Evaluate invalid
                invalid = [ind for ind in offspring if not ind.fitness.valid]
                fits = list(self.toolbox.map(self.toolbox.evaluate, invalid))
                for ind, fit in zip(invalid, fits):
                    ind.fitness.values = fit

                pop[:] = offspring

                # Stats
                sizes = [ind.fitness.values[0] for ind in pop]
                costs = [ind.fitness.values[1] for ind in pop]
                #if self.verbose: print(f"[STATS] Max size={max(sizes)}, Min cost={min(costs)}")

                        # Extract Pareto front
            front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
            # Filter final by budget
            valid_front = [ind for ind in front if ind.fitness.values[1] <= self.graph.budget]
            if self.verbose: print(f"[FINAL] Pareto solutions within budget: {len(valid_front)}")

            # Convert binary individuals to actual node IDs
            results = []
            for ind in valid_front:
                seed_set = [self.node_list[i] for i, g in enumerate(ind) if g == 1]
                results.append({
                    'seed_set': seed_set,
                    'fitness': ind.fitness.values
                })
            #if self.verbose: print(f"RESULT: {results}")
            # Trova il seed set con fitness massimo (priorità: dimensione, poi costo)
            best_result = max(results, key=lambda r: (r['fitness'][0], -r['fitness'][1]))
            print(f"[RESULT] Best seed set: {best_result['seed_set']} with fitness {best_result['fitness']} with total budget: {self.graph.budget}")
            return best_result['seed_set']
