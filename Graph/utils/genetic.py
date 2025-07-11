"""
Genetic Algorithm Multi-Objective (Seed Size and Cost) with Budget Constraint
NSGA-II selection filtered by budget
Custom valid individual initializer
Random injection per generation
GPU parallelization with CuPy
"""
from deap import base, creator, tools, algorithms
from Graph.graph import Graph
import random
import cupy as cp
import cupyx.scipy.sparse as cp_sparse
import numpy as np
import time

#===============================================================================
# Evaluation function for GPU
#===============================================================================
def evaluate_individuals_gpu(population, node_costs, budget, fitness_function_gpu, adj_matrix):
    """
    Valuta la popolazione su GPU.
    Input:
        population: cp.ndarray (pop_size, node_num) - matrice binaria degli individui
        node_costs: cp.ndarray (node_num,) - costi dei nodi
        budget: float - vincolo di budget
        fitness_function_gpu: funzione GPU-compatibile per calcolare lo spread
        adj_matrix: cp.sparse.csr_matrix - matrice di adiacenza del grafo
    Output:
        cp.ndarray (pop_size, 2) - fitness (size, cost) per ogni individuo
    """
    pop_size, node_num = population.shape
    fitness = cp.zeros((pop_size, 2), dtype=cp.float32)

    # Calcola dimensione del seed set (somma degli 1 per riga)
    fitness[:, 0] = cp.sum(population, axis=1)

    # Calcola costo (prodotto matriciale tra popolazione e node_costs)
    costs = cp.dot(population, node_costs)

    # Calcola spread
    spreads = fitness_function_gpu(population, adj_matrix)

    # Applica penalità per individui non validi
    invalid_mask = (costs <= 0) | (costs > budget)
    fitness[invalid_mask, 0] = 0
    fitness[invalid_mask, 1] = budget * 2

    # Per individui validi, usa dimensione e costo
    valid_mask = ~invalid_mask
    fitness[valid_mask, 1] = costs[valid_mask]

    return fitness

def fitness_function_gpu(population, adj_matrix, max_iterations=10):
    """
    Calcola lo spread della cascata maggioritaria per ogni individuo su GPU, emulando
    Graph.calc_majority_cascade_on_seed_set.
    Input:
        population: cp.ndarray (pop_size, node_num) - matrice binaria degli individui,
                    dove population[i, j] = 1 se il nodo j (con vs["name"]=j) è nel seed set
        adj_matrix: cp.sparse.csr_matrix (node_num, node_num) - matrice di adiacenza sparsa
        max_iterations: int - numero massimo di iterazioni per la convergenza
    Output:
        cp.ndarray (pop_size,) - spread (numero di nodi attivati) per ogni individuo
    """
    pop_size, node_num = population.shape
    # Stato iniziale: nodi attivi sono quelli nel seed set
    active_nodes = population.copy().astype(cp.float32)  # (pop_size, node_num)
    spreads = cp.zeros(pop_size, dtype=cp.float32)

    # Calcola il grado di ogni nodo (numero di vicini)
    degrees = cp.array(adj_matrix.sum(axis=1)).flatten()  # (node_num,)
    # Soglia: ceil(degree/2) per ogni nodo, come in calc_majority_cascade
    thresholds = cp.ceil(degrees / 2).astype(cp.float32)  # (node_num,)

    # Itera per simulare la cascata
    for _ in range(max_iterations):
        # Calcola vicini attivi per ogni nodo: prodotto matriciale
        neighbors_active = cp_sparse.csr_matrix.dot(adj_matrix, active_nodes.T).T  # (pop_size, node_num)

        # Aggiorna stato: un nodo diventa attivo se il numero di vicini attivi >= soglia
        # o se è già attivo (parte del seed set o attivato precedentemente)
        new_active = (neighbors_active >= thresholds) | (active_nodes > 0)

        # Controlla convergenza: se nessuno stato cambia, esci
        if cp.all(new_active == active_nodes):
            break

        active_nodes = new_active.astype(cp.float32)

    # Calcola spread: somma dei nodi attivi per ogni individuo
    spreads = cp.sum(active_nodes, axis=1)
    return spreads

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
    """
    Apply NSGA-II selection on individuals within budget.
    If not enough, fallback to full population.
    """
    valid = [ind for ind in population if ind.fitness.values[1] <= budget]
    if len(valid) < k:
        return tools.selNSGA2(population, k)
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
        self.fitness_function = fitness_function  # Original CPU function (for reference)
        self.fitness_function_gpu = fitness_function_gpu  # GPU-compatible function
        self.verbose = verbose
        self.new_ind_fraction = new_ind_fraction

        # Precompute node costs on GPU
        self.node_costs = cp.array([cost_function(node) for node in self.node_list], dtype=cp.float32)

        # Initialize adjacency matrix
        adj_matrix_np = self.graph.get_adjacency_matrix()  # Assumes Graph provides this
        self.adj_matrix = cp_sparse.csr_matrix(adj_matrix_np)

        self._setup_deap()

    def _setup_deap(self):
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, self.node_num)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("init_valid_ind", init_valid_individual,
                         self.node_list, self.graph, self.cost_function)
        toolbox.register("mate", tools.cxUniform, indpb=self.indpb_crossover)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.indpb_mutation)
        toolbox.register("select", sel_nsga2_filtered, budget=self.graph.budget)

        self.toolbox = toolbox

    def run(self):
        start_time = time.time()

        # Inizializza popolazione
        pop = [self.toolbox.init_valid_ind() for _ in range(self.pop_size)]
        pop_array = cp.array(pop, dtype=cp.int8)

        # Valutazione iniziale su GPU
        fits = evaluate_individuals_gpu(pop_array, self.node_costs, self.graph.budget,
                                       self.fitness_function_gpu, self.adj_matrix)
        pop_fitness = [creator.FitnessMulti() for _ in range(self.pop_size)]
        for i, fit in enumerate(fits.get()):
            pop_fitness[i].values = tuple(fit)

        for ind, fit in zip(pop, pop_fitness):
            ind.fitness.values = fit.fitness.values

        # Evoluzione
        for gen in range(1, self.num_generations + 1):
            if self.verbose:
                print(f"\n[RUN DEBUG] === Generation {gen} ===")

            offspring = self.toolbox.select(pop, len(pop))
            offspring = [creator.Individual(ind[:]) for ind in offspring]

            for i in range(1, len(offspring), 2):
                if random.random() < self.cxpb:
                    self.toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            for ind in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values

            num_new = max(1, int(self.pop_size * self.new_ind_fraction))
            new_inds = [self.toolbox.init_valid_ind() for _ in range(num_new)]
            offspring[-num_new:] = new_inds

            offspring_array = cp.array(offspring, dtype=cp.int8)
            fits = evaluate_individuals_gpu(offspring_array, self.node_costs, self.graph.budget,
                                           self.fitness_function_gpu, self.adj_matrix)
            for i, fit in enumerate(fits.get()):
                offspring[i].fitness.values = tuple(fit)

            pop[:] = offspring
            pop_array = cp.array(offspring, dtype=cp.int8)

            if self.verbose:
                sizes = [ind.fitness.values[0] for ind in pop]
                costs = [ind.fitness.values[1] for ind in pop]
                print(f"[STATS] Max size={max(sizes)}, Min cost={min(costs)}")

        front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        valid_front = [ind for ind in front if ind.fitness.values[1] <= self.graph.budget]
        if self.verbose:
            print(f"[FINAL] Pareto solutions within budget: {len(valid_front)}")

        results = []
        for ind in valid_front:
            seed_set = [self.node_list[i] for i, g in enumerate(ind) if g == 1]
            results.append({
                'seed_set': seed_set,
                'fitness': ind.fitness.values
            })

        best_result = max(results, key=lambda r: (r['fitness'][0], -r['fitness'][1]))

        end_time = time.time()
        elapsed = end_time - start_time
        if self.verbose:
            print(f"[TIMER] Total run() time: {elapsed:.2f} seconds")
            print(f"[RESULT] Best seed set: {best_result['seed_set']} with fitness {best_result['fitness']} with total budget: {self.graph.budget}")

        return best_result['seed_set']