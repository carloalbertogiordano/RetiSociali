"""
Genetic Algorithm Multi-Objective (Seed Size and Cost) with Budget Constraint
– NSGA-II selection filtered by budget
– Custom valid individual initializer
– Random injection per generation
– Detailed debug printing
– Proper error handling
– Parallelizzazione con Dask Distributed
"""
import os
import logging
import random
from functools import partial
from dask.distributed import Client
import cloudpickle
from deap import base, creator, tools
from Graph.graph import Graph

# Configura il logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#===============================================================================
# Evaluation function
#===============================================================================
def evaluate_individual(individual, node_list, fitness_function, cost_function, graph):
    """Valuta un individuo calcolando dimensione del seed set e costo."""
    return (sum(individual), 0)
    logger.debug("Inizio valutazione individuo")
    active = individual.count(1)
    seed_set = {node_list[i] for i, g in enumerate(individual) if g == 1}

    # Compute spread
    try:
        spread_res = fitness_function(seed_set)
        logger.debug(f"Risultato spread: {spread_res}")
    except Exception as e:
        logger.error(f"Errore in fitness_function per seed_set {seed_set}: {e}")
        spread_res = 0

    if isinstance(spread_res, (set, list)):
        spread = len(spread_res)
    elif isinstance(spread_res, (int, float)):
        spread = spread_res
    else:
        spread = 0

    # Compute cost
    cost = graph.cost_seed_set(seed_set, cost_function)
    logger.debug(f"Costo calcolato: {cost}")

    # Penalty for invalid
    if cost <= 0 or cost > graph.budget:
        logger.debug("Individuo non valido: costo fuori dai limiti")
        return (0, graph.budget * 2)

    # Valid individual
    size = len(seed_set)
    logger.debug(f"Fine valutazione individuo: size={size}, cost={cost}")
    return (size, cost)

#===============================================================================
# Custom initializer within budget
#===============================================================================
def init_valid_individual(node_list, graph, cost_function):
    """Inizializza un individuo valido rispettando il budget."""
    logger.debug("Inizio inizializzazione individuo valido")
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

    logger.debug(f"Fine inizializzazione individuo: costo totale={total_cost}")
    return creator.Individual(individual)

#===============================================================================
# Selection: NSGA-II filtered by budget
#===============================================================================
def sel_nsga2_filtered(population, k, budget):
    """Selezione NSGA-II con filtro sul budget."""
    logger.debug("Inizio selezione NSGA-II filtrata")
    valid = [ind for ind in population if ind.fitness.values[1] <= budget]
    logger.debug(f"Individui validi: {len(valid)}")
    if len(valid) < k:
        result = tools.selNSGA2(population, k)
    else:
        result = tools.selNSGA2(valid, k)
    logger.debug("Fine selezione NSGA-II filtrata")
    return result

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
                 dask_scheduler='tcp://192.168.188.60:8786'):
        """Inizializza l'algoritmo genetico."""
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

        # Connessione al cluster Dask con controllo
        logger.info("Tentativo di connessione al cluster Dask")
        try:
            self.client = Client(dask_scheduler)
            logger.info(f"Connesso al cluster Dask: {self.client}")
        except Exception as e:
            logger.error(f"Errore connessione al cluster Dask: {e}")
            raise

        # Controlli di serializzazione iniziali
        self._check_serialization()

        # Setup DEAP
        self._setup_deap()

        # 🔧 REGISTRAZIONE FUNZIONE DI VALUTAZIONE
        eval_fn = partial(evaluate_individual,
                          node_list=self.node_list,
                          fitness_function=self.fitness_function,
                          cost_function=self.cost_function,
                          graph=self.graph)
        self.toolbox.register("evaluate", eval_fn)
        self.eval_fn = eval_fn

    def _check_serialization(self):
        """Verifica la serializzabilità degli oggetti critici con cloudpickle."""
        logger.info("Verifica della serializzabilità degli oggetti critici")

        # Controllo per graph
        try:
            cloudpickle.dumps(self.graph)
            logger.info("Oggetto 'graph' serializzabile")
        except Exception as e:
            logger.error(f"Errore serializzazione 'graph': {e}")
            raise

        # Controllo per fitness_function
        try:
            cloudpickle.dumps(self.fitness_function)
            logger.info("Funzione 'fitness_function' serializzabile")
        except Exception as e:
            logger.error(f"Errore serializzazione 'fitness_function': {e}")
            raise

        # Controllo per cost_function
        try:
            cloudpickle.dumps(self.cost_function)
            logger.info("Funzione 'cost_function' serializzabile")
        except Exception as e:
            logger.error(f"Errore serializzazione 'cost_function': {e}")
            raise

    def _setup_deap(self):
        """Configura gli strumenti DEAP."""
        logger.debug("Configurazione DEAP")
        # Multi-objective: maximize size, minimize cost
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
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
        logger.debug("Fine configurazione DEAP")

    def run(self):
        ind = self.toolbox.init_valid_ind()
        future = self.client.submit(self.toolbox.evaluate, ind)

        try:
            result = self.client.gather(future)
            print("Risultato valutazione:", result)
        except Exception as e:
            logger.error("Errore durante la valutazione distribuita:")
            logger.exception(e)  # mostra stack trace completo
