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

        self._setup_deap()

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
        """Esegue l'algoritmo genetico."""
        logger.info("Inizio esecuzione algoritmo genetico")

        # Registra la funzione di valutazione
        eval_fn = partial(evaluate_individual,
                          node_list=self.node_list,
                          fitness_function=self.fitness_function,
                          cost_function=self.cost_function,
                          graph=self.graph)
        self.toolbox.register("evaluate", eval_fn)

        # Controllo serializzabilità della funzione di valutazione
        try:
            cloudpickle.dumps(eval_fn)
            logger.info("Funzione di valutazione 'eval_fn' serializzabile")
        except Exception as e:
            logger.error(f"Errore serializzazione 'eval_fn': {e}")
            raise

        # Inizializza popolazione
        logger.info("Inizializzazione popolazione")
        pop = [self.toolbox.init_valid_ind() for _ in range(self.pop_size)]
        logger.info(f"Popolazione inizializzata con {len(pop)} individui")

        # Controllo serializzabilità di un individuo
        try:
            cloudpickle.dumps(pop[0])
            logger.info("Individuo 'pop[0]' serializzabile")
        except Exception as e:
            logger.error(f"Errore serializzazione individuo 'pop[0]': {e}")
            raise

        # Controllo serializzabilità della popolazione (opzionale)
        try:
            cloudpickle.dumps(pop)
            logger.info("Popolazione 'pop' serializzabile")
        except Exception as e:
            logger.error(f"Errore serializzazione 'pop': {e}")
            raise

        # Prima valutazione
        batch_size = 50
        logger.info(f"Valutazione iniziale in batch di {batch_size}")
        for i in range(0, len(pop), batch_size):
            batch = pop[i:i + batch_size]
            logger.debug(f"Invio batch {i//batch_size} con {len(batch)} individui")
            futures = self.client.map(self.toolbox.evaluate, batch)
            try:
                fits = self.client.gather(futures)
                for ind, fit in zip(batch, fits):
                    ind.fitness.values = fit
                logger.debug(f"Batch {i//batch_size} valutato con successo")
            except Exception as e:
                logger.error(f"Errore nel gather del batch {i//batch_size}: {e}")
                raise

        # Ciclo evolutivo
        for gen in range(1, self.num_generations + 1):
            logger.info(f"\n[RUN DEBUG] === Generation {gen} ===")

            # Selezione
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [creator.Individual(ind[:]) for ind in offspring]

            # Crossover
            logger.debug("Inizio crossover")
            for i in range(1, len(offspring), 2):
                if random.random() < self.cxpb:
                    self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            logger.debug("Fine crossover")

            # Mutazione
            logger.debug("Inizio mutazione")
            for ind in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values
            logger.debug("Fine mutazione")

            # Iniezione nuovi individui
            num_new = max(1, int(self.pop_size * self.new_ind_fraction))
            new_inds = [self.toolbox.init_valid_ind() for _ in range(num_new)]
            offspring[-num_new:] = new_inds
            logger.debug(f"Iniettati {num_new} nuovi individui")

            # Valuta solo gli invalidi
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            logger.info(f"Individui da valutare: {len(invalid)}")
            for i in range(0, len(invalid), batch_size):
                batch = invalid[i:i + batch_size]
                logger.debug(f"Invio batch {i//batch_size} gen {gen} con {len(batch)} individui")
                futures = self.client.map(self.toolbox.evaluate, batch)
                try:
                    fits = self.client.gather(futures)
                    for ind, fit in zip(batch, fits):
                        ind.fitness.values = fit
                    logger.debug(f"Batch {i//batch_size} gen {gen} valutato")
                except Exception as e:
                    logger.error(f"Errore gather batch {i//batch_size} gen {gen}: {e}")
                    raise

            pop[:] = offspring

            # Statistiche
            sizes = [ind.fitness.values[0] for ind in pop]
            costs = [ind.fitness.values[1] for ind in pop]
            if self.verbose:
                logger.info(f"[STATS] Gen {gen}: max size={max(sizes)}, min cost={min(costs)}")

        # Estrai Pareto front
        front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        valid_front = [ind for ind in front if ind.fitness.values[1] <= self.graph.budget]
        logger.info(f"[FINAL] Soluzioni Pareto entro budget: {len(valid_front)}")

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
        logger.info(f"[RESULT] Miglior seed set: {best_result['seed_set']} "
                    f"con fitness {best_result['fitness']} (budget={self.graph.budget})")

        return best_result['seed_set']