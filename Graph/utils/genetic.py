from deap import base, creator, tools, algorithms
from Graph.graph import Graph
import random
import multiprocessing
from functools import partial
from math import ceil


def evaluate_individual(individual, node_list, fitness_function, cost_function, graph):
    """
    Evaluate the fitness of an individual by converting its binary genome
    to the actual graph node IDs forming the seed set.

    Parameters:
    - individual: list of 0/1 values representing node inclusion (genes)
    - node_list: list of graph node IDs in fixed order to map genes to nodes
    - fitness_function: function to calculate spread (influence) from seed set

    Returns:
    - tuple with single fitness value (spread), as required by DEAP
    """
    # Extract seed nodes where gene=1
    seed_set = set(node_list[i] for i, gene in enumerate(individual) if gene == 1)

    # Calculate spread or influence by applying fitness function on seed set
    spread_nodes = fitness_function(seed_set) 

    # Convert spread result to a numeric value (length if set/list, or numeric directly)
    if isinstance(spread_nodes, (set, list)):
        spread = len(spread_nodes)
    elif isinstance(spread_nodes, (int, float)):
        spread = spread_nodes
    else:
        raise TypeError(f"Fitness function returned invalid type: {type(spread_nodes)}")
    
    """cost_seed_set = graph.cost_seed_set(seed_set, cost_function)
    if cost_seed_set < graph.budget:
        return (ceil(spread / cost_seed_set),)
    else:
        return (0,)"""
    
    return (ceil(spread / graph.cost_seed_set(seed_set, cost_function)), )  # Return as a tuple for DEAP compatibility


class GeneticAlgo:
    def __init__(
        self,
        graph: Graph,
        cxpb,
        mutpb,
        indpb_crossover,
        indpb_mutation,
        population_size,
        num_generations,
        cost_function,
        fitness_function=Graph.calc_majority_cascade_on_seed_set,
        verbose=True
    ):
        """
        Initialize the genetic algorithm engine with parameters and graph.

        Parameters:
        - graph: Graph object implementing methods for nodes and cascade
        - cxpb: probability of crossover between individuals (0 to 1)
        - mutpb: probability of mutation for individuals (0 to 1)
        - indpb_crossover: probability of swapping gene during uniform crossover
        - indpb_mutation: probability of mutating each gene in shuffle mutation
        - population_size: number of individuals per generation
        - num_generations: number of generations to evolve
        - fitness_function: function to evaluate fitness, defaults to graph method
        - verbose: whether to print algorithm progress during execution
        """
        self.graph = graph
        self.node_number = graph.get_node_num()  # Number of nodes in the graph

        self.cxpb = cxpb
        self.mutpb = mutpb
        self.indpb_crossover = indpb_crossover
        self.indpb_mutation = indpb_mutation
        self.population_size = population_size
        self.num_generations = num_generations
        self.verbose = verbose
        self.cost_function = cost_function

        # List of nodes in fixed order, used to map genome bits to actual nodes
        self.node_list = self.graph.get_nodes_list()
        self.fitness_function = fitness_function

        # Setup DEAP toolbox, creators and operators
        self._setup_deap()

    def _setup_deap(self):
        """
        Initialize DEAP's creator, toolbox and register genetic operators.
        """

        # Create a maximizing fitness (single objective)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # Define an individual as a list with the maximizing fitness attribute
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create toolbox to register functions and operators
        self.toolbox = base.Toolbox()

        # Attribute generator: random bit 0 or 1
        self.toolbox.register("attr_bool", random.randint, 0, 1)

        # Individual generator: list of binary genes, length = number of graph nodes
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_bool, self.node_number)

        # Population generator: list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register crossover operator: uniform crossover with gene swap prob indpb_crossover
        self.toolbox.register("mate", tools.cxUniform, indpb=self.indpb_crossover)

        # Register mutation operator: shuffle mutation with probability indpb_mutation per gene
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=self.indpb_mutation)

        # Register selection operator: roulette wheel selection by fitness proportion
        self.toolbox.register("select", tools.selRoulette)

        # Default map function to Python's built-in map (serial evaluation)
        self.toolbox.register("map", map)

        # Note: Evaluation function will be registered dynamically in run() for multiprocessing

    def run(self):
        """
        Run the genetic algorithm optimization.

        Uses multiprocessing Pool for parallel evaluation of individuals.

        Returns:
        - best_individual: the fittest individual (binary gene list)
        - best_seed_set: list of graph nodes corresponding to 1 genes in best_individual
        - best_fitness: fitness value (spread) of the best individual
        """

        # Create a multiprocessing pool for parallel evaluation
        with multiprocessing.Pool() as pool:
            # Use functools.partial to bind node_list and fitness_function to the evaluator
            eval_func = partial(evaluate_individual,
                                node_list=self.node_list,
                                fitness_function=self.fitness_function,
                                cost_function=self.cost_function,
                                graph=self.graph)

            # Register the evaluation function to the toolbox
            self.toolbox.register("evaluate", eval_func)

            # Register pool.map for parallel evaluation in DEAP
            self.toolbox.register("map", pool.map)

            # Initialize population
            population = self.toolbox.population(n=self.population_size)

            # Hall of Fame to store best individual found
            hof = tools.HallOfFame(1)

            # Run the simple evolutionary algorithm with parameters set
            algorithms.eaSimple(
                population,
                self.toolbox,
                cxpb=self.cxpb,
                mutpb=self.mutpb,
                ngen=self.num_generations,
                halloffame=hof,
                verbose=self.verbose
            )

        # Extract the best individual after evolution
        best_individual = hof[0]

        # Convert best individual's genome to actual node IDs in the seed set
        best_seed_set = [self.node_list[i] for i, gene in enumerate(best_individual) if gene == 1]

        # Get fitness value (spread) of the best individual
        best_fitness = best_individual.fitness.values[0]
        
        print(f"Best_individual: {best_individual}")
        print(f"Best_seed_set: {best_seed_set}")
        print(f"Best_fitness: {best_fitness}")
        print(f"Budget: {self.graph.budget}")
        print(f"Seed set cost: {self.graph.cost_seed_set(best_seed_set, self.cost_function)}")

        return best_individual, best_seed_set, best_fitness
