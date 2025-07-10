from deap import base, creator, tools, algorithms
from Graph.graph import Graph
import random


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
        fitness_function = Graph.calc_majority_cascade_on_seed_set,
        verbose=True
    ):
        """
        Initialize the Genetic Algorithm engine.

        Parameters:
        - graph: Graph wrapper object (must implement get_node_num and cascade logic)
        - cxpb: Crossover probability (between 0 and 1)
        - mutpb: Mutation probability (between 0 and 1)
        - indpb_crossover: Probability of swapping a gene in uniform crossover
        - indpb_mutation: Probability of mutating a gene in scramble mutation
        - population_size: Number of individuals per generation
        - num_generations: Number of evolutionary generations
        - verbose: Whether to print progress each generation
        """
        self.graph = graph
        self.node_number = graph.get_node_num()

        # Genetic algorithm parameters
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.indpb_crossover = indpb_crossover
        self.indpb_mutation = indpb_mutation
        self.population_size = population_size
        self.num_generations = num_generations
        self.verbose = verbose

        self.node_list = self.graph.get_nodes_list()  # Must return list of graph node IDs, in fixed order
        self.fitness_function = fitness_function

        self._setup_deap()

    def _setup_deap(self):
        """
        Initialize DEAP components: creator, toolbox, and operators.
        """
        # Define fitness and individual structure (only once globally)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Each gene is a binary value (0 or 1)
        self.toolbox.register("attr_bool", random.randint, 0, 1)

        # An individual is a list of N binary genes
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_bool, self.node_number)

        # A population is a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Evaluation function using the Graph wrapper's cascade model
        self.toolbox.register("evaluate", self._evaluate)

        # Uniform crossover: genes are swapped with a given probability
        self.toolbox.register("mate", tools.cxUniform, indpb=self.indpb_crossover)

        # Scramble mutation: a slice of the genome is randomly shuffled
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=self.indpb_mutation)

        # Selection by fitness proportionate method (roulette wheel)
        self.toolbox.register("select", tools.selRoulette)


    def _evaluate(self, individual):
        """
        Evaluate the fitness of an individual by converting its binary genome
        to the actual graph node IDs forming the seed set.

        Parameters:
        - individual: list of 0/1 values representing node inclusion

        Returns:
        - tuple with single fitness value (spread)
        """
        # Map indices of bits set to actual node IDs
        seed_set = set(self.node_list[i] for i, gene in enumerate(individual) if gene == 1)

        # On calc_majority_cascade_on_seed_set this will be a set instead of a number
        spread_nodes = self.fitness_function(seed_set)

        # Ensure spread is always a number
        if isinstance(spread_nodes, (set, list)):
            spread = len(spread_nodes)
        elif isinstance(spread_nodes, (int, float)):
            spread = spread_nodes
        else:
            raise TypeError(f"Fitness function returned invalid type: {type(spread_nodes)}")

        return (spread, )


    def run(self):
        """
        Run the genetic algorithm and return the best solution.

        Returns:
        - best_individual: binary list of genes (0/1)
        - best_seed_set: list of actual graph node IDs included in the seed set
        - best_fitness: float or int, spread value
        """
        population = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)

        algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.num_generations,
            halloffame=hof,
            verbose=self.verbose
        )

        best_individual = hof[0]
        best_seed_set = [self.node_list[i] for i, gene in enumerate(best_individual) if gene == 1]
        best_fitness = best_individual.fitness.values[0]

        return best_individual, best_seed_set, best_fitness
