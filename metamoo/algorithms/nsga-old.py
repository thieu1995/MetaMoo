import numpy as np


class NSGA:
    def __init__(self, population_size, num_generations, num_objectives, bounds, crossover_prob=0.9, mutation_prob=0.1):
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_objectives = num_objectives
        self.bounds = bounds
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def initialize_population(self):
        """Randomly initialize the population within the bounds."""
        population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.population_size, len(self.bounds)))
        return population

    def evaluate_objectives(self, population):
        """Evaluate the objectives for each individual in the population."""
        objectives = np.zeros((self.population_size, self.num_objectives))
        # Define objective functions (for simplicity, we use example objectives)
        for i, individual in enumerate(population):
            objectives[i, 0] = individual[0]  # Example objective 1: Minimize x1
            objectives[i, 1] = 1 / individual[0] + individual[1]  # Example objective 2: Some arbitrary function
        return objectives

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (using Pareto dominance)."""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def non_dominated_sort(self, objectives):
        """Perform non-dominated sorting."""
        population_size = len(objectives)
        domination_count = np.zeros(population_size, dtype=int)
        dominated_solutions = [[] for _ in range(population_size)]
        fronts = [[]]

        for i in range(population_size):
            for j in range(population_size):
                if i != j:
                    if self.dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove last empty front

    def crossover(self, parent1, parent2):
        """Simulate crossover between two parents."""
        if np.random.rand() < self.crossover_prob:
            alpha = np.random.rand()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        """Mutate an individual by adding small random noise."""
        if np.random.rand() < self.mutation_prob:
            mutation_value = np.random.normal(0, 0.1, size=len(self.bounds))
            individual += mutation_value
            # Make sure the mutated individual is within bounds
            individual = np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])
        return individual

    def evolve(self):
        """Main loop of NSGA algorithm."""
        population = self.initialize_population()

        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}")

            # Step 1: Evaluate the population
            objectives = self.evaluate_objectives(population)

            # Step 2: Non-dominated sorting
            fronts = self.non_dominated_sort(objectives)

            # Step 3: Print the Pareto front (Front 0) for the current generation
            print("Pareto front (Front 0):")
            for idx in fronts[0]:
                print(f"Solution: {population[idx]}, Objectives: {objectives[idx]}")
            print("=" * 50)

            # Step 3: Generate offspring using crossover and mutation
            new_population = []
            while len(new_population) < self.population_size:
                parents = np.random.choice(np.arange(self.population_size), 2, replace=False)
                child1, child2 = self.crossover(population[parents[0]], population[parents[1]])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            # Step 4: Selection (replace the population with the new population)
            population = np.array(new_population[:self.population_size])

        # Final evaluation and return the non-dominated solutions
        final_objectives = self.evaluate_objectives(population)
        final_fronts = self.non_dominated_sort(final_objectives)
        return population, final_objectives, final_fronts

# Define the problem
population_size = 20
num_generations = 100
num_objectives = 2
bounds = np.array([[0.1, 5.0], [0.1, 5.0]])  # Variable bounds

# Initialize the NSGA
nsga = NSGA(population_size, num_generations, num_objectives, bounds)

# Run the algorithm
final_population, final_objectives, final_fronts = nsga.evolve()

# Display the non-dominated front (Pareto front)
print("Non-dominated solutions (Pareto front):")
for idx in final_fronts[0]:
    print(f"Solution: {final_population[idx]}, Objectives: {final_objectives[idx]}")

