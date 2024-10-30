import numpy as np
import cv2
import random

class GA:
    def __init__(self, population_size, max_generations, crossover_rate, mutation_rate):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def initialize_population(self, image_shape):
        return np.random.randint(2, size=(self.population_size, *image_shape))

    def fitness(self, individual, original_image):
        if len(original_image.shape) == 3:
            grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = original_image
        # Define a fitness function. For now, let's just use the sum of the individual
        return np.sum(individual * grayscale_image)

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, parent1.shape[0])
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual):
        for i in range(individual.shape[0]):
            for j in range(individual.shape[1]):
                if random.random() < self.mutation_rate:
                    individual[i, j] = 1 - individual[i, j]
        return individual

    def segment_image(self, image, initial_population=None):
        if len(image.shape) == 3:  # If image is color, convert to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = image

        height, width = grayscale_image.shape
        population = initial_population if initial_population is not None else self.initialize_population((height, width))

        for generation in range(self.max_generations):
            # Evaluate fitness
            fitness_values = [self.fitness(individual, image) for individual in population]

            # Select parents (roulette wheel selection)
            total_fitness = sum(fitness_values)
            selected_parents = []
            for _ in range(self.population_size // 2):
                rand_point = random.uniform(0, total_fitness)
                accumulator = 0
                for i in range(self.population_size):
                    accumulator += fitness_values[i]
                    if accumulator >= rand_point:
                        selected_parents.append(population[i])
                        break

                    # Ensure an even number of selected_parents
            if len(selected_parents) % 2 != 0:
                selected_parents.pop()
# Crossover and mutation
            new_population = []
            for i in range(0, len(selected_parents), 2):
                child1, child2 = self.crossover(selected_parents[i], selected_parents[i+1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            population = np.array(new_population)

        # Return the best individual as the segmented image
        best_individual = population[np.argmax(fitness_values)]
        return best_individual * 255

