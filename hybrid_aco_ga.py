# hybrid_aco_ga.py
from aco import ACO
from ga import GA
import numpy as np


def hybrid_ACO_GA(image, aco_params, ga_params):
    aco = ACO(**aco_params)
    ga = GA(**ga_params)

    # Run ACO
    aco_segmentation = aco.segment_image(image)

    # Prepare GA
    initial_population_seed = np.where(aco_segmentation > np.mean(aco_segmentation), 1, 0)
    initial_population = ga.initialize_population(image.shape[:2])
    initial_population[0] = initial_population_seed

    # Run GA
    ga_segmentation = ga.segment_image(image, initial_population)

    return ga_segmentation
