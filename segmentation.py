# segmentation.py
import os
from hybrid_aco_ga import hybrid_ACO_GA
from utils import load_image

def segment_images(metadata,image_dir,aco_params, ga_params):
    segmented_images = []

    for index, row in metadata.iterrows():
        image_filename = row['image_id'] + '.jpg'  # Ensure your metadata has 'image_id'
        image_path = os.path.join(image_dir, image_filename)
        image = load_image(image_path)
        segmented_image = hybrid_ACO_GA(image, aco_params, ga_params)
        segmented_images.append((image_path, segmented_image))  # Save as a tuple with path

    return segmented_images
