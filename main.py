import pandas as pd
import os
from segmentation import segment_images
#from feature_extraction import extract_features
#from classification import train_model, evaluate_model
from utils import save_segmented_images
# ACO and GA configuration parameters
aco_params = {
    'ants': 10,
    'max_iterations': 100,
    'alpha': 1.0,
    'beta': 1.0,
    'rho': 0.5,
    'Q': 100
}

ga_params = {
    'population_size': 50,
    'max_generations': 100,
    'crossover_rate': 0.8,
    'mutation_rate': 0.02
}
def main():
    print("Starting the segmentation process.")
    # Define paths to your metadata files and directories for segmented images
    base_dir = ('/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/HAM10015_images')
    train_metadata_path = os.path.join(base_dir, '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/train_metadata.csv')
    val_metadata_path = os.path.join(base_dir, '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/val_metadata.csv')
    test_metadata_path = os.path.join(base_dir, '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/test_metadata.csv')
    segmented_images_dir = os.path.join(base_dir, '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/segmented_images')
    image_dir = os.path.join(base_dir, '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/HAM10015_images')

    # Ensure the segmented_images_dir exists
    if not os.path.exists(segmented_images_dir):
        os.makedirs(segmented_images_dir)
        print(f"Created segmented images directory: {segmented_images_dir}")
    # Load metadata into DataFrames
    print("Loading metadata...")
    train_metadata = pd.read_csv(train_metadata_path)
    val_metadata = pd.read_csv(val_metadata_path)
    test_metadata = pd.read_csv(test_metadata_path)
    print("Metadata loaded successfully.")


    # Preprocess, segment images, and save segmented images
   # print("Segmenting training images...")
    #segmented_train_images = segment_images(train_metadata, image_dir, aco_params, ga_params)
    #print("Segmenting validation images...")
   # segmented_val_images = segment_images(val_metadata, image_dir, aco_params, ga_params)
    #print("Segmenting test images...")
    #segmented_test_images = segment_images(test_metadata, image_dir, aco_params, ga_params)
    for dataset_type, metadata in [('train', train_metadata), ('val', val_metadata), ('test', test_metadata)]:
        print(f"Segmenting {dataset_type} images...")
        segmented_images = segment_images(metadata, image_dir, aco_params, ga_params)
        save_segmented_images(segmented_images, os.path.join(segmented_images_dir, dataset_type))
        print(f"Segmented {dataset_type} images saved successfully.")
    # Save segmented images to disk
    #save_segmented_images(segmented_train_images, os.path.join(segmented_images_dir, 'train'))
    #save_segmented_images(segmented_val_images, os.path.join(segmented_images_dir, 'val'))
    #save_segmented_images(segmented_test_images, os.path.join(segmented_images_dir, 'test'))
    #print("Segmented images saved successfully.")

def new_func(image_dir):
    return image_dir
    # Extract features from segmented images
    #train_features, train_labels = extract_features(segmented_train_images)
    #val_features, val_labels = extract_features(segmented_val_images)
    #test_features, test_labels = extract_features(segmented_test_images)
    
    # Train and validate model
    #model = train_model(train_features, train_labels)
    #validate_model(model, val_features, val_labels)
    
    # Test model
    #test_results = evaluate_model(model, test_features, test_labels)
    
    # Output results
    #print(test_results)
    # Optionally, save the model and results to files

if __name__ == "__main__":
    main()
