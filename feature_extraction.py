import cv2
import numpy as np
import mahotas
import pandas as pd
import os

# Define the function to extract features
def extract_features(original_image, segmentation_mask):
    # Load the original image and segmentation mask
    original_image = cv2.imread(original_image)
    segmentation_mask = cv2.imread(segmentation_mask, cv2.IMREAD_GRAYSCALE)

    # Apply the segmentation mask to isolate the ROI
    roi = cv2.bitwise_and(original_image, original_image, mask=segmentation_mask)

    # Calculate features from the ROI
    # ... (insert your existing feature extraction code here)

    # For example, calculating area and perimeter:
    area = np.sum(segmentation_mask > 0)
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0

    # Color statistics
    mean_color = cv2.mean(roi, mask=segmentation_mask)[:3]  # Excluding the alpha channel if present

    # Texture descriptors using the gray-level co-occurrence matrix (GLCM)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    textures = mahotas.features.haralick(gray_roi).mean(axis=0)

    # Combine all features into a single vector
    features = np.concatenate([[area, perimeter], mean_color, textures])

    return features

# Function to iterate over the dataset and extract features
def extract_features_for_dataset(metadata_df, image_dir, segmentation_dir):
    feature_list = []
    for _, row in metadata_df.iterrows():
        # Construct the full paths to the images and segmentation masks
        image_path = os.path.join(image_dir, row['image_id'] + '.jpg')
        segmentation_mask_path = os.path.join(segmentation_dir, row['image_id'] + '_segmentation.png')
        
        # Extract features using the paths
        features = extract_features(image_path, segmentation_mask_path)
        feature_list.append(features)
    return feature_list

# Define paths
metadata_path = '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/HAM10000_metadata.csv'  # Update with your actual path
image_dir = '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/HAM10015_images'  # Update with your actual path
segmentation_dir = '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/HAM10015_segmentation'  # Update with your actual path
output_directory = '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output'  # Update with your actual path for saving the features CSV

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load metadata
metadata_df = pd.read_csv(metadata_path)

# Add full paths to the metadata DataFrame
metadata_df['image_path'] = metadata_df['image_id'].apply(lambda x: os.path.join(image_dir, x + '.jpg'))
metadata_df['segmentation_path'] = metadata_df['image_id'].apply(lambda x: os.path.join(segmentation_dir, x + '_segmentation.png'))

# Extract features for all images in the DataFrame
all_features = extract_features_for_dataset(metadata_df, image_dir, segmentation_dir)

# Convert to a DataFrame
feature_columns = ['Area', 'Perimeter', 'Mean_Color_R', 'Mean_Color_G', 'Mean_Color_B', 'Texture_Feature_1', 'Texture_Feature_2', 'Texture_Feature_3', 'Texture_Feature_4','Feature_10', 'Feature_11', 'Feature_12', 
    'Feature_13', 'Feature_14', 'Feature_15', 
    'Feature_16', 'Feature_17', 'Feature_18']
feature_df = pd.DataFrame(all_features, columns=feature_columns)

# Add image IDs to the DataFrame
feature_df['Image_ID'] = metadata_df['image_id']

# Save the features to a CSV file in the output directory
output_file = os.path.join(output_directory, 'extracted_features.csv')
feature_df.to_csv(output_file, index=False)

