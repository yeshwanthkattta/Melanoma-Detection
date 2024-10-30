import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from glob import glob

# Set your base directory for the skin cancer dataset
base_skin_dir = "/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/dataverse_files/HAM10015_images"  # Update with your path
metadata_path = os.path.join(base_skin_dir, '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/dataverse_files/HAM10000_metadata.csv')

# Load the metadata
skin_df = pd.read_csv(metadata_path)

# Map image paths
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
skin_df["path"] = skin_df["image_id"].map(imageid_path_dict.get)

# Mapping lesion types
lesion_type_dict = {
    'nv': 'Melanocytic_nevi',
    'mel': 'melanoma',
    'bkl': 'Benign_keratosis-like_lesions',
    'bcc': 'Basal_cell_carcinoma',
    'akiec': 'Actinic_keratoses',
    'vasc': 'Vascular_lesions',
    'df': 'Dermatofibroma'
}
skin_df["cell_type"] = skin_df["dx"].map(lesion_type_dict.get)

# Mapping malignant or benign
lesion_danger = {
    'nv': 0,  # 0 for benign
    'mel': 1,  # 1 for malignant
    'nv': 0, # 0 for benign
    'mel': 1, # 1 for malignant
    'bkl': 0, # 0 for benign
    'bcc': 1, # 1 for malignant
    'akiec': 1, # 1 for malignant
    'vasc': 0,
    'df': 0
}
skin_df["Malignant"] = skin_df["dx"].map(lesion_danger.get)

# Basic Information about the Dataset
print("Basic Information about the Dataset:")
print(skin_df.info())
print("\n")

# Check for missing values
print("Missing Values in Each Column:")
print(skin_df.isnull().sum())
print("\n")

# Distribution of lesion types (diagnosis)
plt.figure(figsize=(15, 10))
sns.countplot(x='cell_type', data=skin_df)
plt.title('Distribution of Lesion Types (Diagnosis)')
plt.savefig('/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output/EDA OUTPUTS/lesion_type_distribution.png')  # Update path
plt.show()

# Age distribution among patients
plt.figure(figsize=(10, 6))
sns.histplot(skin_df['age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution Among Patients')
plt.savefig('/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output/EDA OUTPUTS/age_distribution.png')  # Update path
plt.show()

# Gender distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', data=skin_df)
plt.title('Gender Distribution')
plt.savefig('/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output/EDA OUTPUTS/gender_distribution.png')  # Update path
plt.show()

# Localization of lesions
plt.figure(figsize=(10, 6))
sns.countplot(y='localization', data=skin_df, order = skin_df['localization'].value_counts().index)
plt.title('Localization of Lesions')
plt.savefig('/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output/EDA OUTPUTS/localization_distribution.png')  # Update path
plt.show()

# Diagnosis Type distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='dx_type', data=skin_df)
plt.title('Diagnosis Type Distribution')
plt.savefig('/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output/EDA OUTPUTS/diagnosis_type_distribution.png')  # Update path
plt.show()

# Benign vs Malignant Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Malignant', data=skin_df)
plt.title('Benign vs Malignant Distribution')
plt.savefig('/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output/EDA OUTPUTS/malignant_distribution.png')  # Update path
plt.show()

print("EDA complete. Plots saved to specified directory.")
