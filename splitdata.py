import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the metadata CSV file
metadata_path = '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/dataverse_files/HAM10000_metadata.csv'

# Directory where the split metadata CSV files will be saved
output_dir = '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/dataverse_files'

# Load the metadata
metadata_df = pd.read_csv(metadata_path)

# Assuming 'dx' is the column with the diagnosis or category labels
# Replace 'dx' with the actual name of your label column
X = metadata_df.drop('dx', axis=1)
y = metadata_df['dx']

# Splitting the dataset into training (72%), validation (18%), and test (10%) sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.28, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.35714, stratify=y_val_test, random_state=42)

# Combine X and y for each split
train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the split metadata
train_data.to_csv(output_dir + 'train_metadata.csv', index=False)
val_data.to_csv(output_dir + 'val_metadata.csv', index=False)
test_data.to_csv(output_dir + 'test_metadata.csv', index=False)

# Print sizes of each dataset to verify the splits
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")
