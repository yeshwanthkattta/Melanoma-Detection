import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
file_path = '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output/extracted_features.csv'
output_directory = '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output/eda feature extraction/'
data = pd.read_csv(file_path)

# Basic EDA
print("First few rows of the dataset:")
print(data.head())

print("\nSummary Statistics:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

# Visualize distributions of each feature
for col in data.drop('Image_ID', axis=1).columns:  # Assuming 'Image_ID' is not a feature
    plt.figure(figsize=(10, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(output_directory + f'distribution_{col}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to avoid displaying it in the notebook

# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='viridis')
plt.title('Feature Correlation Matrix')
plt.savefig(output_directory + 'correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature Selection
X = data.drop(['Image_ID'], axis=1)  # Drop non-feature columns
y = data['target']

# Use RandomForest for feature importances
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_

# Get feature importances and sort them
features = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
features.sort_values(by='Importance', ascending=False, inplace=True)

# Save the feature importances
features.to_csv(output_directory + 'feature_importances.csv', index=False)

print("\nFeature Importances:")
print(features)

variances = X.var().sort_values(ascending=False)

# Select the top 5 features with the highest variance
top_5_features = variances.head(5).index.tolist()

print("Top 5 features based on variance:")
print(top_5_features)

# Save the top 5 features
with open(output_directory + 'top_5_features.txt', 'w') as file:
    file.write('\n'.join(top_5_features))

# Optionally, you can use SelectKBest for feature selection
# selector = SelectKBest(f_classif, k=5)
# X_new = selector.fit_transform(X, y)
# top_5_features_kbest = X.columns[selector.get_support(indices=True)]
# print("\nTop 5 Features using SelectKBest:")
# print(top_5_features_kbest)
