import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, hinge_loss
import os

# Define the output directory
output_dir = '/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output/svm outputs'
os.makedirs(output_dir, exist_ok=True)

# Load the datasets
features_df = pd.read_csv('/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/output/extracted_features.csv')
metadata_df = pd.read_csv('/Users/yeshwanthreddykatta/Desktop/GWU Semester 2/Machine Intelligence/Final project Information/MelanomaDetection/data/HAM10000_metadata.csv')

# Merge the datasets on 'Image_ID'
merged_df = pd.merge(features_df, metadata_df, left_on='Image_ID', right_on='image_id')

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
merged_df['full_lesion_type'] = merged_df['dx'].map(lesion_type_dict)

# Drop the identifier columns
merged_df = merged_df.drop(['Image_ID', 'image_id', 'dx'], axis=1)

# Drop rows with NaN values
merged_df = merged_df.dropna()

# Identify categorical and numeric columns
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in merged_df.columns if merged_df[col].dtype == 'object' and col != 'full_lesion_type']

# Separate features and target
X = merged_df.drop('full_lesion_type', axis=1)
y = merged_df['full_lesion_type']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformations
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Function to train and evaluate SVM classifier
def train_evaluate_svm(kernel_type):
    svm_classifier = SVC(kernel=kernel_type, decision_function_shape='ovo')
    svm_classifier.fit(X_train_transformed, y_train)

    # Predict and evaluate
    y_pred = svm_classifier.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {kernel_type} kernel: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title(f'Confusion Matrix - {kernel_type} Kernel')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{kernel_type}.png'))
    plt.close()

    # Classification Report
    report = classification_report(y_test, y_pred)
    with open(os.path.join(output_dir, f'classification_report_{kernel_type}.txt'), 'w') as file:
        file.write(report)
        file.write(f"\nAccuracy: {accuracy:.4f}")
train_evaluate_svm('linear')
# Train and evaluate with RBF kernel
train_evaluate_svm('rbf')

# Train and evaluate with Polynomial kernel
train_evaluate_svm('poly')



#print(f"Average hinge loss on the test set: {test_loss:.4f}")

print("Outputs saved in the output directory.")
