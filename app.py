# Importing necessary libraries
# Libraries for data manipulation, visualization, model training, and evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from collections import Counter

# Load the dataset
data = pd.read_csv("weather_forecast_data.csv")  # Replace with your dataset file name

# Task 1: Preprocessing
print("\nTask 1: Preprocessing")

# Step 1: Checking for missing data
# Identifies columns with missing values and their counts
print("Step 1: Checking for missing data:")
missing_data = data.isnull().sum()
print(missing_data)

# Step 2: Handling missing data
print("\nStep 2: Handling missing data")

# Dropping rows with missing values
# Removes rows that have any missing values
data_dropped = data.dropna()

# Replacing missing values with the mean of each numeric column
# Ensures numeric columns are filled with the column's mean
data_filled = data.copy()
for column in data.select_dtypes(include=["float64", "int64"]).columns:
    data_filled[column] = data_filled[column].fillna(data_filled[column].mean())

# Encoding the target column if it is non-numeric
# Converts categorical target variable to numeric if necessary
if "target" in data_filled.columns and data_filled["target"].dtype == 'object':
    encoder = LabelEncoder()
    data_filled["target"] = encoder.fit_transform(data_filled["target"])
    print("Target column encoded:", encoder.classes_)

# Check class distribution
# Displays the distribution of target variable values
print("\nClass Distribution in Target:")
if "target" in data_filled.columns:
    print(data_filled["target"].value_counts())

# Step 3: Feature scaling
# Scales numeric features to a standard range (mean=0, variance=1)
print("\nStep 3: Feature scaling")
numeric_features = data_filled.select_dtypes(include=["float64", "int64"]).columns
scaler = StandardScaler()
data_scaled = data_filled.copy()
data_scaled[numeric_features] = scaler.fit_transform(data_filled[numeric_features])

# Step 4: Splitting the data
# Divides the dataset into training and testing subsets
print("\nStep 4: Splitting the data")
X = data_scaled.drop("Rain", axis=1)  # Features (independent variables)
y = data_scaled["Rain"]  # Target (dependent variable)

# Splitting into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Step 5: Training models using scikit-learn
print("\nStep 5: Training a Decision Tree Model")

# Initializing the decision tree model
# DecisionTreeClassifier is used for classification tasks
dt = DecisionTreeClassifier()

# Training the model
# Fits the model to the training data
dt.fit(X_train, y_train)

# Step 6: Making predictions
# Predicts outcomes on the test data
print("\nStep 6: Making predictions")
y_pred = dt.predict(X_test)

# Step 7: Evaluating the model
# Calculates accuracy, precision, and recall for the Decision Tree model
print("\nStep 7: Evaluating the model")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Decision Tree model: {accuracy:.2f}")

print("\nDecision Tree Metrics:")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")

# Feature importance for Decision Tree
# Identifies which features are most impactful in the tree's decisions
print("\nFeature Importances (Decision Tree):")
feature_importances = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances)

# Save decision tree plot
# Visualizes the Decision Tree and saves it as a file
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=["No Rain", "Rain"])
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree_visualization.png")
plt.show()  # Display the plot in the output

# Print the decision tree rules as text
# Explains the tree's logic with textual descriptions of node splits
tree_rules = export_text(dt, feature_names=list(X.columns))
print("\nDecision Tree Rules:")
print(tree_rules)

# Training a Naive Bayes Model
print("\nNaïve Bayes Model")
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Evaluating the Naive Bayes Model
print("\nNaïve Bayes Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_nb, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_nb, average='weighted'):.2f}")

# Custom kNN Model Implementation
class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = np.linalg.norm(self.X_train - test_point, axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            predictions.append(Counter(k_labels).most_common(1)[0][0])
        return predictions

custom_knn = CustomKNN(k=5)
custom_knn.fit(X_train.values, y_train.values)
custom_predictions = custom_knn.predict(X_test.values)

# Evaluate Custom kNN Model
print("\nCustom kNN Metrics:")
print(f"Accuracy: {accuracy_score(y_test, custom_predictions):.2f}")
print(f"Precision: {precision_score(y_test, custom_predictions, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_test, custom_predictions, average='weighted'):.2f}")

# Scikit-learn kNN Model
print("\nScikit-learn kNN Model")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Evaluate Scikit-learn kNN Model
print("\nScikit-learn kNN Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_knn, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_knn, average='weighted'):.2f}")

# Performance Comparison for Custom and Scikit-learn kNN Models
print("\nPerformance Comparison between Custom kNN and Scikit-learn kNN:")
print(f"Custom kNN Accuracy: {accuracy_score(y_test, custom_predictions):.2f}")
print(f"Scikit-learn kNN Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
print(f"Custom kNN Precision: {precision_score(y_test, custom_predictions, average='weighted'):.2f}")
print(f"Scikit-learn kNN Precision: {precision_score(y_test, y_pred_knn, average='weighted'):.2f}")
print(f"Custom kNN Recall: {recall_score(y_test, custom_predictions, average='weighted'):.2f}")
print(f"Scikit-learn kNN Recall: {recall_score(y_test, y_pred_knn, average='weighted'):.2f}")

# Evaluate kNN with Different k Values
for k_value in [1, 3, 5, 7, 9]:
    custom_knn = CustomKNN(k=k_value)
    custom_knn.fit(X_train.values, y_train.values)
    custom_predictions = custom_knn.predict(X_test.values)
    print(f"\nMetrics for k={k_value} in Custom kNN:")
    print(f"Accuracy: {accuracy_score(y_test, custom_predictions):.2f}")
    print(f"Precision: {precision_score(y_test, custom_predictions, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, custom_predictions, average='weighted'):.2f}")
