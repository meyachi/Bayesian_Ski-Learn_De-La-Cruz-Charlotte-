"""
Scikit-learn

What is scikit-learn? 
Scikit-learn or also known as (sklearn) is one of the most widely used open‑source Python libraries for machine learning. It provides simple and efficient tools for data analysis, modeling, and prediction. Scikit-learn was built on top of other scientific Python libraries such as NumPy, SciPy, and Matplotlib. It is commonly used for classification, regression, clustering, dimensionality reduction, and model evaluation. (http://scikit-learn.org/stable/)

What are the Key-Features of Sci-kit Learn

Scikit-learn provides several features that make it a powerful library for machine learning in Python.

1. Supervised Learning
Scikit-learn supports supervised learning techniques used to predict outputs from labeled data. These include classification and regression algorithms such as logistic regression, decision trees, random forests, support vector machines, and linear regression.

2. Unsupervised Learning
The library also provides tools for unsupervised learning, which analyzes data without predefined labels. Examples include clustering methods like K-Means, DBSCAN, and hierarchical clustering, as well as dimensionality reduction techniques such as PCA.

3. Data Preprocessing Tools
Scikit-learn contains utilities that help prepare datasets before training models. These include data splitting, feature scaling, feature selection, and feature extraction, which improve model performance and data quality.

4. Model Evaluation and Selection
The library includes functions for evaluating machine learning models using metrics such as accuracy, precision, recall, F1-score, and mean squared error. It also supports techniques like grid search and randomized search for tuning model parameters.

5. Built-in Datasets and Easy-to-Use API
Scikit-learn offers several sample datasets for experimentation and has a consistent and user-friendly API, making it easier for beginners and researchers to build and test machine learning models.


https://www.geeksforgeeks.org/machine-learning/what-is-python-scikit-library/ 




Common Algorithms Available in Scikit-Learn

Scikit-learn includes a wide range of machine learning algorithms categorized into different types.

Classification Algorithms
Logistic Regression
Decision Trees
Random Forest
Support Vector Machines (SVM)
Gradient Boosting

Regression Algorithms
Linear Regression
Support Vector Regression
Decision Tree Regression

Clustering Algorithms
K-Means Clustering
DBSCAN
Hierarchical Clustering

Dimensionality Reduction Algorithms
Principal Component Analysis (PCA)

These algorithms allow developers and researchers to build models for prediction, pattern recognition, and data analysis.

Installing Scikit-learn
Scikit-learn can be installed using pip or conda, which are commonly used package managers for Python.

Using pip
pip install scikit-learn
Using conda
conda install -c conda-forge scikit-learn

After installation, the library can be imported in Python using:

import sklearn

These commands install the scikit-learn package along with its required dependencies so it can be used for machine learning tasks in Python.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/francescaquebec/Francesca-Quebec/main/Supervised%20datasets.csv"
data = pd.read_csv(url)

# Display first rows
print(data.head())

# Select features and target
X = data.drop(columns=['Number', 'species'])
y = data['species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, predictions))