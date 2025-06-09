import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from lr import LogisticRegression  # custom LogisticRegression class from earlier


# 1. Load the Breast Cancer dataset from sklearn
# This dataset is commonly used for binary classification (malignant vs. benign tumors)
bc = datasets.load_breast_cancer()

# Features (input variables): measurements of tumors (like radius, texture, etc.)
# Labels (targets): 0 = malignant, 1 = benign
X, y = bc.data, bc.target

# 2. Split the data into training and test sets
# 80% training, 20% test
# `random_state` ensures reproducibility (same split every time)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 3. Initialize and train the model

# This uses the custom Logistic Regression class from scratch (in `lr.py`)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# 4. Use the trained model to make predictions
y_pred = classifier.predict(X_test)

# 5. Define an accuracy function
# Accuracy = (# of correct predictions) / (total predictions)
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

# 6. Evaluate and print final accuracy
acc = accuracy(y_pred, y_test)
print(acc)  # Final accuracy: 0.8947368421052632
