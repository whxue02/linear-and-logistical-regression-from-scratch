import numpy as np

# Sigmoid activation function
def sigmoid(x):
    # Sigmoid squashes any real-valued input into a value between 0 and 1
    # It's defined as: sigmoid(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


# Implementation of Logistic Regression from scratch
class LogisticRegression():
    def __init__(self, lr=0.001, n_inters=1000):
        """
        Constructor to initialize hyperparameters
        lr: Learning rate (controls how big the weight updates are)
        n_inters: Number of iterations (how many times to loop over the training data)
        """
        self.lr = lr                      # learning rate
        self.n_inters = n_inters          # number of training steps
        self.weights = None               # model weights (to be initialized in fit)
        self.bias = None                  # model bias (scalar)

    def fit(self, X, y):
        """
        Train the logistic regression model on the input data.
        X: Training features (numpy array of shape [n_samples, n_features])
        y: Target labels (numpy array of shape [n_samples])
        """
        n_samples, n_features = X.shape   # number of samples and features in the dataset
        
        # Initialize weights and bias to zero
        self.weights = np.zeros(n_features)  # one weight per feature
        self.bias = 0                         # single bias term

        # Training loop
        for _ in range(self.n_inters):
            # Step 1: Compute linear combination of inputs and weights
            # z = X·w + b
            linear_predictions = np.dot(X, self.weights) + self.bias

            # Step 2: Apply the sigmoid activation to get probabilities
            # This maps z to values between 0 and 1
            predictions = sigmoid(linear_predictions)

            # Step 3: Compute gradients for weights and bias using binary cross-entropy derivative

            # Gradient w.r.t. weights:
            # dw = (1 / n_samples) * X.T · (predictions - y)
            # This shows how much we need to adjust each weight
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))

            # Gradient w.r.t. bias:
            # db = (1 / n_samples) * sum(predictions - y)
            # How much the total prediction error affects the bias
            db = (1 / n_samples) * np.sum(predictions - y)

            # Step 4: Update parameters using gradient descent
            # Move weights and bias in the direction that reduces the loss
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        """
        Make binary predictions (0 or 1) on new data.
        X: Input features (numpy array of shape [n_samples, n_features])
        Returns: List of 0 or 1 predictions
        """
        # Compute linear combination of input features and trained weights
        linear_predictions = np.dot(X, self.weights) + self.bias

        # Apply sigmoid to get probabilities
        y_pred = sigmoid(linear_predictions)

        # Convert probabilities to binary classes using a threshold of 0.5
        # If probability > 0.5 → class 1, else class 0
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]

        return class_pred
