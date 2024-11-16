from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import numpy as np


class LogisticRegressionGD(BaseEstimator, ClassifierMixin):
    
    def __init__(self, lr=0.01, epochs=100, threshold=0.5, random_state=0):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.threshold = threshold
        self.random_state= random_state
        self.loss= None

    def sigmoid(self, z):
       
        return np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)

    def fit(self, X, y):
        # Storing the unique classes seen during fit
        self.classes_ = unique_labels(y)
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Adding intercept
        np.random.seed(self.random_state)
        self.weights = np.random.randn(X.shape[1])
        self.loss=[]
        for epoch in range(self.epochs):
            predictions = self.sigmoid(np.dot(X, self.weights))
            errors = predictions - y
            gradient = np.dot(X.T, errors)
            self.weights -= self.lr * gradient
            # Optional: calculate the loss at each step (for monitoring/plotting purposes)
            self.loss.append(self.binary_cross_entropy_loss(y, predictions))
        return self

    def predict_proba(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Ensure intercept
        probabilities = self.sigmoid(np.dot(X, self.weights))
        return np.c_[1 - probabilities, probabilities]

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)               #0.492

    def binary_cross_entropy_loss(self, y, predictions):
        # Avoiding log(0) by adding a small number to predictions
        predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    
    



from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import numpy as np

class MultinomialLogisticRegressionGD(BaseEstimator, ClassifierMixin):
    
    def __init__(self, lr=0.01, epochs=100, random_state=0):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.random_state = random_state
        self.loss= None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        # Storing the unique classes seen during fit
        self.classes_ = unique_labels(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Adding intercept
        np.random.seed(self.random_state)
        self.weights = np.random.randn(X.shape[1], n_classes)
        
        # Transform y into a one-hot encoded version
        y_encoded = np.zeros((X.shape[0], n_classes))
        for idx, class_val in enumerate(self.classes_):
            y_encoded[np.where(y == class_val), idx] = 1
        self.loss=[]
        for epoch in range(self.epochs):
            predictions = self.softmax(np.dot(X, self.weights))
            errors = predictions - y_encoded
            gradient = np.dot(X.T, errors) 
            self.weights -= self.lr * gradient
            # Optional: calculate the loss at each step (for monitoring/plotting purposes)
            self.loss.append(self.cross_entropy_loss(y_encoded, predictions))
        return self

    def predict_proba(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Ensure intercept
        return self.softmax(np.dot(X, self.weights))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def cross_entropy_loss(self, y_encoded, predictions):
        # Avoiding log(0) by adding a small number to predictions
        predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
        return -np.sum(y_encoded * np.log(predictions)) 


