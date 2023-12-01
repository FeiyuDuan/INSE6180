import numpy as np
from sklearn.preprocessing import LabelBinarizer

class SVMMulti:

    # def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
    #     self.lr = learning_rate
    #     self.lambda_param = lambda_param
    #     self.n_iters = n_iters
    #     self.w = None
    #     self.b = None

    # def fit(self, X, y):
    #     n_samples, n_features = X.shape
    #     self.w = np.zeros((n_features, len(np.unique(y))))  # One weight vector per class
    #     self.b = np.zeros(len(np.unique(y)))  # One bias term per class

    #     self.classes_ = np.unique(y)
    #     y_encoded = LabelBinarizer().fit_transform(y)
    #     y_encoded = np.where(y_encoded == 1, 1, -1)

    #     for i, y_ in enumerate(y_encoded.T):  # Train a separate SVM for each class
    #         # Initialize weights and bias for the current class
    #         self.w[:, i] = np.zeros(n_features)
    #         self.b[i] = 0

    #         # Perform gradient descent for current class
    #         for _ in range(self.n_iters):
    #             for idx, x_i in enumerate(X):
    #                 condition = y_[idx] * (np.dot(x_i, self.w[:, i]) - self.b[i]) >= 1
    #                 if condition:
    #                     self.w[:, i] -= self.lr * (2 * self.lambda_param * self.w[:, i])
    #                 else:
    #                     self.w[:, i] -= self.lr * (2 * self.lambda_param * self.w[:, i] - np.dot(x_i, y_[idx]))
    #                     self.b[i] -= self.lr * y_[idx]

    # def predict(self, X):
    #     approximations = np.dot(X, self.w) - self.b.reshape(1, -1)
    #     return self.classes_[np.argmax(approximations, axis=1)]
    #     # Calculate the decision function for each classifier
    #     # decision_function = np.dot(X, self.w) - self.b
    #     # # Choose the class with the highest score
    #     # return self.classes_[np.argmax(decision_function, axis=1)]
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Initialize weights and biases for each class
        self.w = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        # Convert labels to one-vs-rest format
        y_ovr = np.array([np.where(y == class_, 1, -1) for class_ in range(n_classes)])

        # Train a separate classifier for each class
        for class_idx in range(n_classes):
            y_ = y_ovr[class_idx]
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    condition = y_[idx] * (np.dot(x_i, self.w[class_idx]) - self.b[class_idx]) >= 1
                    if condition:
                        self.w[class_idx] -= self.lr * (2 * self.lambda_param * self.w[class_idx])
                    else:
                        self.w[class_idx] -= self.lr * (2 * self.lambda_param * self.w[class_idx] - np.dot(x_i, y_[idx]))
                        self.b[class_idx] -= self.lr * y_[idx]

    def predict(self, X):
        # Calculate the decision function for each class
        approx = np.dot(X, self.w.T) - self.b
        # Pick the class with the highest decision function value
        return np.argmax(approx, axis=1)

# Usage Example
# model = SVM()
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
