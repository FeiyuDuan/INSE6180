import numpy as np

class NaiveBayesMulti:
    
    def fit(self, X, y, ls=0.01):
        self.ls = ls
        self.y_classes, y_counts = np.unique(y, return_counts=True)
        self.x_classes = [np.unique(x) for x in X.T]
        self.phi_y = 1.0 * y_counts/y_counts.sum()
        self.phi_x = self.mean_X(X, y)
        self.c_x = self.count_x(X, y)
        return self
    
    def mean_X(self, X, y):
        return [[self.ls_mean_x(X, y, k, j) for j in range(len(self.x_classes))] for k in self.y_classes]
    
    def ls_mean_x(self, X, y, k, j):
        x_data = (X[:,j][y==k].reshape(-1,1) == self.x_classes[j])
        return (x_data.sum(axis=0) + self.ls ) / (len(x_data) + (len(self.x_classes) * self.ls))
    
    def get_mean_x(self, y, j):
        return 1 + self.ls / (self.c_x[y][j] + (len(self.x_classes) * self.ls))
        
    def count_x(self, X, y):
        return [[len(X[:,j][y==k].reshape(-1,1) == self.x_classes[j])
                       for j in range(len(self.x_classes))]
                      for k in self.y_classes]

    def predict(self, X):
        return np.apply_along_axis(lambda x: self.compute_probs(x), 1, X)
    
    def compute_probs(self, x):
        probs = np.array([self.compute_prob(x, y) for y in range(len(self.y_classes))])
        return self.y_classes[np.argmax(probs)]
    
    def compute_prob(self, x, y):
        Pxy = 1
        for j in range(len(x)):
            x_clas = self.x_classes[j]
            if x[j] in x_clas:
                i = list(x_clas).index(x[j])
                p_x_j_y = self.phi_x[y][j][i] # p(xj|y)
                Pxy *= p_x_j_y
            else:
                Pxy *= self.get_mean_x(y, j)
        return Pxy * self.phi_y[y]
    
    def evaluate(self, X, y):
        return (self.predict(X) == y).mean()