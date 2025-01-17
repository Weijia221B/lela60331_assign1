import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, num_features, num_classes, learning_rate=0.01, use_sigmoid=False):
        """
        Initialize logistic regression model
        Args:
            num_features: Number of input features for the model
            num_classes: Number of target classes for classification
            learning_rate: Learning rate for gradient descent optimization (default: 0.01)
            use_sigmoid: Boolean flag to use sigmoid activation for binary classification (default: False)      
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.lr = learning_rate
        self.use_sigmoid = use_sigmoid
        
        # Initialize weights based on classification type
        if use_sigmoid and num_classes == 2:
            self.weights = np.random.rand(num_features, 1)
        else:
            self.weights = np.random.rand(num_features, num_classes)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, n_iters=2500):
        """Train the model"""
        num_samples = len(X)
        logistic_loss = []
        
        if self.use_sigmoid and self.num_classes == 2:
            # Binary classification - use sigmoid
            y = np.array(y).reshape(-1, 1)
            
            for i in tqdm(range(n_iters), desc="Training"):
                # Forward propagation
                z = X.dot(self.weights)
                h = self.sigmoid(z)
                
                # Calculate loss
                loss = -np.mean(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))
                logistic_loss.append(loss)
                
                # Backpropagation
                dw = X.T.dot(h - y) / num_samples
                self.weights = self.weights - (dw * self.lr)
                
        else:
            # Multi-class classification - use softmax
            unique_labels = list(set(y))
            unique_one_hot = np.diag(np.ones(len(unique_labels)))
            y = np.array([list(unique_one_hot[unique_labels.index(label)]) for label in y])
            
            for i in tqdm(range(n_iters), desc="Training"):
                # Forward propagation
                z = X.dot(self.weights)
                z_sum = np.exp(z).sum(axis=1)
                q = np.array([list(np.exp(z_i)/z_sum[i]) for i, z_i in enumerate(z)])
                
                # Calculate loss
                loss = np.mean(-np.log2((np.sum((y*q), axis=1))))
                logistic_loss.append(loss)
                
                # Backpropagation
                dw = X.T.dot((q-y))/num_samples
                self.weights = self.weights - (dw*self.lr)
        
        return logistic_loss
    
    def predict(self, X):
        """Predict class"""
        if self.use_sigmoid and self.num_classes == 2:
            # Binary classification prediction
            z = X.dot(self.weights)
            probabilities = self.sigmoid(z)
            return (probabilities >= 0.5).astype(int).flatten()
        else:
            # Multi-class prediction
            z = X.dot(self.weights)
            z_sum = np.exp(z).sum(axis=1)
            q = np.array([list(np.exp(z_i)/z_sum[i]) for i, z_i in enumerate(z)])
            return np.argmax(q, axis=1)
    
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        metrics = {}
        
        if self.num_classes == 2:
            # Binary classification
            y_true = np.array(y_true).flatten()
            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            FN = np.sum((y_pred == 0) & (y_true == 1))
            TN = np.sum((y_pred == 0) & (y_true == 0))
            
            # Calculate accuracy
            accuracy = (TP + TN) / len(y_true)
            
            precision = TP/(TP+FP) if (TP+FP) > 0 else 0
            recall = TP/(TP+FN) if (TP+FN) > 0 else 0
            
            metrics['accuracy'] = accuracy
            metrics['positive_class'] = {
                'precision': precision,
                'recall': recall
            }
            
        else:
            # Multi-class classification
            unique_labels = list(set(y_true))
            y_true_idx = np.array([unique_labels.index(label) for label in y_true])
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_true_idx)
            metrics['accuracy'] = accuracy
            
            # Evaluate each class using one-vs-all approach
            for j, label in enumerate(unique_labels):
                # Current class as positive, others as negative
                TP = np.sum((y_pred == j) & (y_true_idx == j))
                FP = np.sum((y_pred == j) & (y_true_idx != j))
                FN = np.sum((y_pred != j) & (y_true_idx == j))
                
                precision = TP/(TP+FP) if (TP+FP) > 0 else 0
                recall = TP/(TP+FN) if (TP+FN) > 0 else 0
                
                metrics[str(label)] = {
                    'precision': precision,
                    'recall': recall
                }
            
            # Calculate macro average
            metrics['macro_avg'] = {
                'precision': np.mean([m['precision'] for m in metrics.values() if isinstance(m, dict)]),
                'recall': np.mean([m['recall'] for m in metrics.values() if isinstance(m, dict)])
            }
        
        return metrics 