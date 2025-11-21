import random

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.bias = random.uniform(-1,1)
        self.learning_rate = learning_rate
    
    def activation_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x, y):
        # w1*x + w2*y + b 
        linear_output = (x*self.weights[0]) + (y*self.weights[1]) + self.bias
        return self.activation_function(linear_output)
    
    def train(self, x, y, target):
        guess = self.predict(x, y)
        error = target - guess
        if error != 0:
            self.weights[0] += error * self.learning_rate*x
            self.weights[1] += error * self.learning_rate*y
            self.bias += error * self.learning_rate
