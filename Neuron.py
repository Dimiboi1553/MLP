import random
import math

class Neuron():
    def __init__(self, In):
        self.ConnectionWeights = []
        self.Bias = random.uniform(-0.5, 0.5)
        
        self.initWeights(In)
        

        self.Output = 0

    def initWeights(self, In):
        for i in range(In):
            self.ConnectionWeights.append(random.uniform(-1, 1))

    def Forward(self, Inputs):
        InputForNeuron = 0

        for x,i in enumerate(Inputs):
            InputForNeuron += self.ConnectionWeights[x] * i

        InputForNeuron += self.Bias

        self.Output = self.Sigmoid(InputForNeuron)
        
        return self.Output
    
    def Backward(self, Gradient, learning_rate, Inputs):
        # Calculate the gradient of the sigmoid function
        sigmoid_gradient = self.Output * (1 - self.Output)
        
        # Update bias using the gradient
        self.Bias -= learning_rate * Gradient * sigmoid_gradient
        
        #Update weights using the gradient
        for i, weight in enumerate(self.ConnectionWeights):
            weight_gradient = Gradient * Inputs[i] * sigmoid_gradient
            self.ConnectionWeights[i] -= learning_rate * weight_gradient
    
    def Sigmoid(self, x):
        try:
            Total =  1 / (1 + math.exp(-x))
        except:
            Total = 1
        
        return Total
