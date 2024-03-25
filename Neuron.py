import random
import math

class Neuron():
    def __init__(self, In):
        self.ConnectionWeights = []
        self.Bias = random.uniform(-0.5, 0.5)
        
        self.initWeights(In)
        
        self.Input = 0

    def initWeights(self, In):
        for i in range(In):
            self.ConnectionWeights.append(random.uniform(-1, 1))

    def Forward(self, Inputs):
        Input = 0

        for x,i in enumerate(Inputs):
            Input += (self.ConnectionWeights[x] * i)
        
        Input += self.Bias

        self.Input = self.Sigmoid(Input)

        return self.Input
    
    def Backward(self, Gradient, learning_rate, Inputs):
        print(Inputs)
        #TODO RIGHT NOW INPUTS IS A 2D LIST OF THE INPUTS ITS EXPECTING A 1D
        weight_gradients = [Gradient * i for i in Inputs]
        
        #Update weights
        for x in range(len(self.ConnectionWeights)):
            self.ConnectionWeights[x] += learning_rate * weight_gradients[x]

        #Update bias
        self.Bias -= learning_rate * Gradient

    def Sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
