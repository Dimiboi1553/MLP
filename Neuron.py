import random

class Neuron():
    def __init__(self, Out):
        self.ConnectionWeights = []
        self.Bias = random.randint(-1,4)
        
        self.initWeights(Out)
        
        self.Output = 0

    def initWeights(self, Out):
        for i in range(Out):
            self.ConnectionWeights.append(random.uniform(-1,4))

    def Forward(self, Inputs):
        Output = 0

        for x,i in enumerate(Inputs):
            Output += (self.ConnectionWeights[x] * i)
        
        Output += self.Bias

        self.Output = self.ReLU(Output)

        return self.Output
    
    def Backward(self, Gradient, learning_rate, Inputs):
        weight_gradients = [Gradient * i for i in Inputs]
        
        #Update weights
        for x in range(len(self.ConnectionWeights)):
            self.ConnectionWeights[x] += learning_rate * weight_gradients[x]

        #Update bias
        self.Bias += learning_rate * Gradient

    def ReLU(self, z):
        return max(0,z)
