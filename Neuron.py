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
        sum_weight_gradients = [0 for _ in self.ConnectionWeights]
    
        #Iterate over each instance in the batch
        for input_vector in Inputs:
            #Calculate weight gradients for the current input and add to the sum
            for i, input_val in enumerate(input_vector):
                sum_weight_gradients[i] += Gradient * input_val
        
        #Calculate the average gradient for each weight
        avg_weight_gradients = [sum_grad / len(Inputs) for sum_grad in sum_weight_gradients]
        
        # Update weights with the average gradient
        for i in range(len(self.ConnectionWeights)):
            self.ConnectionWeights[i] += learning_rate * avg_weight_gradients[i]

        #Update bias
        self.Bias -= learning_rate * Gradient

    def Sigmoid(self, x):
        try:
            Total =  1 / (1 + math.exp(-x))
        except:
            Total = 1
        
        return Total
