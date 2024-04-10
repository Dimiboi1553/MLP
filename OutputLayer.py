from Neuron import *
import math

class OutputLayer():
    def __init__(self, OutputNeurons, PreviousLayerNeuron):
        self.Neurons = []
        self.Outputs = []
        
        self.init_Neurons(OutputNeurons, PreviousLayerNeuron)

        #Cross Entropy list isnt the actual cross entropy values its just a place to store them to pass them into the MLP later for backpropagation
        self.CrossEntropy = []

    def init_Neurons(self, NeuronCount, In):
        for i in range(NeuronCount):
            #Init All neurons
            self.Neurons.append(Neuron(In))

    def Forward(self, PreviousLayer):
        Inputs = PreviousLayer.GetInputsForNextLayer()

        self.Outputs.clear()

        for Neuron in self.Neurons:
            self.Outputs.append(Neuron.Forward(Inputs))

        if len(self.Neurons) >= 2:
            #If more than 1 Neuron than Classification problem
            return self.Softmax()
        else:
            #One neuron is a regression problem so output neuron
            return self.Outputs[0]
    
    def Backpropagation(self, Gradient, learning_rate, Inputs):
        #Update weights according to Gradient
        for neurons in self.Neurons:
            neurons.Backward(Gradient, learning_rate, Inputs)

    def Softmax(self):
        #For softmax, first, we calculate the denominator
        max_value = max(self.Outputs)   
        exp_values = [math.exp(i - max_value) for i in self.Outputs]

        denominator = sum(exp_values)
        #Initialize variables to track the greatest softmax value and its index
        greatest_softmax = -1
        greatest_index = -1     

        for x, exp_i in enumerate(exp_values):
            #Calculate the softmax of the neuron
            softmax_i = exp_i / denominator
            
            self.CrossEntropyFunc(softmax_i, x)

            #Compare with the current greatest softmax value
            if softmax_i > greatest_softmax:
                greatest_softmax = softmax_i
                greatest_index = x

        #Return the greatest softmax value, its index, and the original output value
        return [greatest_softmax, greatest_index, self.Outputs[greatest_index]]

    def CrossEntropyFunc(self, p, i):
        epsilon = 1e-10

        CrossEntropyVal = math.log(p + epsilon, math.e)
        self.CrossEntropy.append([CrossEntropyVal, i])#I is the index :)

    def GetCrossEntropy(self):
        return self.CrossEntropy


        

