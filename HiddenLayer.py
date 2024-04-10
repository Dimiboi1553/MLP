from Neuron import *

class HiddenLayer():
    def __init__(self, NeuronCount, PreviousLayerNeuronCount, Name):
        self.Neurons = []
        self.InputsForNextLayer = []
        self.Name = Name
        self.init_Neurons(NeuronCount, PreviousLayerNeuronCount)

    def init_Neurons(self, NeuronCount, PreviousLayerNeuronCount):
        for i in range(NeuronCount):
            #Init All neurons
            self.Neurons.append(Neuron(PreviousLayerNeuronCount))
    
    def Forward(self, PreviousLayer):
        self.InputsForNextLayer.clear()

        Inputs = PreviousLayer.GetInputsForNextLayer()

        for Neuron in self.Neurons:
            self.InputsForNextLayer.append(Neuron.Forward(Inputs))

    def Backpropagation(self, Output, Target, Learning_rate, Input):
        #Update weights according to Gradient
        for neurons in self.Neurons:
            neurons.Backward(Output, Target, Learning_rate, Input)
    
    def GetInputsForNextLayer(self):
        return self.InputsForNextLayer 

