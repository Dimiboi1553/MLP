from Neuron import *

class HiddenLayer():
    def __init__(self, NeuronCount, Out):
        self.Neurons = []
        self.InputsForNextLayer = []

        self.init_Neurons(NeuronCount, Out)

    def init_Neurons(self, NeuronCount, Out):
        for i in range(NeuronCount):
            #Init All neurons
            self.Neurons.append(Neuron(Out))
    
    def Forward(self, PreviousLayer):
        Inputs = PreviousLayer.GetInputsForNextLayer()
        
        for Neuron in self.Neurons:
            self.InputsForNextLayer(Neuron.Forward(Inputs))
    
    def GetOutput(self):
        return self.InputsForNextLayer 

