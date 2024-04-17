class InputLayer():
    def __init__(self):
        self.InputsForOtherLayers = []

    def Forward(self, Inputs):
        #Previous layer is not used to simplify the forward process
        self.InputsForOtherLayers = Inputs

    def GetInputsForNextLayer(self):
        return self.InputsForOtherLayers

    def Backpropagation(self, Gradient, learning_rate, Inputs):
        pass