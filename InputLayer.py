class InputLayer():
    def __init__(self, Inputs):
        self.Inputs = Inputs
    
    def GetInputsForNextLayer(self):
        return self.Inputs