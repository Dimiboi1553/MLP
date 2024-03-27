from HiddenLayer import *
from InputLayer import *
from OutputLayer import *
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MLP():
    def __init__(self, HiddenLayers, NeuronsPerLayer, NoOfOutputs, InputLayerNeurons ,learning_rate=0.01):
        #Define Lr
        self.Learning_rate = learning_rate

        #No of Outputs determines the type of MLP(Classification or regression)
        #Define the Layers
        self.Layers = []

        #Firstly, define Input Layer and add it to the self.Layers
        self.InputLayer = InputLayer()
        self.Layers.append(self.InputLayer)

        #Output Layer we will define this in _init_Layers
        self.OutputLayer = None
        
        #Outputs
        self.NoOfOutputs = NoOfOutputs

        #Hidden Layers
        self._init_Layers(HiddenLayers, NeuronsPerLayer, NoOfOutputs, InputLayerNeurons)
        
        #Loss
        self.Loss = 0

    def _init_Layers(self, HiddenLayers, NeuronsPerLayer, NoOfOutputs, InputLayerNeurons):
        # NeuronsPerLayer is a list showing the Neurons per layer e.g [32,64,128] Hidden Layer 1 will have 32, Layer 2 64 and Layer 3 128
        if len(NeuronsPerLayer) != HiddenLayers:
            raise ValueError("Neurons per layer and Hidden layers do not match. Number of items in list NeuronsPerLayer needs to equal HiddenLayers.")
        
        PrevLayerNeurons = InputLayerNeurons

        for Neurons in NeuronsPerLayer:

            #print(f"{Neurons}, {PrevLayerNeurons}")
            self.Layers.append(HiddenLayer(Neurons, PrevLayerNeurons))

            PrevLayerNeurons = Neurons

        # Ok, we added all hidden Layers. Let's add the OutputLayer.
        self.OutputLayer = OutputLayer(NoOfOutputs, NeuronsPerLayer[-1])
    
    def Forward(self, Input):
        #Give the InputLayer the input
        self.InputLayer.Forward(Input)

        #Do the forward propagation through the hidden layers

        for i in range(1, len(self.Layers)):
            self.Layers[i].Forward(self.Layers[i-1])

        #Finally, use the output layer to get the output layer
        Output = self.OutputLayer.Forward(self.Layers[-1])

        return Output

    def Learn(self, X, Y, Epochs, Verbose=0, Frequency=32):
        if len(X) != len(Y):
            raise ValueError("Training data is unequal x != y")
        
        OutputValues = []
        TargetValues = []
        Inputs = []

        for i in range(Epochs):
            self.Loss = 0

            #Step 1: Forward pass
            for j, x in enumerate(X):
                #Add Y to target values
                TargetValues.append(Y[j])  # Target value
                #Add x sublist to Inputs
                Inputs.append(x)
                #x is already defined as the sublist
                OutputValues.append(self.Forward(x))

                #Step 2: Backwards pass per Frequency outputs batches
                if len(OutputValues) % (Frequency) == 0:
                    #Calculate loss only if it's time to calculate
                    Total_Error = self.CalculateSlope(OutputValues, TargetValues)
                    self.Loss += Total_Error
                    self.Backpropagation(Total_Error, Inputs)

                    OutputValues.clear()
                    TargetValues.clear()
                    Inputs.clear()

            if Verbose != 0 and i % Verbose == 0:
                print(f"Total Loss: {self.Loss}, Epoch: {i}")

    def CalculateSlope(self, Outputs, Targets):
        # There are two cases 1: regression model use MSE
        if self.NoOfOutputs == 1:
            Total_Error = 0

            for x, i in enumerate(Outputs):
                y = Targets[x][0]  # Target Value
                # i is our observed value
                Total_Error += (y - i)**2

            Total_Error = Total_Error / len(Outputs)
            
            return Total_Error
        
    def Backpropagation(self, Gradient, Inputs):
        for Layers in reversed(self.Layers):
            Layers.Backpropagation(Gradient, self.Learning_rate, Inputs)

def load_data():
    # Load the California Housing dataset
    california_housing = fetch_california_housing()
    X, y = california_housing.data, california_housing.target

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the normalized dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

    return X_train, X_test, y_train, y_test

def test_mlp():
    # Load the California Housing dataset
    X_train, X_test, y_train, y_test = load_data()

    # Define the architecture of the MLP
    hidden_layers = 2
    neurons_per_layer = [10, 10]  # Adjust the number of neurons as needed
    num_outputs = 1  # Adjust the number of output neurons as needed

    # Create an instance of the MLP
    my_mlp = MLP(hidden_layers, neurons_per_layer, num_outputs, X_train.shape[1])

    # Train the MLP
    my_mlp.Learn(X_train, y_train.reshape(-1, 1), Epochs=100, Verbose=1, Frequency=1)

# Run the test case
test_mlp()