import torch
from torch.nn import Module, Parameter, ModuleList, Linear, Sigmoid, ReLU

import numpy as np


BYTE_SIZE = 5
SCALE_FACTOR = float(2 ** BYTE_SIZE - 1)

class NetworkFromWeights(Module):

    def __init__(self, genes, v_max=10.0):
        super(NetworkFromWeights, self).__init__()
        # decode the genes to weights and biases
        decode_genes = self.decode_genes(genes)
        weights, biases = self.get_weights_biases(decode_genes)
        self.weights = weights
        self.biases = biases
        self.layers = []
        self.activations = []
        self.num_layers = len(weights)
        for i in range(self.num_layers):
            self.layers.append(Linear(weights[i].shape[0], weights[i].shape[1]))
            self.activations.append(ReLU()) #Sigmoid()
            self.layers[i].weight.data = Parameter(torch.tensor(weights[i], dtype=torch.float32).T)
            self.layers[i].bias.data = Parameter(torch.tensor(biases[i], dtype=torch.float32).T)
        self.layers = ModuleList(self.layers)
        self.activations = ModuleList(self.activations)
        self.v_max = v_max


    def forward(self, x, delayed_outputs=None):
        if delayed_outputs is None:
            delayed_outputs = torch.zeros((1,2), dtype=x.dtype, device=x.device)
        # concat delayed inputs to the raw input 
        x = torch.cat((x, delayed_outputs), dim=1)
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.activations[i](x)
        # clamp the output to the range of [-5, 5]
        x = torch.clamp(x, min=0, max=1) * self.v_max - 5
        return x 

    def get_weights_biases(self, raw_genes: np.ndarray):
        """ Decode weights and biases from raw genes
        """
        weights = []
        biases = []
        # rehspae the genes to the shape of the weights and biases for different layers
        weights.append(raw_genes[:28].reshape(14, 2))
        biases.append(raw_genes[28:30].reshape(2))
        return weights, biases

    def decode_genes(self, binary_gene: str):
        """ Decode the binary gene raw genetic code"""
        # 30 = 14 * 2 + 2
        if len(binary_gene) != 30 * BYTE_SIZE:
            raise ValueError("Input string must be exactly 240 bits long.")
        # Split the binary string into 30 parts, each 8 bits
        bytes_list = [binary_gene[i:i + BYTE_SIZE] for i in range(0, 30 * BYTE_SIZE, BYTE_SIZE)]
        decimal_values = []
        for byte in bytes_list:
            # Get the sign bit (MSB)
            sign_bit = int(byte[0])  # Convert the MSB to an integer
            # Convert the remaining 7 bits to a decimal integer
            # Convert binary to decimal
            # Scale the unsigned value to be between 0 and 1
            decimal_value = int(byte[1:], 2) / SCALE_FACTOR
            # Apply the sign bit
            if sign_bit == 1:
                # Make the value negative if the sign bit is 1
                decimal_value = -decimal_value

            decimal_values.append(decimal_value)

        return np.array(decimal_values)

class NetworkFromWeights_2(Module):

    def __init__(self, genes, v_max=10.0):
        super(NetworkFromWeights_2, self).__init__()
        # decode the genes to weights and biases
        decode_genes = self.decode_genes(genes)
        weights, biases = self.get_weights_biases(decode_genes)

        self.weights = weights # tuples
        self.biases = biases # tuples
        self.layers = []
        self.activations = []
        self.num_layers = len(weights)
        for i in range(self.num_layers):
            self.layers.append(Linear(weights[i].shape[0], weights[i].shape[1]))
            self.activations.append(ReLU()) #Sigmoid()
            self.layers[i].weight.data = Parameter(torch.tensor(weights[i], dtype=torch.float32).T)
            self.layers[i].bias.data = Parameter(torch.tensor(biases[i], dtype=torch.float32).T)
        self.layers = ModuleList(self.layers)
        self.activations = ModuleList(self.activations)
        self.v_max = v_max

    def forward(self, x, delayed_outputs=None):
        if delayed_outputs is None:
            delayed_outputs = torch.zeros((1,2), dtype=x.dtype, device=x.device)
        # concat delayed inputs to the raw input 
        x = torch.cat((x, delayed_outputs), dim=1)
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.activations[i](x)
        # clamp the output to the range of [-5, 5]
        x = torch.clamp(x, min=0, max=1) * self.v_max - 5
        return x 

    def get_weights_biases(self, raw_genes: np.ndarray):
        """ Decode weights and biases from raw genes
        """
        # achitecture: 14 -> 16 -> 2
        arch = [10, 4, 2]
        weights = []
        biases = []
        # rehspae the genes to the shape of the weights and biases for different layers
        w_index_1 = arch[0]*arch[1]
        w_index_2 = w_index_1 + arch[1] * arch[2]
        b_index_1 = w_index_2 + arch[1]
        b_index_2 = b_index_1 + arch[2] 

        weights.append(raw_genes[ : w_index_1].reshape(arch[0], arch[1]))
        weights.append(raw_genes[w_index_1 : w_index_2].reshape(arch[1], arch[2]))
        
        biases.append(raw_genes[w_index_2 : b_index_1].reshape(arch[1]))
        
        biases.append(raw_genes[b_index_1 : b_index_2].reshape(arch[2]))

        return weights, biases

    def decode_genes(self, binary_gene: str):
        """ Decode the binary gene raw(decimal) genetic code"""
        # total_params = w1+b1 + w2+b2, 14*16 + 16 + 16*2 + 2 = 274
        # 6*4 + 4 + 4*2 + 2 = 30
        if len(binary_gene) != 54 * BYTE_SIZE:
            raise ValueError("Input string must be exactly 274 bits long.")
        # Split the binary string into 30 parts, each 8 bits
        bytes_list = [binary_gene[i:i + BYTE_SIZE] for i in range(0, 54 * BYTE_SIZE, BYTE_SIZE)]
        decimal_values = []
        for byte in bytes_list:
            # Get the sign bit (MSB)
            sign_bit = int(byte[0])  # Convert the MSB to an integer
            # Convert the remaining 7 bits to a decimal integer
            # Convert binary to decimal
            # Scale the unsigned value to be between 0 and 1
            decimal_value = int(byte[1:], 2) / SCALE_FACTOR
            # Apply the sign bit
            if sign_bit == 1:
                # Make the value negative if the sign bit is 1
                decimal_value = -decimal_value

            decimal_values.append(decimal_value)

        return np.array(decimal_values)


if __name__ == "__main__":
    # test the network with random weights
    # weights = []
    # # shape  2 layers, -> (8,4), (4,2) 
    # weights_layer_1 = np.random.rand(4, 8)
    # weights_layer_2 = np.random.rand(2, 4)
    # biases_layer_1 = np.random.rand(4)
    # biases_layer_2 = np.random.rand(2)
    # weights = [weights_layer_1, weights_layer_2]
    # biases = [biases_layer_1, biases_layer_2]

    # # x = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
    # x = torch.randn(1, 8)
    # print("Shapes: ", x.shape, weights_layer_1.shape, weights_layer_2.shape, biases_layer_1.shape, biases_layer_2.shape)
    # model = NetworkFromWeights(weights, biases)
    # print("model", model)
    # op = model.forward(x)

    # # print(f"weights: {weights}")
    # # print(f"biases: {biases}")
    # print(f"output: {op}")

    # test with raw genes
    # genes = np.random.rand(30)
    # input = torch.randn(1,14)
    # print("model", model)
    # print(f"genes: {genes.shape}, weights: {weights.shape}, biases: {biases.shape},input: {input.shape}") 
    # op = model.forward(input)
    # print(f"genes: {genes.shape}, weights: {model.weights[0].shape}, biases: {model.biases[0].shape}, input: {input.shape}, output: {op.shape}")

    # # test decode genes
    # genes = '01111111' * 30  # Example with a 240-bit binary string filled with zeros
    # model = NetworkFromWeights(genes)
    # input = torch.ones(1, 14)
    # output = model.forward(input)
    # print(output[0], output[1])
    # print(output.shape, type(output), input.shape, type(input))

    # # test delayed output 
    # genes = '01111111' * 30  # Example with a 240-bit binary string filled with zeros
    # model = NetworkFromWeights(genes)
    # delayed_outputs = None
    # for i in range(1):
    #     input = torch.ones(1, 12)
    #     output = model.forward(input, delayed_outputs)
    #     delayed_outputs = output
    #     print(output, output.shape, type(output), input.shape, type(input))
    
    # test 2 layer networks
    gene = np.random.choice([0,1], size= 274* BYTE_SIZE).astype(str)
    gene = ''.join(gene)
    model = NetworkFromWeights_2(gene)
    input = torch.randn(1,12)
    op = model.forward(input)
    print(len(gene), "\n", model)
    print(model.weights[0].shape, model.biases[0].shape, model.weights[1].shape, model.biases[1].shape, op)