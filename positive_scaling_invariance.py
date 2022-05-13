import torch
import torch.nn as nn


def perform_neural_telep(net, alpha) :
    """this function sends a given network to another, isomorphic network, following
    the teleportation vector alpha.
    
    net : a nn.Module equipped with a layer_list
    alpha : a list of tensors"""

    # For the neural network function to remain the same, the teleportation
    # must not change the weights between the input and the first hidden
    # layer, nor those between the last hidden layer and the output.
    alpha.insert(0, torch.ones(net.layer_list[0][0].in_features))
    alpha.append(torch.ones(net.layer_list[-1][0].out_features))

    alpha = [l for l in alpha]

    for layer in range(len(net.layer_list)) :
        if net.layer_list[layer][0].bias is not None:
                
            # multiply the bias by the alpha value of the next layer :
            net.layer_list[layer][0].bias = \
                nn.parameter.Parameter(net.layer_list[layer][0].bias * alpha[layer + 1])

        # layer by layer positive scaling invariance on the hidden layers :
        # the input weights of a layer are divided by the alpha values, of
        # the corresponding layer, and the output weights are multiplied by
        # the alpha values of the next layer.
        inverse_input_alpha_matrix = torch.inverse(torch.diag(alpha[layer]))
        output_alpha_matrix = torch.diag(alpha[layer+1])
        with torch.no_grad():
            net.layer_list[layer][0].weight.copy_(torch.einsum('ik, kl, lj -> ij',\
                [output_alpha_matrix, net.layer_list[layer][0].weight, inverse_input_alpha_matrix]))
        
        # delete the ones added at both ends of the alpha list :
    del alpha[0]
    del alpha[-1]
    
def random_telep(net, low_bound=0.5, high_bound=1.5) :
    """this function performs a random (uniform) teleportation, with entries of the
    teleportation vector comprised between low_bound and high_bound.
    
    net : a nn.Module equipped with a layer_list and a hidden_layers_list
    low_bound (float): the lower bound of the uniform distribution.
    high_bound (float): the higher bound of the uniform distribution."""
    
    alpha = []
    for layer in net.hidden_layers_list :
        # generate a random vector with which the same dimension as the layer.
        # Each neuron is has its input weigts multiplied, and its output weights
        # divided by the corresponding value in the vector.
        alpha.append((high_bound - low_bound) * torch.rand(layer[0].out_features) + low_bound)

    perform_neural_telep(net, alpha)
