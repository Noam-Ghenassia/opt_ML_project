from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from einops import rearrange
import math
from collections import defaultdict
import copy
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#fix the seed for reproductibility 
torch.manual_seed(0)

class Dataset_2D(ABC):
    """Abstract class for 2-classes datasets in 2 dimensions.

    n_points (int) : the number of samples in each class.
    """

    def __init__(self, n_points=150):
        self.n_points = n_points
        self.dataset = self._create_dataset()
        super().__init__()
    
    @abstractmethod
    def _create_dataset(self):
        pass

    def plot(self, figure):
        """This methods plots the dataset on the figure passed as argument.

        Args:
            figure (matplotlib.axes.Axes): the pyplot figure on which the dataset is plotted.
        """
        dataset = self.get_dataset()
        data = dataset[:, 1:].float()
        labels = dataset[:, :1].long()
        labels = rearrange(labels, 'h w -> (h w)')
        ind_0 = torch.nonzero(labels==0)
        ind_1 = torch.nonzero(labels==1)

        figure.plot(data[ind_0, 0], data[ind_0, 1], 'b.',
                    data[ind_1, 0], data[ind_1, 1], 'r.')

    def get_dataset(self):
        """Accessor method.

        Returns:
            (torch.tensor): a tensor with rows the datapoints, and columns the features.
            the first column contains the labels.
        """
        return self.dataset.to(device)

class figure_8(Dataset_2D):
    """ This dataset consists of 2 classes : one of them consists of 2 circles with centers
    on the x axis, at distance 5 from the origin. the second one is the interior of the circles.
    """
    def __init__(self, n_points, var):
        self.var = var
        super().__init__(int(math.ceil(n_points/4)))
    
    def _create_dataset(self):
        
        # class 0
        r11 = torch.abs(torch.normal(0, self.var**0.5, size = (1,self.n_points)).squeeze())
        theta11 = 2*math.pi*torch.rand(self.n_points)
        x11 = r11*torch.cos(theta11)+5
        y11 = r11*torch.sin(theta11)

        r12 = torch.abs(torch.normal(0, self.var**0.5, size = (1,self.n_points)).squeeze())
        theta12 = 2*math.pi*torch.rand(self.n_points)
        x12 = r12*torch.cos(theta12)-5
        y12 = r12*torch.sin(theta12)

        C11 = torch.unsqueeze(torch.transpose(torch.cat([x11, x12]), dim0=0, dim1=-1), dim=1)
        C12 = torch.unsqueeze(torch.transpose(torch.cat([y11, y12]), dim0=0, dim1=-1), dim=1)
        C1 = torch.cat([C11, C12], axis=-1)
        C1 = torch.cat([torch.zeros((2*self.n_points, 1)), C1], 1)

        # class 1
        r21 = 4+torch.abs(torch.normal(0, self.var**0.5, size = (1,self.n_points)).squeeze())   # abs ?
        theta21 = 2*math.pi*torch.rand(self.n_points)
        x21 = r21*torch.cos(theta21)+5
        y21 = r21*torch.sin(theta21)

        r22 = 4+torch.abs(torch.normal(0, self.var**0.5, size = (1,self.n_points)).squeeze())
        theta22 = 2*math.pi*torch.rand(self.n_points)
        x22 = r22*torch.cos(theta22)-5
        y22 = r22*torch.sin(theta22)

        C21 = torch.unsqueeze(torch.transpose(torch.cat([x21, x22]), dim0=0, dim1=-1), dim=1)
        C22 = torch.unsqueeze(torch.transpose(torch.cat([y21, y22]), dim0=0, dim1=-1), dim=1)
        C2 = torch.cat([C21, C22], axis=-1)
        C2 = torch.cat([torch.ones((2*self.n_points, 1)), C2], 1)

        # full dataset
        dataset = torch.cat([C1, C2], 0)
        dataset = dataset[torch.randperm(dataset.size()[0])]
        self.n_points = dataset.shape[0]
        return dataset

class net(nn.Module):
    """A dense neural network with ReLU activation functions. The structure argument is a tuple
    whose n-th entry is the number of neurons in the n-th hidden layer.
    """
    def __init__(self, structure=(10, 10, 10), with_bias=True):
        #super(net, self).__init__()
        super().__init__()
        self.structure = structure

        self.layer_list = torch.nn.ModuleList()
        # add a class attribute containing only the hidden layers, that is later
        # used for teleportations.
        self.hidden_layers_list = torch.nn.ModuleList()

        self.layer_list.append(nn.Sequential(nn.Linear(2, structure[0], bias=False)))

        for ii in range(len(self.structure)):
            self.layer_list.append(
                self.hidden_layer(self.structure[ii] , self.structure[ii], with_bias=with_bias, use_batch_norm=False)
            )
          
        self.layer_list.append(nn.Sequential(nn.Linear(structure[-1], 2, bias=False)))


        for layer in range(len(self.layer_list)-1) :
            self.hidden_layers_list.append(self.layer_list[layer])


    def hidden_layer(self,input, output, with_bias=True, use_batch_norm=False):
        linear = nn.Linear(input, output, bias=with_bias)
        relu = nn.ReLU()
        bn = nn.BatchNorm1d(output)
        if use_batch_norm:
            return(nn.Sequential(linear, relu, bn))
        else:
            return(nn.Sequential(linear, relu))
    
    def reset(self):
        for layer in self.children():
            layer.reset_parameters()


    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x
    
    def make_batches(self, dataset, batch_size=32):
        """This method returns a list containing batches sampled randomly from the
        dataset, with the indicated batch size (except for the last batch, which
        might be smaller"""

        n_points = dataset.shape[0]
        remainder = n_points % batch_size
        num_full_batches = (n_points - remainder)/batch_size
        perm_dataset = dataset[torch.randperm(dataset.size()[0])]
        batches_list = []
        for batch in range(int(num_full_batches)):
            batches_list.append(perm_dataset[batch:batch+batch_size, :])
        batches_list.append(perm_dataset[-remainder-1:-1, :])
        return batches_list

    def ADAM_train(self, dataset, n_epochs):
        """This method allows to train the neural network with the Adam optimizer."""
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters())

        for epoch in range(n_epochs):

            batches = self.make_batches(dataset)
            for batch in batches :
                data = batch[:, 1:].float()
                labels = batch[:, :1].long()
                labels = rearrange(labels, 'h w -> (h w)')

                optimizer.zero_grad()
                out = self.forward(data)
                loss = criterion(out, labels)

                loss.backward()
                optimizer.step()
    
    def ASAM_train(self, dataset, n_epochs):
        """This method allows to train the neural network with the Adam optimizer and Asam minimizer."""

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters())
        minimizer = ASAM(optimizer, self, 0.5, 0.01)

        # Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        for epoch in range(n_epochs):

            self.train()

            batches = self.make_batches(dataset)
            for batch in batches :
                data = batch[:, 1:].float()
                labels = batch[:, :1].long()
                labels = rearrange(labels, 'h w -> (h w)')
                
                #Ascent Step
                out = self.forward(data)
                loss = criterion(out, labels)
                loss.mean().backward()
                minimizer.ascent_step()

                # Descent Step
                criterion(self.forward(data), labels).mean().backward()
                minimizer.descent_step()

            scheduler.step()

    def SAM_train(self, dataset, n_epochs):
        """This method allows to train the neural network with the Adam optimizer and Sam minimizer."""
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters())
        minimizer = SAM(optimizer, self, 0.5, 0.01)

        # Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        self.train()

        for epoch in range(n_epochs):

            batches = self.make_batches(dataset)
            for batch in batches :
                data = batch[:, 1:].float()
                labels = batch[:, :1].long()
                labels = rearrange(labels, 'h w -> (h w)')
                
                #Ascent Step
                out = self.forward(data)
                loss = criterion(out, labels)
                loss.mean().backward()

                minimizer.ascent_step()

                # Descent Step
                criterion(self.forward(data), labels).mean().backward()
                minimizer.descent_step()

            scheduler.step()



    
    def test(self, dataset):
        """"This method is to calculate the test error"""
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()
            data = dataset[:, 1:].float()
            labels = dataset[:, :1].long()
            labels = rearrange(labels, 'h w -> (h w)')

            out = self.forward(data)
            loss = criterion(out, labels)

            return float(loss)

    
    def plot_decision_boundary(self, ax, x_min=-15., x_max=15., y_min=-10., y_max=10.):
        """This function allows to plot the network's decision boundary on a given figure.
        Args:
            ax (matplotlib.axes.Axes): the pyplot figure on which the dataset is plotted.
            x_min (float, optional): lower bound of the x axis of the plot. Defaults to -10..
            x_max (float, optional): high bound of the x axis of the plot. Defaults to 10..
            y_min (float, optional): lower bound of the y axis of the plot. Defaults to -10..
            y_max (float, optional): high bound of the y axis of the plot. Defaults to 10..
        """
        x = np.linspace(x_min, x_max, 220)
        y = np.linspace(y_min, y_max, 220)
        grid_x = np.meshgrid(x, y)[0].reshape(-1, 1)
        grid_y = np.meshgrid(x, y)[1].reshape(-1, 1)
        grid = np.concatenate([grid_x, grid_y], axis=1)

        grid = torch.tensor(np.expand_dims(grid, axis=1)).float().to(device)
        out = self.forward(grid)
        out = torch.squeeze(out, axis=1)
        out = nn.Softmax(dim=1)(out)
        out = out[: , 1].reshape(len(x), -1).detach().cpu().numpy()
        
        ax.contourf(np.meshgrid(x, y)[0], np.meshgrid(x, y)[1], out)


###############################################################################
###############################################################################

class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    def loss(self, data, labels, criterion):
        model_clone = copy.deepcopy(self.model)
        out = self.model.forward(data)
        loss = criterion(out, torch.squeeze(labels, dim=1).long())
        #loss.mean().backward()
        loss.backward()
        ascented_params = self.ascent_step(return_result_of_ascent=True)
        self.model.zero_grad()
        for name, params in model_clone.named_parameters():
            params.data.copy_(ascented_params[name])
        preds = model_clone(data)
        return criterion(preds, torch.squeeze(labels, dim=1).long())

    @torch.no_grad()
    #def ascent_step(self):
    def ascent_step(self, return_result_of_ascent=False):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w      # comment this one ?
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16

        if return_result_of_ascent:
            # retain parameters before ascent step
            old_params = {}
            for name, params in self.model.named_parameters():
                old_params[name] = params.clone()

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

        if return_result_of_ascent:
            # retain parameters after ascent step
            ascented_params = {}
            for name, params in self.model.named_parameters():
                ascented_params[name] = params.clone()
            
            # reload parameters from before ascent step in the model
            for name, params in self.model.named_parameters():
                params.data.copy_(old_params[name])
            
            return ascented_params

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):

    def loss(self, data, labels, criterion):
        model_clone = copy.deepcopy(self.model)
        out = self.model.forward(data)
        loss = criterion(out, torch.squeeze(labels, dim=1).long())
        loss.backward()
        ascented_params = self.ascent_step(return_result_of_ascent=True)
        self.model.zero_grad()
        for name, params in model_clone.named_parameters():
            params.data.copy_(ascented_params[name])
        preds = model_clone(data)
        return criterion(preds, torch.squeeze(labels, dim=1).long())

    @torch.no_grad()
    def ascent_step(self, return_result_of_ascent=False):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16

        if return_result_of_ascent:
            # retain parameters before ascent step
            old_params = {}
            for name, params in self.model.named_parameters():
                old_params[name] = params.clone()

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

        if return_result_of_ascent:
            # retain parameters after ascent step
            ascented_params = {}
            for name, params in self.model.named_parameters():
                ascented_params[name] = params.clone()
            
            # reload parameters from before ascent step in the model
            for name, params in self.model.named_parameters():
                params.data.copy_(old_params[name])
            
            return ascented_params


def test_performance_ADAM_ASAM_SAM(number_points, var, epochs):
    """
    Train three neural networks (with ADAM,ASAM and SAM) on a train dataset with variance (var).
    Test the trained neural networks on different datasets with varing variances to compare ADAM, ASAM and SAM. 

    INPUT : 
    number_points : numbers_point_per_datasets
    var : variance of one dataset
    epochs : number of training epochs
    """

    #fix the seed for reproductibility 
    torch.manual_seed(0)

    variances = np.arange(1,80,0.5) #List of variances
    errors_ADAM = [] #List to store errors during testing
    errors_ASAM = [] #List to store errors during testing
    errors_SAM = []  #List to store errors during testing

    #creation of the dataset
    train_dataset = figure_8(number_points, var)

    #creation of test datasets
    test_datasets = []
    for variance in variances :
        test_datasets.append(figure_8(number_points, variance))

    #creation of the nets
    net_ADAM = net()
    net_ASAM = net()
    net_SAM = net()

    #training
    print("training ADAM...")
    net_ADAM.ADAM_train(train_dataset.get_dataset(), epochs)
    print("training ASAM...")
    net_ASAM.ASAM_train(train_dataset.get_dataset(), epochs)
    print("training SAM...")
    net_SAM.SAM_train(train_dataset.get_dataset(), epochs)

    #testing
    for test_dataset in test_datasets:
        errors_ADAM.append(net_ADAM.test(test_dataset.get_dataset()))
        errors_ASAM.append(net_ASAM.test(test_dataset.get_dataset()))
        errors_SAM.append(net_SAM.test(test_dataset.get_dataset()))

    #ploting
    fig = plt.figure(figsize=(10,5))
    plt.plot(variances, errors_ADAM, label='ADAM')
    plt.plot(variances, errors_ASAM, label='ASAM')
    plt.plot(variances, errors_SAM, label='SAM')
    plt.xlabel('Variances')
    plt.ylabel('Test Error')
    plt.title(f"Generalization for ADAM, ASAM and SAM trained one dataset with variance = {var} and for {epochs} epochs for the training")
    plt.legend()

    plt.show()
    

def plot_dataset(variance, number_points = 2000):
    """
    Function to plot example of datasets
    """
    fig, ax = plt.subplots()
    torch.manual_seed(0)
    figure_8(number_points, variance).plot(ax)


