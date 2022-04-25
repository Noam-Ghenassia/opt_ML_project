from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from einops import rearrange
import math

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

        figure.plot(data[ind_0, 0], data[ind_0, 1], 'bo',
                    data[ind_1, 0], data[ind_1, 1], 'ro')

    def get_dataset(self):
        """Accessor method.

        Returns:
            (torch.tensor): a tensor with rows the datapoints, and columns the features.
            the first column contains the labels.
        """
        return self.dataset

class figure_8(Dataset_2D):
    """ This dataset consists of 2 classes : one of them consists of 2 circles with centers
    on the x axis, at distance 5 from the origin. the second one is the interior of the circles.
    """
    def __init__(self, n_points):
        super().__init__(int(math.ceil(n_points/4)))
    
    def _create_dataset(self):
        
        # class 0
        r11 = torch.abs(1.6*torch.randn(self.n_points))
        theta11 = 2*math.pi*torch.rand(self.n_points)
        x11 = r11*torch.cos(theta11)+5
        y11 = r11*torch.sin(theta11)

        r12 = torch.abs(1.6*torch.randn(self.n_points))
        theta12 = 2*math.pi*torch.rand(self.n_points)
        x12 = r12*torch.cos(theta12)-5
        y12 = r12*torch.sin(theta12)

        C11 = torch.unsqueeze(torch.transpose(torch.cat([x11, x12]), dim0=0, dim1=-1), dim=1)
        C12 = torch.unsqueeze(torch.transpose(torch.cat([y11, y12]), dim0=0, dim1=-1), dim=1)
        C1 = torch.cat([C11, C12], axis=-1)
        C1 = torch.cat([torch.zeros((2*self.n_points, 1)), C1], 1)

        # class 1
        r21 = 4+torch.abs(1.6*torch.randn(self.n_points))
        theta21 = 2*math.pi*torch.rand(self.n_points)
        x21 = r21*torch.cos(theta21)+5
        y21 = r21*torch.sin(theta21)

        r22 = 4+torch.abs(1.6*torch.randn(self.n_points))
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
    def __init__(self, structure=(10, 10, 10)):
        super(net, self).__init__()
        self.structure = structure

        self.layer_list = torch.nn.ModuleList()
        print("structure [0] : ", structure[0], type(structure[0]))
        self.layer_list.append(nn.Sequential(nn.Linear(2, structure[0], bias=False)))

        for ii in range(len(self.structure)):
            self.layer_list.append(
                self.hidden_layer(self.structure[ii] , self.structure[ii], use_batch_norm=False)
            )
          
        self.layer_list.append(nn.Sequential(nn.Linear(structure[-1], 2, bias=False)))


    def hidden_layer(self,input, output, use_batch_norm=False):
        linear = nn.Linear(input, output, bias=True)
        relu = nn.ReLU()
        bn = nn.BatchNorm1d(output)
        if use_batch_norm:
            return(nn.Sequential(linear, relu, bn))
        else:
            return(nn.Sequential(linear, relu))


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

    def train(self, dataset, n_epochs):
        """This method allows to train the neural network with the Adam optimizer."""
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters())

        for epoch in range(n_epochs):

            if epoch % 30 == 0:
                print("epoch : ", epoch)

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

        grid = torch.tensor(np.expand_dims(grid, axis=1)).float()
        out = self.forward(grid)
        out = torch.squeeze(out, axis=1)
        out = nn.Softmax(dim=1)(out)
        out = out[: , 1].reshape(len(x), -1).detach().numpy()
        
        ax.contourf(np.meshgrid(x, y)[0], np.meshgrid(x, y)[1], out)