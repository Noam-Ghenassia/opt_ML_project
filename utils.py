from abc import ABC, abstractmethod
import torch
from torch import nn
from einops import rearrange
import math

class Dataset_2D(ABC):
    """Abstract class for 2-classes datasets in 2 dimensions.

    n_points (int) : the number of samples in each class.
    """

    def __init__(self, n_points=150):
        self.n_points = n_points
        self.data, self.labels = self._create_dataset()
        super().__init__()
    
    @abstractmethod
    def _create_dataset(self):
        pass

    def plot(self, figure):
        """This methods plots the dataset on the figure passed as argument.

        Args:
            figure (matplotlib.axes.Axes): the pyplot figure on which the dataset is plotted.
        """
        ind_0 = torch.nonzero(self.labels==0)
        ind_1 = torch.nonzero(self.labels==1)

        figure.plot(self.data[ind_0, 0], self.data[ind_0, 1], 'bo',
                    self.data[ind_1, 0], self.data[ind_1, 1], 'ro')

    def get_dataset(self):
        """Accessor method.

        Returns:
            (torch.tensor, torch.tensor): the data and labels of the dataset.
        """
        return self.data, self.labels


class figure_8(Dataset_2D):
    """ This dataset consists of 2 classes : one of them consists of 2 circles with centers
    on the x axis, at distance 5 from the origin. the second one is the interior of the circles.
    """
    def __init__(self, n_points):
        super().__init__(int(math.ceil(n_points/2)))
    
    def _create_dataset(self):
        
        #inner right (11)
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

        dataset = torch.cat([C1, C2], 0)
        dataset = dataset[torch.randperm(dataset.size()[0])]
        data = dataset[:, 1:].float()
        labels = dataset[:, :1].long()
        labels = rearrange(labels, 'h w -> (h w)')

        return data, labels

class net(nn.Module):
    """A dense neural network with ReLU activation functions. The structure argument is a tuple
    whose n-th entry is the number of neurons in the n-th hidden layer.
    """
    def __init__(self, structure=(10, 10, 10)):
        super(net, self).__init__()
        self.structure = structure

        self.layer_list = torch.nn.ModuleList()

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
