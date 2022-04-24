from abc import ABC, abstractmethod
import torch
from einops import rearrange
import math

class Dataset_2D(ABC):
    """Abstract class for 2-classes datasets in 2 dimensions.

    n_points (int) : the number of samples in each class.
    """

    def __init__(self, n_points):
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

    def __init__(self, n_points):
        super().__init__(n_points)
    
    def _create_dataset(self, plot=False):
        
        r11 = torch.abs(torch.random.normal(0, 1.6, self.n_points))
        theta11 = torch.random.uniform(0, 2*math.pi, self.n_points)
        x11 = r11*torch.cos(theta11)+5
        y11 = r11*torch.sin(theta11)

        r12 = torch.abs(torch.random.normal(0, 1.6, self.n_points))
        theta12 = torch.random.uniform(0, 2*math.pi, self.n_points)
        x12 = r12*torch.cos(theta12)-5
        y12 = r12*torch.sin(theta12)

        C11 = torch.transpose(torch.tensor([x11, y11]))
        C12 = torch.transpose(torch.tensor([x12, y12]))
        C1 = torch.cat([C11, C12], axis=0)
        C1 = torch.cat([torch.zeros((2*self.n_points, 1)), C1], 1)

        r21 = torch.random.normal(0, 1, self.n_points)+5
        theta21 = torch.random.uniform(0, 2*math.pi, self.n_points)
        x21 = r21*torch.cos(theta21)+5
        y21 = r21*torch.sin(theta21)

        r22 = torch.random.normal(0, 1, self.n_points)+5
        theta22 = torch.random.uniform(0, 2*math.pi, self.n_points)
        x22 = r22*torch.cos(theta22)-5
        y22 = r22*torch.sin(theta22)

        C21 = torch.transpose(torch.tensor([x21, y21]))
        C22 = torch.transpose(torch.tensor([x22, y22]))
        C2 = torch.cat([C21, C22], axis=0)
        C2 = torch.cat([torch.ones((2*self.n_points, 1)), C2], 1)

        dataset = torch.cat([C1, C2], 0)
        dataset = dataset[torch.randperm(dataset.size()[0])]
        data = dataset[:, 1:].float()
        labels = dataset[:, :1].long()
        labels = rearrange(labels, 'h w -> (h w)')

        return data, labels
