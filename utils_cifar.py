from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from timm.loss import LabelSmoothingCrossEntropy
from homura.vision.models.cifar_resnet import wrn28_10


def imshow(img):
    """Function to show an image"""
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_cifar(batch_size=256, num_workers=2):
    """Function which aims to load cifar and returns the train loader, test loader and the differents classes"""
    """The values for the mean and the std used in the transformation of the dataset"""
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Transforms
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(mean, std)
                         ])
 
    # DataLoader
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers)
    
    return train_loader, test_loader, classes
    


class Net(nn.Module):
    """Simple neural network able to work on cifar10"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.to(device))))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) #flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
def train_ADAM(train_loader, test_loader, model, epochs=200, plot=True):
    """
    This method allows to train the neural network on cifar10 with the Adam optimizer.
    It prints the training loss and accuracy, and the testing loss and accuracy
        Inputs : *trainloader : Dataloader of the training set
                 *testloader : Dataloader of the test set 
                 *model : neural network architecture used
                 *epochs : number of time we go trhough the dataset to train the model
         Output : NONE
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    
    best_accuracy = 0.
    test_acc = []
    train_acc = []
    for epoch in range(epochs):
        # Train
        model.train()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        running_loss = 0.
        i = 0
        for inputs, targets in train_loader:
            # get the inputs; data is a list of [inputs, targets]
            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            predictions = model(inputs.to(device))
            batch_loss = criterion(predictions, targets.to(device))
            batch_loss.mean().backward()
            optimizer.step()

            with torch.no_grad():
                loss += batch_loss.sum().item()
                running_loss += batch_loss.sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
            
            if (i+1) % 100 == 0:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            i+=1
            
        loss /= cnt
        accuracy *= 100. / cnt
        train_acc.append(accuracy)
        print(f"Epoch: {epoch+1}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")

        # Test
        model.eval()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs)
                loss += criterion(predictions, targets).sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            loss /= cnt
            accuracy *= 100. / cnt
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            
        test_acc.append(accuracy)
        print(f"Epoch: {epoch+1}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
    print(f"Best test accuracy: {best_accuracy} %")
    
    if(plot) :
        plt.plot(np.arange(1, epochs+1), train_acc, label='train')
        plt.plot(np.arange(1, epochs+1), test_acc, label='test')
        plt.legend()
        plt.xlabel('epochs'); plt.ylabel('Accuracy (%)')
        plt.title('Accuracy of '+type(model).__name__+' in function of the epoch on cifar10')
        plt.show()
    
def train_minimizer(train_loader, test_loader, model, minimizer=ASAM, epochs=200, rho_=0.5, smoothing_=0, plot=True): 
    """
    This method allows to train the neural network on cifar10 with the SAM or ASAM minimizer.
    It prints the training loss and accuracy, and the testing loss and accuracy
        Inputs : *trainloader : Dataloader of the training set
                 *testloader : Dataloader of the test set 
                 *model : neural network architecture used
                 *epochs : number of time we go trhough the dataset to train the model
                 *rho_ : Neighborhood size ρ>0
                 *smoothing_ : smoothing factor for the LabelSmoothingCrossEntropy 
                         (Use this to not punish model as harshly, such as when incorrect labels are expecte)
         Output : NONE
    """

    # Minimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, 
                                momentum=0.9, weight_decay=5e-4)
    minimizer = minimizer(optimizer, model, rho=rho_, eta=0.0)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, epochs)

    # Loss Functions
    if smoothing_!=0:
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing_)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.
    train_acc = []
    test_acc = []
    for epoch in range(epochs):
        # Train
        model.train()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        running_loss = 0.
        i = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Ascent Step
            predictions = model(inputs)
            batch_loss = criterion(predictions, targets)
            batch_loss.mean().backward()
            minimizer.ascent_step()

            # Descent Step
            criterion(model(inputs), targets).mean().backward()
            minimizer.descent_step()

            with torch.no_grad():
                loss += batch_loss.sum().item()
                running_loss += batch_loss.sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
            
            if (i+1) % 100 == 0:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            i +=1
                    
        loss /= cnt
        accuracy *= 100. / cnt
        train_acc.append(accuracy)
        print(f"Epoch: {epoch+1}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
        scheduler.step()

        # Test
        model.eval()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs)
                loss += criterion(predictions, targets).sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            loss /= cnt
            accuracy *= 100. / cnt
        if best_accuracy < accuracy:
            best_accuracy = accuracy
        test_acc.append(accuracy)
        print(f"Epoch: {epoch+1}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
    print(f"Best test accuracy: {best_accuracy} %")

    if(plot) :
        plt.plot(np.arange(1, epochs+1), train_acc, label='train')
        plt.plot(np.arange(1, epochs+1), test_acc, label='test')
        plt.legend()
        plt.xlabel('epochs'); plt.ylabel('Accuracy (%)')
        plt.title('Accuracy of '+type(model).__name__+' in function of the epoch on cifar10')
        plt.show()

def test(dataset, model):
    """"
    This method is to calculate the test accuracy
    Inputs : * dataset : dataloader of the test set
             * model : model we want to use to test its accuracy
    """
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(device))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def test_classes(dataset, model, classes) :
    """
    This method calculates the test accuracy for each class of the cifar10 dataset
    Inputs : * dataset : dataloader of the test set
             * model : model we want to use to test its accuracy
             * classes : The 10 different classes of cifar10
    """
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            outputs = model(images.to(device))
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def save_model(model, FILENAME, PATH_FILE='./') :
    """
    Function that saves the weights of the model in a file
    Inputs : * model : model we want to save
             * FILENAME : Name of the file created
             * PATH_FILE : where the file will be stored
    """
    path = PATH_FILE + str(FILENAME) + '.pth'
    torch.save(model.state_dict(), path)