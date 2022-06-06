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
    

"""
class Net(nn.Module):
    def __init__(self, name='net'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.name = name

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.to(device))))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self, dataset, n_epochs):
        This method allows to train the neural network with the Adam optimizer.
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters())
        
        
        for epoch in range(n_epochs):

            #if epoch % 30 == 0:
                #print("epoch : ", epoch)

            running_loss = 0.0
            for i, data in enumerate(dataset, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0    
                    
        print('Finished Training')
                
        
    def test(self, dataset):
        This method is to calculate the test accuracy
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in dataset:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.forward(images.to(device))
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        
    def test_classes(self, dataset, classes) :
        This method calculates the test accuracy for each class of the cifar10 dataset
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in dataset:
                images, labels = data
                outputs = self.forward(images.to(device))
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

    def save_model(self, PATH_FILE='./') :
        path = PATH_FILE + str(self.name) + '.pth'
        torch.save(self.state_dict(), path)
"""
        
def train_ADAM(train_loader, test_loader, model, epochs=200):
    """This method allows to train the neural network with the Adam optimizer."""
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    
    best_accuracy = 0.
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
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            i+=1
            
        loss /= cnt
        accuracy *= 100. / cnt
        print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
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
        print(f"Epoch: {epoch}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
    print(f"Best test accuracy: {best_accuracy}")


def test(test_loader, model, dataset):
    """"This method is to calculate the test accuracy"""
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

def test_classes(test_loader, model, classes) :
    """This method calculates the test accuracy for each class of the cifar10 dataset"""
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
    path = PATH_FILE + str(FILENAME) + '.pth'
    torch.save(model.state_dict(), path)
        

def train_minimizer(train_loader, test_loader, model, minimizer=ASAM, epochs=200, rho_=0.5, smoothing_=0): 
    # Data Loader
    #train_loader, test_loader = load_cifar(eval(args.dataset), args.batch_size)
    #num_classes = 10 if args.dataset == 'CIFAR10' else 100

    # Model
    #model = wrn28_10(num_classes=10).cuda()

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
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            i +=1
                    
        loss /= cnt
        accuracy *= 100. / cnt
        print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
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
        print(f"Epoch: {epoch}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
    print(f"Best test accuracy: {best_accuracy}")
    
    
    
    
    
    
    
    
    
    
    
"""    
def ASAM_train(self, dataset, n_epochs):
        This method allows to train the neural network with the Adam optimizer and Asam minimizer.
        #Essayer de mettre le learning rate schedulter (mis dans exemple samsung)
        #Essayer de mettre model.train() (Possible que ce soit non négligable, apparement permet de mettre layer en training mode)
        

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters())
        minimizer = ASAM(optimizer, self, 0.5, 0.01) #Le model c'est la structure du NN

        
        for epoch in range(n_epochs):

            #self.train() #ATTENTION

            running_loss = 0.0
            for i, data in enumerate(dataset, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                #optimizer.zero_grad()
                
                #Ascent Step
                outputs = self.forward(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                #loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward() #attention j'ai ajouté le .mean()
                minimizer.ascent_step()

                # Descent Step
                criterion(self.forward(inputs.to(device)), labels.to(device)).mean().backward() #criterion(model(inputs), targets).mean().backward()
                minimizer.descent_step()
                
                # print statistics
                with torch.no_grad():
                    running_loss += loss.sum().item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0    
                    
        print('Finished Training')
                
    def SAM_train(self, dataset, n_epochs):
        This method allows to train the neural network with the Adam optimizer and sam minimizer.
        #Essayer de mettre le learning rate schedulter (mis dans exemple samsung)
        #Essayer de mettre modet.train() (Possible que ce soit non négligable, apparement permet de mettre layer en training mode)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters())
        minimizer = SAM(optimizer, self, 0.5, 0.01) #Le model c'est la structure du NN

        #self.train()
        
        for epoch in range(n_epochs):

            self.train()
            running_loss = 0.0
            for i, data in enumerate(dataset, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                #optimizer.zero_grad()

                #Ascent Step
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.mean().backward() #attention j'ai ajouté le .mean()
                minimizer.ascent_step()

                # Descent Step
                criterion(self.forward(inputs), labels).mean().backward()
                minimizer.descent_step()

                # print statistics
                with torch.no_grad():
                    running_loss += loss.sum().item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0    

        print('Finished Training')
"""