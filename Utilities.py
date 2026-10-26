#from module 07B
# Import necessary libraries

# Import the PyTorch library
# PyTorch is an open-source machine learning library used for applications such as computer vision & natural language processing
import torch

# Import the neural network module from PyTorch
# torch.nn provides classes & functions to build neural networks
import torch.nn as nn

# Import the optimization module from PyTorch
# torch.optim provides various optimization algorithms such as SGD, Adam, etc.
import torch.optim as optim

# Import the torchvision library
# torchvision provides tools to handle image & video data
import torchvision

# Import the transforms module from torchvision
# torchvision.transforms provides common image transformations for data augmentation & preprocessing
import torchvision.transforms as transforms

# Import the matplotlib library for plotting
# matplotlib.pyplot provides functions to create various types of plots & visualizations
import matplotlib.pyplot as plt

# Import the tqdm library for progress bars
# tqdm provides a fast, extensible progress bar for loops & tasks
from tqdm import tqdm

# Import the numpy library
# numpy provides support for large, multi-dimensional arrays & matrices, along with a collection of mathematical functions to operate on these arrays
import numpy as np

# Import the argparse library
# https://docs.python.org/3/library/argparse.html
import argparse 

# Create the parser that holds the rules for the command-line arguments
# Below are all the default values from the assignment
def get_args():
    # The description is shown at the top of the help message (python run.py -h)
    parser = argparse.ArgumentParser(description= " using a CNN on a MNIST with Pytorch")
    parser.add_argument("-e", "--num_epochs", type = int, default = 2, help= "numbers of epochs (default: 2)")
    parser.add_argument("-b","--batch_size", type= int, default= 1000, help= "batch size (default:1000)")
    parser.add_argument("-l", "--learning_rate", type= float, default=0.001, help= "learning rate(defualt: .001)" )

    return parser.parse_args()

# MNIST dataset
# Download & load the training dataset
# Download & load the test dataset
def get_loaders(batch_size):
    train_dataset= torchvision.datasets.MNIST(root= './data', train=True, transform= transforms.ToTensor(), download= True)
    test_dataset= torchvision.datasets.MNIST(root= './data', train = False, transform= transforms.ToTensor())
    
    # Data loader
    # DataLoader provides an iterable over the dataset with support for batching
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader



# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
         # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Fully connected layer: input size 7*7*64, output size 1000
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Trainer class: holds the model, data loaders, loss, & optimizer, & runs the training
#much of the code comes from Module 7 B 
class Trainer:
    def __init__(self, model, train_loader, test_loader, learning_rate, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        # GPU if available, otherwise CPU
        self.device = device
        # Loss & optimizer
        self.criterion = nn.CrossEntropyLoss()
        # Adam optimizer with the specified learning rate
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def track_accuracy(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.model.train()
        return 100 * correct / total

    def train(self, num_epochs): 
        # List to store loss values
        # List to store training accuracy values
        # List to store testing accuracy values
        loss_list, acc_train, acc_test = [], [], []
        # Loop over the number of epochs
        for epoch in range(num_epochs):
                    # Loop over each batch in the training DataLoader
            pbar = tqdm(self.train_loader,
                        desc=f"Training: {epoch + 1:03d}/{num_epochs}",
                        ncols=125, leave=True)
            # Creating a running loss empty list
            running_loss = []
            for i, (images, labels) in enumerate(pbar, 1):
            # Important note: In PyTorch, the images in a batch is typically represented as (batch_size, channels, height, width)
            # For example, (100, 1, 28, 28) for the MNIST data
                images = images.to(self.device) # Move images to the configured device
                labels = labels.to(self.device) # Move labels to the configured device


                # Forward pass: compute predicted y by passing x to the model
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())

                # Backward pass & optimize
                self.optimizer.zero_grad() # Zero the gradients
                loss.backward() # Backward pass
                self.optimizer.step() # Update weights

                pbar.set_postfix(loss=f"{sum(running_loss) / i : 10.6f}")
            # Get the average of all losses in the running_loss as the loss of the current epoch
            loss_list.append(sum(running_loss) / i)
            # Track accuracy on the train set
            acc_train.append(self.track_accuracy(self.train_loader))
            # Track accuracy on the test set
            acc_test.append(self.track_accuracy(self.test_loader))
        return loss_list, acc_train, acc_test

#@title Plot
#given by assignment
def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    # Plot the loss & accuracy curves
    plt.figure(figsize=(10, 4))

    # Plot the training loss over iterations
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(acc_train)
    plt.plot(acc_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Model Accuracy")
    plt.legend(['Training', 'Testing'])
    plt.savefig(save_path)
    plt.close()