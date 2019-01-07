---
title: "Artificial Neural Network (ANN)"
teaching: 0
exercises: 0
questions:
- "How to prepare data?"
- "How to load data?"
- "How to run your ANN with your data?"
objectives:
- "Learn how to prepare your data for PyTorch"
- "Define your neural network and train it with your data"
- "Test your neural network with your test data"
keypoints:
- "ANN"
---

# Prepare your Data

## Training data versus Test data
splitting 
~~~

~~~
{: .language-python}

## Normalisation

The learning problem for neural networks is sensitive to input scaling (2009, Hastie et.al.). Scaling the inputs determines the effective scaling of the weights and can have a large effect on the quality of the final solution. There are two usual ways: min-max scaling and standardisation. It is mostly recommended to standardise the inputs to have mean zero and standard deviation one. Then, all inputs are treated equally in the regularisation process.
Scikit-learn has a function to standardise. 
~~~
from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.], [ 2.,  0.,  0.],[ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled)
print(X_scaled.mean(axis=0))
print( X_scaled.std(axis=0))
~~~
{: .language-python}




# Load your data

Let's use the MNIST dataset and download it with PyTorch. We define sizes of the training and test batches. Loading this dataset from torchvision, it is possible to normalise it with the *torchvision.transforms.Normalize* by giving the mean (0.1307) and the standard deviation (0.3081) of the dataset. 
It is necessary to set a seed when dealing with random numbers.
~~~
import torch
import torchvision
torch.random.manual_seed(1)

batch_size_train = 64
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
  
~~~
{: .language-python}
# Define your network

In PyTorch, the neural networks are built as classes. The last layer should have an output dimension equal to the number of classes in the classification problem. The first layer has an input size equal to the dimension of the input. 
In the forward function, the input is passed through the layers and the output is returned.

Number of hidden layers

~~~
class ANN(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 300)   #input size
        self.fc2 = torch.nn.Linear(300,300)
        self.fc3 = torch.nn.Linear(300,300)
        self.fc4 = torch.nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # the size -1 is inferred from other dimensions (28*28)
        x = torch.sigmoid(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x          
~~~
{: .language-python}

First, the network class is initialised. Then, we choose an optimiser and a learning rate for the optimisation. In order to update our learning rate, we can use a scheduler to reduce the learning rate. The *ReduceLROnPlateau* reduces the learning rate when a chosen metric has stopped improving after some epochs. In *mode='min'*, the lr will be reduced when the quantity monitored has stopped decreasing.
The criterion usually used for training a classification problem is the Cross Entropy loss.  

~~~
network = ANN()
learning_rate = 0.0001
optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate,
                      weight_decay=0.005) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,mode='min')
criterion = torch.nn.CrossEntropyLoss()         

~~~
{: .language-python}

# Train your neural network
In the training phase, the weights and biases are calculated. The loss is calculated with the criterion and backpropagated to change the models parameters.
~~~
def train(epoch):
    network.train()
    training_loss = 0
    train_corrects = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimiser.zero_grad()
        output = network(data)
        _, preds = torch.max(output.data, 1)
        loss = criterion(output, target)
        loss.backward()
        optimiser.step()
        training_loss += loss.item()
        train_corrects += torch.sum(preds == target.data) 

        train_losses.append(loss.item())
        train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
        
    torch.save(network.state_dict(), 'model.pth')
    torch.save(optimiser.state_dict(), 'optimiser.pth')
  
    epoch_acc= train_corrects.double() / len(train_loader.dataset)
    training_losses.append(training_loss/len(train_loader.dataset))
    print('Epoch {} , Average training loss is {:.6f} and accuracy is {}/{} {:.0f}%'.format((epoch+1),
                        training_loss/len(train_loader),train_corrects.double(),
                                        len(train_loader.dataset),epoch_acc*100.))
~~~
{: .language-python}

# Test your neural network
~~~
def test():
    network.eval()
    test_loss = 0
    test_corrects = 0
    total= 0
    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(test_loader):
            output = network(data)
            _, preds = torch.max(output.data, 1)
            test_loss += criterion(output, target).item()
            total += target.size(0)
            test_corrects += torch.sum(preds == target.data) 
        
        epoch_acc= test_corrects.double() / len(test_loader.dataset)
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
       
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_corrects, len(test_loader.dataset),
            100. * epoch_acc))
        scheduler.step(test_loss)
~~~
{: .language-python}
# Run train and test for network
~~~
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
n_epochs = 10
test()
for epoch in range(n_epochs):
    train(epoch)
    test()
~~~
{: .language-python}

To check the final parameters of the model in each layer, we should call the network parameters. Here the weights in the first layer are presented.
~~~
network.fc1.weight
~~~
{: .language-python}
And the biases:
~~~
network.fc1.bias
~~~
{: .language-python}



# References
Hastie, T., Tibshirani, R., Friedman, J. , The Elements of Statistical Learning, Springer, 2009.
{% include links.md %}
