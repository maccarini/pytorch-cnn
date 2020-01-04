# PyTorch Convolutional Neural Network
An implementation of a convolutional neural network using pyTorch. Grid search is used to find the optimal hyperparameters and
tensorboard is used to visualize and compare the results.

## The dataset
The dataset used for this project is the Fashion MNIST dataset . It consists of 70000, 28 x 28 grayscale images of clothing items with 60000 being in the training set and 10000 being in the test set. Zalando intends Fashion-MNIST to serve as a "direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms". The dataset contains 10 classes which are listed bellow

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

A sample from the dataset can be visualized as follows
```python
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

sample = train_set.data[0]
plt.imshow(sample)
```
![sample-data](https://github.com/maccarini/pytorch-cnn/blob/master/assets/sample.png "Sample image")

We can also get its corresponding label
```python
sample_label = train_set.targets[0].item()
sample_label
```
And we get the number that maps to a class, in this case an ankle boot
```
9
```
Further information about this dataset is available in [Zalando's official repository](https://github.com/zalandoresearch/fashion-mnist). 

## The problem
The goal of this neural network is to predict clothing articles from the 10 classes with the highest possible accuracy, for this we will use a convolutional neural network consisting of 2 convolutional layers (with ReLU as activation function), and two hidden layer for the fully conected layer. Batch normalization is applied on the convolutional layers.
Additionally, a small grid search is applied to find optimal hyperparameters and tensorboard is used to visualize the results.

## Results
As expected the loss function decreased after being optimized every epoch, we end up with a curve like this one
![train-loss-curve](https://github.com/maccarini/pytorch-cnn/blob/master/assets/loss_decrease.png "Loss")

After the training process concluded we ended up with 6 diferent combinations of learning rates and batch sizes, which can be visualized in tensorboard running the following command on the command line
```
tensorboard --logdir=./data --port=8008
```
There you can see accuracy, loss and val loss comparisons between runs. All of the results were obtain after training for 25 epochs.
This is the loss comparison between runs.
![tb-comparison](https://github.com/maccarini/pytorch-cnn/blob/master/assets/tb-loss.svg "TensorBoard losses")

Aditionally we compute the confusion matrix and plot it as a heat map to visualize how accurately our network is predicting the classes and most importantly, what classes the network is predicting wrongly.
![confusion-matrix](https://github.com/maccarini/pytorch-cnn/blob/master/assets/heat-map.png "Confusion Matrix")

As we can see the network predicts accurately most of the time, but sometimes it predicts wrong. Looking at the confusion matrix we can
see the classes the network predicts wrongly with higher frecuency, for example we can see that the network predicts many times the input to be a shirt when it is actually a t-shirt which is understandable taking in account the similarities between both classes.
