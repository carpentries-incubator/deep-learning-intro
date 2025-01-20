---
title: Lesson design
---

This page documents the design process and motivation of this lesson material.

**Lesson Title: An Introduction to Deep Learning**

## Target audience

The main audience of this carpentry lesson is PhD students that have little to no experience with deep learning. In addition, we expect them to know basics of statistics and machine learning.

### Notes

- Probably have overhyped expectations of deep learning.
- They don’t know if it’s the right tool for their situations.
- They have no idea what it takes to actually do deep learning.
- Want to quickly have some useful skills for their own data.

#### Required Pre-Knowledge

- **Python** – Previous programming experience in Python is required (Refer to Python Data Carpentry Lesson)
- **Pandas** – Knowledge of the Pandas Python package
- **Basic Machine Learning Knowledge** – Data cleaning, train & test split, overfitting & underfitting, metrics (accuracy, recall, etc.),

## Learning objectives

> ## Overview
> After following this lesson, learners will be able to:
>  
> - Prepare input data for use for deep learning
> - Design and train a Deep Neural Network
> - Troubleshoot the learning process
> - Measure the performance of the network
> - Visualizing data and results
> - Re-use existing network architectures with and without pre-trained weights
>


The following offers more details to each learning objective based on Bloom's Taxonomy. For hints on how to use this approach, see [lesson 18 of the instructor training](https://carpentries.github.io/instructor-training/18-preparation.html)

### Prepare input data for use for deep learning

This includes cleaning data, filling missing values, normalizing, and transforming categorical columns into dummy encoding.

After this module, learners can ...

- define a checklist for data analysis steps before applying Deep Learning to the data
- describe criteria by which to judge good or bad data, e.g. how a column's values should be distributed
- execute a min-max normalization on floating point data
- sketch how to insert missing timestamps or literal values (i.e. factors or non-numeric entries)
- implement a transformation of categorical values into a numerical encoding (`int8`)
- argue for or against strategies to normalize data
- formulate techniques to prepare (clean) data for training a deep learning network

### Design and train a Deep Neural Network

This includes knowledge of when to different types of layers

After this module, learners can ...

- list/repeat the three ingredients to a feed forward network: input, hidden layers, output
- classify/categorize parts of a feed forward network when presented a network architecture (as from `keras.model.summary()`)
- describe a fully connected (dense) layer
- describe a convolutional layer
- describe a max pooling layer
- describe an activation function
- describe a softmax layer
- argue against abundant use of the sigmoid function (exploding/vanishing gradients)
- calculate the output data shape of an image when transformed by a fixed convolutional layer
- interpret errors with convolutional layers
- execute a 3 layer network on the MNIST data (or similar)
- differentiate a dense layer and a convolutional layer
- experiment with values of dense layer and a convolutional layer
- select a layer type depending on the input data
- develop a 5 layer network that comprises both layer types

### Monitoring and Troubleshooting the learning process

Often when designing neural networks training will not automatically work very well. This requires setting the parameters of the training algorithm correctly, modifying the design of the network or changing the data pre-processing. After training, the performance of the network should be checked to prevent overfitting.

After this module, learners can ...

- define precision and recall/accuracy for a classification task
- state that cross-validation is used in Deep Learning too
- describe how to split a dataset into training/test/validation set
- describe how Drop-Out Layers work
- execute a plot to draw the loss per epoch for training and test set
- compare values of precision and recall
- differentiate a overfitting network from a well-behaved network
- detect when a network is underfitting or overfitting
- design countermeasures for overfitting (e.g. more dropout layers, reduce model size)
- design countermeasures for underfitting (e.g. larger model)
- critique a provided network design

### Visualizing Data and Results

Within each episode how to visualize data and results

After this module, learners can ...

- identify important plots to create at the end of training (provide selected samples and their prediction)
- execute plotting of important variables during training (loss, ROC)
- use tensorboard and related callbacks during training
- examine the results of a partners network
- critique the results of a partners network

### Re-use existing network architectures with and without pre-trained weights

Re-use of architectures is common in deep learning. Especially when using pre-trained weights (transfer-learning) it can also be very powerful.

After this module, learners can ...

- describe what transfer learning stands for
-  explain in what situations transfer learning is beneficial
- solve common issues of transfer learning (such as different resolutions of the original training and the training at hand)
- test training under different data shape mitigation strategies
- relate training time of a de-novo network and a pretrained one
- relate prediction quality of a de-novo network and a pretrained one

