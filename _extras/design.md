---
title: Lesson Design
---

This page documents the design process and motivation of this lesson material.

**Lesson Title: An Introduction to Deep Learning**

## Target audience

The main audience of this carpentry lesson is PhD students that have little to no experience with deep learning. In addition, we expect them to know basics of statistics and machine learning.

### Learner Profiles

#### Ann from Meteorology

Ann has collected 2-3 GB of structured image data from several autonomous microscope on baloon expeditions into the atmostphere within her PhD programme. Each image has a time stamp to it which can be related to the height of the baloon at this point and the current weather conditions. The images are unstructured and she would like to detect from the images if the baloon traversed a cloud or not. She has tried to do that with standard image processing methods, but the image artifacts to descriminate are somewhat diverse. Ann has used machine learning on tabular data before and would like to use Deep Learning for the images at hand. She saw collaborators in another lab do that and would like to pick up this skill.

#### Barbara from Material Science

Barbara just started her PostDoc in Material Science. Her new group has a large amount of scanning electron miscroscope images stored which exhibit several metals when exposed to a plasma. The team also made the effort to highlight solid deposits in these images and thus obtained 20,000 images with such annotations. Barbara performed some image analysis before and hence has the feeling that Deep Learning may help her in this task. She saw her labmates use ML algorithms for this and is motivated to finally understand these approaches.

#### Dan from Life Sciences

Dan produced a large population of bacteria that were subject to genetic alterations resulting in 10 different phenotypes. The latter can be identified by different colors, shapes and movement speed under a fluorescence microscope. Dan has not a lot of experience with image processing techniques to segment these different objects, but used GUI based tools like [fiji](https://fiji.sc) and others. He has recorded 50-60 movies of 30 minutes each. 10 of these movies have been produced with one type of phenotype only. Dan doesn't consider himself a strong coder, but would need to identify bacteria of the phenotypes in the dataset. He is interested to learn if Deep Learning can help.

#### Eric from Pediatrics Science

Eric ran a large array of clinical trials in his hospital to improve children pharmaceutics for treating a common (non-lethal) virus. He obtained a table that lists the progression of the treatment for each patient, the dose of the drug given, whether the patient was in the placebo group or not, etc. As the table has more than 100 000 rows, Eric is certain that he can use ML to cluster the rows in one column where the data taking was inconsistent. Eric has touched coding here and there where necessary, but never saw it necessary to learn coding. His cheatsheet is his core wisdom with code. So his supervisor invited him to take a course on ML as "this is the tech of these days!" as his boss said.

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
> - Design and train a Neural Network for tabular data
> - Evaluate the performance of a classification network
> - Design and train a Neural Network for image data
> - Troubleshoot the learning process
> - Measure the performance of the network
> - Visualizing data and results
> - Re-use existing network architectures with and without pre-trained weights
>
{: .objectives }


The following offers more details to each learning objective based on Bloom's Taxonomy. For hints on how to use this approach, see [episode 15 of the instructor training](https://carpentries.github.io/instructor-training/15-lesson-study/index.html)

### (episode 1) Prepare input data for use for deep learning

This includes cleaning data, filling missing values, normalizing, and transforming categorical columns into dummy encoding.

After this module, learners can ...

- define a checklist for data analysis steps before applying Deep Learning to the data
- describe criteria by which to judge good or bad data, e.g. how a column's values should be distributed
- execute a min-max normalization on floating point data
- sketch how to insert missing timestamps or literal values (i.e. factors or non-numeric entries)
- implement a transformation of categorical values into a numerical encoding (`int8`)
- argue for or against strategies to normalize data
- formulate techniques to prepare (clean) data for training a deep learning network

### (episode 2) Design and train a Neural Network for tabular data

This includes knowledge for using a naive multilayer perceptron. __Which dataset to use?__

After this module, learners can ...

- list/repeat the three ingredients to a feed forward network: input, hidden layers, output
- classify/categorize parts of a feed forward network when presented a network architecture (as from `keras.model.summary()`)
- describe a fully connected (dense) layer
- describe an activation function
- describe a softmax layer
- argue against abundant use of the sigmoid function (exploding/vanishing gradients)
- experiment with values of a dense layer
- select a layer type depending on the input data
- develop a 3-5 layer network 

### (episode 3) Evaluate the performance of a classification network

How to judge the performance of a trained network.

After this module, learners can ...

- calculate the accuracy of a classifier
- describe the confusion matrix of a classification
- sketch how precision, recall, f1 are calculated based on the confusion matrix
- compare values of precision and recall
- compare accuracy for different network architectures of the MLP
- describe that during training the dataset needs to be split into 2 parts
- describe how to split a dataset into training/test/validation set
- execute a plot to draw the loss per epoch for training and test set

### (episode 4) Design and train a Neural Network for image data

This episode discusses convolutions in neural networks for image data. We will use the `cifar10` or `mnist` or `fashion_mnist` dataset coming with keras.

After this module, learners can ...

- describe a convolutional layer
- describe a max pooling layer
- calculate the output data shape of an image when transformed by a fixed convolutional layer
- interpret errors with convolutional layers
- analyse advantages of convolutional layers with image input data or tabular input data
- execute a 3-5 layer network on the MNIST data (or similar)
- differentiate a dense layer and a convolutional layer


### (episode 5) Monitoring and Troubleshooting the learning process

Often when designing neural networks training will not automatically work very well. This requires setting the parameters of the training algorithm correctly, modifying the design of the network or changing the data pre-processing. After training, the performance of the network should be checked to prevent overfitting.

After this module, learners can ...

- state that cross-validation is used in Deep Learning too
- differentiate a overfitting network from a well-behaved network
- detect when a network is underfitting or overfitting
- design countermeasures for overfitting (e.g. more dropout layers, reduce model size)
- design countermeasures for underfitting (e.g. larger model)
- critique a provided network design
- describe how Drop-Out Layers work

### (episode 6) Visualizing Data and Results

Within each episode how to visualize data and results

After this module, learners can ...

- identify important plots to create at the end of training (provide selected samples and their prediction)
- execute plotting of important variables during training (loss, ROC)
- use tensorboard and related callbacks during training
- examine the results of a partners network
- critique the results of a partners network

### (episode 7) Re-use existing network architectures with and without pre-trained weights

Re-use of architectures is common in deep learning. Especially when using pre-trained weights (transfer-learning) it can also be very powerful.

After this module, learners can ...

- describe what transfer learning stands for
- explain in what situations transfer learning is beneficial
- solve common issues of transfer learning (such as different resolutions of the original training and the training at hand)
- test training under different data shape mitigation strategies
- relate training time of a de-novo networt and a pretrained one
- relate prediction quality of a de-novo networt and a pretrained one

{% include links.md %}
