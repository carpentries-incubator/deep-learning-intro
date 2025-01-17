---
title: Workshop survey templates
---

This page lists possible questions for both pre-workshop and post-workshop surveys, that instructors are free to use. Note, the nature of questions might change throughout the course of development of this lesson module.

In that sense, please provide feedback or experiences with these questions!

## pre-workshop survey

The pre-workshop survey is meant to serve two goals:

- provide the instructors with a way to estimate the knowledge background of learners
- provide the learners another source of evidence, what is expected from them

What worked well was that each participant is asked 2-3 questions for each of the following concepts:

1. self-estimated proficiency in python coding
2. self-estimated proficiency in data science methods
3. self-estimated proficiency in ML methods

Sticking to questions that relate to problem solving rather than focusing on libraries or advanced concepts can help to diversify your learners. But as a whole, the following is meant to support instructors to find out ['what to teach'](https://cdh.carpentries.org/deciding-what-to-teach.html#target-audience). So you'd first start out to describe your target audience and then select subsets of questions that serve this audience.

The following questions all are meant to offer four answers learners can choose from:

> - I know how to do this.
> - I'd consult code I've written.
> - I'd ask a colleague or search for this online.
> - I am not sure what this question is talking about.

This way, the implications of right/wrong answers on the learners are avoided, i.e. no learner needs to feel bad prior to the workshop for not-knowing something. These questions rather aim to probe the self-estimated profiency of learners.

### Questions about python coding

**The following questions range from simple to more advanced.**

- You are provided with a python list of integer values. The list has length 1024 and you would like to obtain all entries from index 50 to 101. How would you do this?

- You need to open 102 data files and extract an object from them. For this, you compose a small function to open a single file which requires 3 input parameters. The parameters are a file location, the name of the object to retrieve and a parameter that controls the verbosity of the function. The latter parameter has the default value “False”.

- Consider a 4 dimensional numpy array of uint32 type numbers. The shape of the array is '(512,3,224,224)' and it is stored in row-major memory ordering. Your goal is to set every other row to 42, i.e. every other entry in dimension 2 to 42. 

- You discovered the numpy 'rot90' method. You observe yourself calling it over and over again with the same parameter for the axis option. You'd like to write a wrapper function which calls 'rot90' having axis set to '(0,1)'. 

- A new version of your favorite package `foo` has just been released on github with a bug fix that you are interested in. You'd like to try this new version out in an environment that is independent of your local user environment which can be populated with 'python -m pip install --user foo'. 

### Questions about data science

**The following questions range from simple to more advanced.**

- You are provided a list of 512 random float values. These values range between 0 and 100. You would like to remove any entry in this list that is larger than 90 or smaller than 10.

- You are provided with a CSV file. The file has 35000 rows. The file has 45 columns. Each column is separated by the “,” symbol from the other. You would like to open the file in python, calculate the arithmetic mean, the minimum and maximum of column number 5, 12 and 39.

- You are provided with a CSV file. The file has 35000 rows. The file has 45 columns. Each column is separated by the “,” symbol from the other. You would like to open the file in python, remove all entries where the value of column 21 is larger than 50. The values removed are to be replaced by 0.

- You are provided with a CSV file. The file has 35000 rows. The file has 45 columns. Each column is separated by the “,” symbol from the other. When you load the file and plot the histogram of column 40, you are suspicious that the floating point values are not normally distributed. But, the producer of the CSV file assures you that all columns are normally distributed. To make sure, you sit down to code a function which tests any given column if it is normally distributed.

- A probability density function (PDF) in statistics is something very distinct from a probability mass function (PMF). What is the difference?

- You are given samples of two sets of observations with type float32. Both arrays have the same shape: "(1024, 1)" You'd like to compare both ensembles if they are produced by the same PDF. How do you do this?

### Questions about machine learning

**The following questions range from simple to more advanced.**

- You are given a dataset from experiments that you want to use for machine learning (13 columns with 25000 rows). One column is particularly useful and is encoded as real numbers in a range from -15 to 12. You would like to normalize this data so that it fits into the range of real numbers between 0 and 1. How would you do this?

- You are helping to organize a conference of more than 1000 attendants. All participants have already paid and are expecting to pick up their conference t-shirt on the first day. Your team is in shock as it discovers that t-shirt sizes have not been recorded during online registration. However, all participants were asked to provide their age, gender, body height and weight. To help out, you sit down to write a python script that predicts the t-shirt size for each participant using a clustering algorithm.

- You are presented a training data set of 10000 samples. You would like to train a classifier on this datasets. However, you observe that class 0 dominates the training set at 60%, the other classes equally share 20% of the rows each. How do you prepare your training procedure?

-  You are given a trained classification model by a colleague. Your peer trained it on a very large dataset. You need it for inference on your data. You reassure yourself if the model you loaded from disk really reproduces the weights by running the `predict` function on an unseen validation data set. The network doesn't make the same predictions as your colleague documented it for you. What could be the cause?

-  You are provided a table of observations with 23 columns and 5000 rows. For half of the rows, there is data available for column 24 as float32 between '[-1, +1]'. You'd like to write a predictor for these missing rows.


