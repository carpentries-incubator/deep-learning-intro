---
title: "Outlook"
teaching: 20
exercises: 20
---

::: questions
- How does what I learned in this course translate to real-world problems?
- How do I organise a deep learning project?
- What are next steps to take after this course?
:::

::: objectives
- Understand that what we learned in this course can be applied to real-world problems
- Use best practices for organising a deep learning project
- Identify next steps to take after this course
:::

You have come to the end of this course.
In this episode we will look back at what we have learned so far, how to apply that to real-world problems, and identify
next steps to take to start applying deep learning in your own projects.

## Real-world application
To introduce the core concepts of deep learning we have used quite simple machine learning problems.
But how does what we learned so far apply to real-world applications?

To illustrate that what we learned is actually the basis of succesful applications in research,
we will have a look at an example from the field of cheminformatics.

We will have a look at [this notebook](https://github.com/matchms/ms2deepscore/blob/0.4.0/notebooks/MS2DeepScore_tutorial.ipynb).
It is part of the codebase for [this paper](https://doi.org/10.1186/s13321-021-00558-4).

In short, the deep learning problem is that of finding out how similar two molecules are in terms of their molecular properties,
based on their mass spectrum.
You can compare this to comparing two pictures of animals, and predicting how similar they are.
A siamese neural network is used to solve the problem.
In a siamese neural network you have two input vectors, let's say two images of animals or two mass spectra.
They pass through a base network. Instead of outputting a class or number with one or a few output neurons, the output layer
of the base network is a whole vector of for example 100 neurons. After passing through the base network, you end up with two of these
vectors representing the two inputs. The goal of the base network is to output a meaningful representation of the input (this is called an embedding).
The next step is to compute the cosine similarity between these two output vectors,
cosine similarity is a measure for how similar two vectors are to each other, ranging from 0 (completely different) to 1 (identical).
This cosine similarity is compared to the actual similarity between the two inputs and this error is used to update the weights in the network.

Don't worry if you do not fully understand the deep learning problem and the approach that is taken here.
We just want you to appreciate that you already learned enough to be able to do this yourself in your own domain.

::: instructor
You don't have to use this project as an example.
It works best to use a suitable deep learning project that you know well and are passionate about.
:::
::: challenge
## Exercise: A real-world deep learning application

1. Looking at the 'Model training' section of the notebook, what do you recognize from what you learned in this course?
2. Can you identify the different steps of the deep learning workflow in this notebook?
3. (Optional): Try to understand the neural network architecture from the first figure of [the paper](https://doi.org/10.1186/s13321-021-00558-4).
    a. Why are there 10.000 neurons in the input layer?
    b. What do you think would happen if you would decrease the size of spectral embedding layer drastically, to for example 5 neurons?

:::: solution
## Solution
1. The model summary for the Siamese model is more complex than what we have seen so far,
but it is basically a repetition of Dense, BatchNorm, and Dropout layers.
The syntax for training and evaluating the model is the same as what we learned in this course.
EarlyStopping as well as the Adam optimizer is used.
2. The different steps are not as clearly defined as in this course, but you should be able to identify '3: Data preparation',
'4: Choose a pretrained model or start building architecture from scratch', '5: Choose a loss function and optimizer', '6: Train the model',
'7: Make predictions' (which is called 'Model inference' in this notebook), and '10: Save model'.
3. (optional)
    a. Because the shape of the input is 10.000. More specifically, the spectrum is binned into a size 10.000 vector,
    apparently this is a good size to represent the mass spectrum.
    b. This would force the neural network to have a representation of the mass spectrum in only 5 numbers.
    This representation would probably be more generic, but might fail to capture all the characteristics found in the spectrum.
    This would likely result in underfitting.
::::
:::

Hopefully you can appreciate that what you learned in this course, can be applied to real-world problems as well.

::: callout
## Extensive data preparation
You might have noticed that the data preparation for this example is much more extensive than what we have done so far
in this course. This is quite common for applied deep learning projects. It is said that 90% of the time in a
deep learning problem is spent on data preparation, and only 10% on modeling!
:::

::: discussion
## Discussion: Large Language Models and prompt engineering
Large Language Models (LLMs) are deep learning models that are able to perform general-purpose language generation.
They are trained on large amounts of texts, such all pages of Wikipedia. 
In recent years the quality of LLMs language understanding and generation has increased tremendously, and since the launch of generative chatbot ChatGPT in 2022 the power of LLMs is now appreciated by the general public.

It is becoming more and more feasible to unleash this power in scientific research. For example, the authors of [Zheng et al. (2023)](https://doi.org/10.1021/jacs.3c05819) guided ChatGPT in the automation of extracting chemical information from a large amount of research articles. The authors did not implement a deep learning model themselves, but instead they designed the right input for ChatGPT (called a 'prompt') that would produce optimal outputs. This is called prompt engineering. A highly simplified example of such a prompt would be: "Given compounds X and Y and context Z, what are the chemical details of the reaction?"

Developments in LLM research are moving fast, at the end of 2023 the newest ChatGPT version [could take images and sound as input](https://openai.com/blog/chatgpt-can-now-see-hear-and-speak). 
In theory, this means that you can solve the Cifar-10 image classification problem from the previous episode by prompt engineering, with prompts similar to "Which out of these categories: [LIST OF CATEGORIES] is depicted in the image".

**Discuss the following statement with your neighbors:**

_In a few years most machine learning problems in scientific research can be solved with prompt engineering._
:::

## Organising deep learning projects
As you might have noticed already in this course, deep learning projects can quickly become messy.
Here follow some best practices for keeping your projects organized:

### 1. Organise experiments in notebooks
Jupyter notebooks are a useful tool for doing deep learning experiments.
You can very easily modify your code bit by bit, and interactively look at the results.
In addition you can explain why you are doing things in markdown cells.
- As a rule of thumb do one approach or experiment in one notebook.
- Give consistent and meaningful names to notebooks, such as: `01-all-cities-simple-cnn.ipynb`
- Add a rationale on top and a conclusion on the bottom of each notebook

[_Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks_](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007007) provides further advice on how to maximise the usefulness and reproducibility of experiments captured in a notebook.

### 2. Use Python modules
Code that is repeatedly used should live in a Python module and not be copied to multiple notebooks.
You can import functions and classes from the module(s) in the notebooks.
This way you can remove a lot of code definition from your notebooks and have a focus on the actual experiment.

### 3. Keep track of your results in a central place
Always evaluate your experiments in the same way, on the exact same test set.
Document the results of your experiments in a consistent and meaningful way.
You can use a simple spreadsheet such as this:

| MODEL NAME              | MODEL DESCRIPTION                          | RMSE | TESTSET NAME  | GITHUB COMMIT | COMMENTS |
|-------------------------|--------------------------------------------|------|---------------|---------------|----------|
| weather_prediction_v1.0 | Basel features only, 10 years. nn: 100-50  | 3.21 | 10_years_v1.0 |  ed28d85      |          |
| weather_prediction_v1.1 | all features, 10 years. nn: 100-50         | 3.35 | 10_years_v1.0 |  4427b78      |          |

You could also use a tool such as [Weights and Biases](https://wandb.ai/site) for this.

::: callout
## Cookiecutter data science
If you want to get more pointers for organising deep learning, or data science projects in general,
we recommend [Cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/).
It is a template for initiating an organized data science project folder structure
that you can adapt to your own needs.
:::
## Next steps
You now understand the basic principles of deep learning and are able to implement your own deep learning pipelines in Python.
But there is still so much to learn and do!

Here are some suggestions for next steps you can take in your endeavor to become a deep learning expert:

* Learn more by going through a few of [the learning resources we have compiled for you](learners/reference.md#external-references)
* Apply what you have learned to your own projects. Use the deep learning workflow to structure your work.
Start as simple as possible, and incrementally increase the complexity of your approach.
* Compete in a [Kaggle competition](https://www.kaggle.com/competitions) to practice what you have learned.
* Get access to a GPU. Your deep learning experiments will progress much quicker if you have to wait for your network to train
in a few seconds instead of hours (which is the order of magnitude of speedup you can expect from training on a GPU instead of CPU).
Tensorflow/Keras will automatically detect and use a GPU if it is available on your system without any code changes.
A simple and quick way to get access to a GPU is to use [Google Colab](https://colab.google/)

::: keypoints
- Although the data preparation and model architectures are somewhat more complex,
what we have learned in this course can directly be applied to real-world problems
- Use what you have learned in this course as a basis for your own learning trajectory in the world of deep learning
:::
