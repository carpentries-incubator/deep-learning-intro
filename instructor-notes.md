---
title: Instructor Notes
---

## Setup before the lesson
The required python packages for this lesson often result in installation issues,
so it is advisable to organize a pre-workshop setup session where learners can show their installation and get help with problems.

Installations on learners' devices have the advantage of lowering the threshold to continue with the material beyond the workshop. Note though, that this lesson can also be taught on a cloud environment such as [Google colab](https://colab.research.google.com/) or [My Binder](https://github.com/carpentries/scaffolds/blob/master/instructions/workshop-coordination.md#my-binder). This can serve as a backup environment if local installations fail. Some cloud environments offer the possibility to run the code on a GPU, which significantly increases the runtime of deep learning code.

## Deep learning workflow
The episodes are quite long, because they cover a full cycle of the deep learning workflow. It really helps to structure your teaching by making it clear where in the 10-step deep learning workflow we are. You can for example use headers in your notebook for each of the steps in the workflow.

## Episode 3: Monitor the training process
When episode 3 is taught on a different day then episode 2, it is very useful to start with a recap of episode 2. The Key Points of episode 2 can be iterated, and you can go through the code of the previous session (without actually running it). This will help learners in the big exercise on creating a neural network.

The following exercises work well to do in groups / break-out rooms:
- Split data into training, validation, and test set
- Create the neural network. Note that this is a fairly challenging exercise, but learners should be able to do this based on their experiences in episode 2 (see also remark about recap).
- Predict the labels for both training and test set and compare to the true values
- Try to reduce the degree of overfitting by lowering the number of parameters
- Create a similar scatter plot for a reasonable baseline
- Open question: What could be next steps to further improve the model?
All other exercises are small and can be done individually.

## Presentation slides
There are no official presentation slides for this lesson, but this material does include some example
slides from when this course was taught by different institutions. These slides can be found in 
the 
[slides](https://github.com/carpentries-incubator/deep-learning-intro/tree/main/instructors/slides)
folder.
