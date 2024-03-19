---
title: 'Introduction to deep learning: Carpentries-style hands-on lesson material for introducing researchers to deep learning'
tags:
  - Python
  - deep learning
  - machine learning
  - Keras
  - neural networks
authors:
  - name: Sven A. van der Burg
    orcid: 0000-0003-1250-6968
    affiliation: 1 # (Multiple affiliations must be quoted, like "1, 2")
  - name: Anne Fouilloux
    orcid: 0000-0002-1784-2920
    affiliation: 2
  - name: Florian Huber
    orcid: 0000-0002-3535-9406
    affiliation: "1, 3"
  - name: Dafne van Kuppevelt
    orcid: 0000-0002-2662-1994
    affiliation: 1
  - name: Peter Steinbach
    orcid: 0000-0002-4974-230X
    affiliation: 4
  - name: Berend Weel
    orcid: 0000-0002-9693-9332
    affiliation: 1
  - name: Colin Sauze
    orcid: 0000-0001-5368-9217
    affiliation: 5
  - name: Samantha Wittke
    orcid: 0000-0002-9625-7235
    affiliation: "6,7"
  - name: Djura Smits
    orcid: 0000-0003-4096-0260
    affiliation: 1
  - name: Cunliang Geng
    orcid: 0000-0002-1409-8358
    affiliation: 1
  - name: Pranav Chandramouli
    orcid: 0000-0002-7896-2969
    affiliation: 1
  - name: Toby Hodges
    orcid: 0000-0003-1766-456X
    affiliation: 8
    

affiliations:
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 1
 - name: Simula Research Laboratory, Oslo, Norway
   index: 2
 - name: Düsseldorf University of Applied Sciences, Düsseldorf, Germany
   index: 3
 - name: Helmholtz-Zentrum Dresden-Rossendorf, Dresden, Germany
   index: 4
 - name: National Oceanography Centre, Liverpool, Great-Britain
   index: 5
 - name: CSC - IT center for Science, Espoo, Finland
   index: 6
 - name: Aalto University, Espoo, Finland
   index: 7
 - name: The Carpentries, USA
   index: 8
date: 8 August 2023
bibliography: paper.bib

---

# Summary
This article describes a hands-on introduction to the first steps in deep learning, 
intended for researchers who are familiar with (non-deep) machine learning.

The use of deep learning has seen a sharp increase in popularity and applicability over the last decade. 
While deep learning can be a useful tool for researchers from a wide range of domains, 
taking the first steps in the world of deep learning can be somewhat intimidating. 
This introduction aims to cover the fundamentals of deep learning in a practical and hands-on manner. By the end of the course, students will be able to train their first neural network and understand the subsequent steps needed to improve the model.

The lesson starts by explaining the basic concepts of neural networks, 
and then guides learners through the different steps of a deep learning workflow.  
After following this lesson, 
learners will be able to prepare data for deep learning, 
implement a basic deep learning model in Python with Keras, 
monitor and troubleshoot the training process, and implement different layer types, 
such as convolutional layers.

# Statement of Need
There are many free online course materials on deep learning, 
see for example: @noauthor_fastai_nodate; @noauthor_udemy_nodate; @noauthor_udemy_nodate-1; @noauthor_udemy_nodate-2; @noauthor_coursera_nodate; @noauthor_freecodecamporg_2022. 

Nonetheless, these resources are often not available open-source and can thus not be easily adapted to the students' needs. 
Also, these resources are intended to use for self-study. Our material can be used for self-study, but it is primarily developed for instructors to use in a workshop.
In addition, although a diverse range of online courses already exists, few are targeted towards academic researchers.

Many computing centers offer (local) deep learning courses, such as @noauthor_csc-_nodate. 
But the lesson material, if it is available, is not easily adopted outside the course organisation.

What works well for learners is to both make them familiar with the key concepts, and also let them 
practice with how to implement it. Eventually resulting in an increase in confidence and  the conviction that 'I can do this myself'. 
The key to getting there is live coding: before the course, learners have to setup a working environment on their own computer.
During the course, learners type in the commands that are explained by the instructor on their own computer.
This design is based on the Software Carpentry [@wilson_software_2006] philosophy.
Live coding ensures that learners master the programmatic implementation of deep learning at the end of the course.
We believe that this makes our lesson a unique and crucial resource.

Researchers can often only free a limited amount of time (maximum 5 consecutive days), since they are so involved in their daily work.
To accomplish this, we created a lesson that can be taught in 2 consecutive days or 4 half days.

Demand for our workshops and feedback gathered from students demonstrated the
need for a low-threshold lesson that lets researchers take the first steps in the field of deep learning.
This impression was validated by other instructors who taught the lesson independently to their own audiences and provided us with feedback on their experience.

# Instructional design
This lesson material was designed using the concepts from The Carpentries Curriculum Development Handbook [@becker_carpentries_nodate].
Most importantly, we used 'backward design': we started with identifying learning objectives, the core skills and concepts that learners should acquire as a result of the lesson.
Next, exercises were designed to assess whether these objectives are met.
Eventually, the content is written to teach the skills and concepts learners need to successfully complete the exercises and, it follows, meet the learning objectives.

Live coding is central to this approach: 
the lesson is built up of small blocks. In each block first the instructor demonstrates how to do something,
and students follow along on their own computer. Then, the students work independently on exercises individually
or in groups to test their skills.
This approach integrates opportunities for guided practice throughout the lesson,
promoting learning by 
helping learners build up a functioning mental model of the domain and 
transfer new knowledge from working memory to long-term memory. 
This is in accordance with research-based successful teaching strategies [@lang_small_2021].

The lesson material is built in the new lesson template: Carpentries Workbench [@noauthor_carpentries_nodate]. 
This makes the lesson material a complete self-study resource. 
But it also serves as lesson material for the instructor teaching the lesson through live-coding, 
in that case the lesson material is only shared with students after the workshop as a reference.
The lesson material can be toggled to the 'instructor view'. This allows to provide instructor notes on how to approach teaching the lesson,
and these can even be included at the level of the lesson content.
In addition, the Carpentries Workbench prioritises accessibility of the content, for example by having clearly visible figure captions
and promoting alt-texts for pictures.

The lesson is split into a general introduction, and 3 episodes that cover 3 distinct increasingly more complex deep learning problems.
Each of the deep learning problems is approached using the same 10-step deep learning workflow (https://carpentries-incubator.github.io/deep-learning-intro/1-introduction.html#deep-learning-workflow).
By going through the deep learning cycle three times with different problems, learners become increasingly confident in applying this deep learning workflow to their own projects.

# Feedback
This course was taught 12 times over the course of 3 years, both online and in-person, by the Netherlands eScience Center
(Netherlands, https://www.esciencecenter.nl/) and Helmholz-Zentrum Dresden-Rossendorf (Germany, https://www.hzdr.de/).
Apart from the core group of contributors, the workshop was also taught at 3 independent institutes, namely:
University of Wisconson-Madison (US, https://www.wisc.edu/), University of Auckland (New Zealand, https://www.auckland.ac.nz/), 
and EMBL Heidelberg (Germany, https://www.embl.org/sites/heidelberg/).
In general, adoption of the lesson material by the instructors not involved in the project went well.
The feedback gathered from our own and others' teachings was used to polish the lesson further.

## Student responses
The feedback we gathered from students is in general very positive, 
with some responses from students to the question 'What was your favourite or most useful part of the workshop. Why?' further confirming our statement of need:

> _I enjoyed the live coding and playing with the models to see how it would effect the results. 
> It felt hands on and made it easy for me to understand the concepts._

> _Well-defined steps to be followed in training a model is very useful. Examples we worked on are quite nice._

> _The doing part, that really helps to get the theory into practice._

Below are two tables summarizing results from our post-workshop survey. We use the students' feedback to continuously improve the lesson. 

|                                                                                            |          STRONGLY DISAGREE    |     DISAGREE    |     UNDECIDED    |     AGREE    |     STRONGLY AGREE    |     TOTAL    |     WEIGHTED AVERAGE    |
|--------------------------------------------------------------------------------------------|-------------------------------|-----------------|------------------|--------------|-----------------------|--------------|-------------------------|
| I can immediately apply what I learned at this workshop.                                   | 0                             | 5               | 6                | 19           | 8                     | 38           | 3,8                     |
| The setup and installation instructions for the lesson were complete and easy to follow.   | 0                             | 0               | 4                | 13           | 21                    | 38           | 4,4                     |
| Examples and tasks in the lesson were relevant and authentic                               | 0                             | 0               | 5                | 19           | 14                    | 38           | 4,2                     |

Table 1: Agreement on statements by students from 2 workshops taught at the Netherlands eScience Center. 
The results from these 2 workshops are a good representation of the general feedback we get when teaching this workshop.

|                                                                           |          POOR    |     FAIR    |     GOOD    |     VERY GOOD    |     EXCELLENT    |     N/A    |     TOTAL    |     WEIGHTED AVERAGE    |
|---------------------------------------------------------------------------|------------------|-------------|-------------|------------------|------------------|------------|--------------|-------------------------|
|     Introduction into Deep Learning                                       | 0 (0%)           | 2 (5%)      | 10 (27%)    | 8 (22%)          | 17 (46%)         | 0 (0%)     | 37           | 4,1                     |
|     Classification by a Neural Network using Keras (penguins dataset)     | 0 (0%)           | 1 (3%)      | 5 (13%)     | 16 (42%)         | 16 (42%)         | 0 (0%)     | 38           | 4,2                     |
|     Monitoring and Troubleshooting the learning process (weather dataset) | 0 (0%)           | 0 (0%)      | 4 (11%)     | 18 (47%)         | 16 (42%)         | 0 (0%)     | 38           | 4,3                     |
|     Advanced layer types (CIFAR-10 dataset)                               | 0 (0%)           | 2 (5%)      | 5 (13%)     | 7 (18%)          | 16 (42%)         | 8 (21%)    | 38           | 4,2                     |

Table 2: Quality of the different episodes of the workshop as rated by students from 2 workshops taught at the Netherlands eScience Center. 
The results from these 2 workshops are a good representation of the general feedback we get when teaching this workshop.

# Conclusion
This lesson can be taught as a stand-alone workshop to students already familiar with machine learning and Python.
It can also be taught in a broader curriculum after an introduction to Python programming (for example: @azalee_bostroem_software_2016) 
and an introduction to machine learning (for example: @noauthor_scikit-learn_2023).
Concluding, the described lesson material is a unique and essential resource aimed at researchers and designed specifically for a live-coding teaching style.
Hopefully, it will help many researchers to set their first steps in a successful application of deep learning to their own domain.

# Acknowledgements
We would like to thank all instructors and helpers that taught the course, 
and the community of people that left contributions to the project, no matter how big or small. 
Also, we thank Chris Endemann (University of Wisconson-Madison, US, https://www.wisc.edu/),
Nidhi Gowdra (University of Auckland, New Zealand, https://www.auckland.ac.nz/), 
,Renato Alves and Lisanna Paladin (EMBL Heidelberg, Germany, https://www.embl.org/sites/heidelberg/),
that piloted this workshop at their institutes.
We thank the Carpentries for providing such a great framework for developing this lesson material.
We thank all students enrolled in the workshops that were taught using this lesson material for providing us with feedback.

# References
