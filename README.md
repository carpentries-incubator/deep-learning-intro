[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/carpentries-incubator/deep-learning-intro/scaffolds)
[![DOI](https://zenodo.org/badge/163412836.svg)](https://zenodo.org/badge/latestdoi/163412836)
[![The Carpentries Lab Review Status](http://badges.carpentries-lab.org/25_status.svg)](https://github.com/carpentries-lab/reviews/issues/25)

# Introduction to deep learning
This lesson gives an introduction to deep learning.

## Teaching this lesson?
Do you want to teach deep learning? This material is open-source and freely available. 
Are you planning on using our material in your teaching? 
We would love to help you prepare to teach the lesson and receive feedback on how it could be further improved, based on your experience in the workshop.

You can notify us that you plan to teach this lesson by creating an issue in this repository or by sending an email to deep-learning-lesson-dev@esciencecenter.nl. Also, it would great if you can update [this overview of all workshops taught with this lesson material](workshops.md). This helps us show the impact of developing open-source lessons to our funders.

## Lesson Design
The design of this lesson can be found in the [lesson design](https://carpentries-incubator.github.io/deep-learning-intro/design.html)

## Target Audience
The main audience of this carpentry lesson is PhD students that have little to no experience with
deep learning. In addition, we expect them to know basics of statistics and machine learning.

## Lesson development sprints
We regularly host lesson development sprints, in which we work together at the lesson.
The next one is scheduled for the 13th and 14th of January 2025. We kickoff with an online meeting at 10:00 CEST.
If you want to join (you are very welcome to join even if you have never contributed so far) send an email to deep-learning-lesson-dev@esciencecenter.nl .

## Contributing

We welcome all contributions to improve the lesson! Maintainers will do their best to help you
if you have any questions, concerns, or experience any difficulties along the way.

We'd like to ask you to familiarize yourself with our [Contribution Guide](CONTRIBUTING.md) and
have a look at the [more detailed guidelines][lesson-example] on proper formatting, ways to
render the lesson locally, and even how to write new episodes.

Please see the current list of
[issues](https://github.com/carpentries-incubator/deep-learning_intro/issues)
for ideas for contributing to this repository.

Please also familiarize yourself with the [lesson design](https://carpentries-incubator.github.io/deep-learning-intro/design.html)

For making your contribution, we use the GitHub flow, which is nicely explained in the
chapter [Contributing to a Project](http://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
in Pro Git by Scott Chacon.
Look for the tag ![good_first_issue](https://img.shields.io/badge/-good%20first%20issue-gold.svg).
This indicates that the maintainers will welcome a pull request fixing this issue.

## Setup the Workshop Website locally

To build this lesson locally, you should follow the [setup instructions for the
workbench](https://carpentries.github.io/sandpaper-docs/#overview). In short,
make sure you have R, Git, and Pandoc installed, open R and use the following
commands to install/update the packages needed for the infrastructure:

```r
# register the repositories for The Carpentries and CRAN
options(repos = c(
  carpentries = "https://carpentries.r-universe.dev/",
  CRAN = "https://cran.rstudio.com/"
))

# Install the template packages to your R library
install.packages(c("sandpaper", "varnish", "pegboard", "tinkr"))
```

## Rendering the website locally

See the [Carpentries Workbench usage instructions](https://carpentries.github.io/workbench/#usage) on how to render the website locally.

## Maintainer(s)

Current maintainers of this lesson are
* Sven van der Burg (s.vanderburg@esciencecenter.nl)
* Carsten Schnober (c.schnober@esciencecenter.nl)

## Citation and authors

To cite this lesson, please consult with [CITATION.cff](CITATION.cff).
This also holds a list of contributors to the lesson.

[cdh]: https://cdh.carpentries.org
[community-lessons]: https://carpentries.org/community-lessons
[lesson-example]: https://carpentries.github.io/lesson-example
