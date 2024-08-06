---
title: Setup
---
## Software Setup

::::::::::::::::::::::::::::::::::::::: discussion

### Installing Python

[Python][python] is a popular language for scientific computing, and a frequent choice
for machine learning as well.
To install Python, follow the [Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide/Download) or head straight to the [download page](https://www.python.org/downloads/).

Please set up your python environment at least a day in advance of the workshop.
If you encounter problems with the installation procedure, ask your workshop organizers via e-mail for assistance so
you are ready to go as soon as the workshop begins.

:::::::::::::::::::::::::::::::::::::::::::::::::::

## Installing the required packages{#packages}

[Pip](https://pip.pypa.io/en/stable/) is the package management system built into Python.
Pip should be available in your system once you installed Python successfully.

Open a terminal (Mac/Linux) or Command Prompt (Windows) and run the following commands.

1. Create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments) called `dl_workshop`:

::: spoiler

### On Linux/macOs

```shell
python3 -m venv dl_workshop
```

:::

::: spoiler

### On Windows

```shell
py -m venv dl_workshop
```

:::

2. Activate the newly created virtual environment:

::: spoiler

### On Linux/macOs

```shell
source dl_workshop/bin/activate
```

:::

::: spoiler

### On Windows

```shell
dl_workshop\Scripts\activate
```

:::

Remember that you need to activate your environment every time you restart your terminal!

3. Install the required packages:

::: spoiler

### On Linux/macOs

```shell
python3 -m pip install jupyter seaborn scikit-learn pandas tensorflow
```

::: spoiler

### Advanced: TensorFlow with support for Mac M1/M2/M3

Recent Macs have special chips (M1/M2/M3) that can accelerate deep learning processes.
Apple has developed the [tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/) package to support these chips in TensorFlow.
This is not supported by the standard TensorFlow installation, and not required for this lesson.

Nevertheless, you can install the on top of the standard `tensorflow`:

```shell
python -m pip install tensorflow-metal
```

:::
:::

::: spoiler

### On Windows

```shell
py -m pip install jupyter seaborn scikit-learn pandas tensorflow
```

:::

Note: Tensorflow makes Keras available as a module too.

## Starting Jupyter Lab

We will teach using Python in [Jupyter Lab][jupyter], a programming environment that runs in a web browser.
Jupyter Lab is compatible with Firefox, Chrome, Safari and Chromium-based browsers.
Note that Internet Explorer and Edge are *not* supported.
See the [Jupyter Lab documentation](https://jupyterlab.readthedocs.io/en/latest/getting_started/accessibility.html#compatibility-with-browsers-and-assistive-technology) for an up-to-date list of supported browsers.

To start Jupyter Lab, open a terminal (Mac/Linux) or Command Prompt (Windows) and type the command:

```shell
jupyter lab
```

To start the Python interpreter without Jupyter Lab, open a terminal (Mac/Linux) or Command Prompt (Windows)
and type the command:

```shell
python
```

## Check your setup
To check whether all packages installed correctly, start a jupyter notebook in jupyter lab as
explained above. Run the following lines of code:
```python
import sklearn
print('sklearn version: ', sklearn.__version__)

import seaborn
print('seaborn version: ', seaborn.__version__)

import pandas
print('pandas version: ', pandas.__version__)

import tensorflow
print('Tensorflow version: ', tensorflow.__version__)
```

This should output the versions of all required packages without giving errors.
Most versions will work fine with this lesson, but:
- For Keras and Tensorflow, the minimum version is 2.12.0
- For sklearn, the minimum version is 1.2.2

## Fallback option: cloud environment
If a local installation does not work for you, it is also possible to run this lesson in [Binder Hub](https://mybinder.org/v2/gh/carpentries-incubator/deep-learning-intro/scaffolds). This should give you an environment with all the required software and data to run this lesson, nothing which is saved will be stored, please copy any files you want to keep. Note that if you are the first person to launch this in the last few days it can take several minutes to startup. The second person who loads it should find it loads in under a minute. Instructors who intend to use this option should start it themselves shortly before the workshop begins.

Alternatively you can use [Google colab](https://colab.research.google.com/). If you open a jupyter notebook here, the required packages are already pre-installed. Note that google colab uses jupyter notebook instead of Jupyter Lab.

## Downloading the required datasets

Download the [weather dataset prediction csv][weatherdata] and [Dollar street dataset (4 files in total)][dollar-street]

[dollar-street]: https://zenodo.org/api/records/10970014/files-archive
[jupyter]: http://jupyter.org/
[jupyter-install]: http://jupyter.readthedocs.io/en/latest/install.html#optional-for-experienced-python-developers-installing-jupyter-with-pip
[python]: https://python.org
[weatherdata]: https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1
