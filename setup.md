---
title: Setup
---
## Software Setup

::::::::::::::::::::::::::::::::::::::: discussion

### Installing Python using Anaconda

[Python][python] is a popular language for scientific computing, and a frequent choice
for machine learning as well. Installing all of its scientific packages
individually can be a bit difficult, however, so we recommend the installer [Anaconda][anaconda]
which includes most (but not all) of the software you will need.

Regardless of how you choose to install it, please make sure you install Python
version 3.x (e.g., 3.4 is fine). Also, please set up your python environment at
least a day in advance of the workshop.  If you encounter problems with the
installation procedure, ask your workshop organizers via e-mail for assistance so
you are ready to go as soon as the workshop begins.

:::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::: solution

### Windows

Checkout the [video tutorial][video-windows] or:

1. Open [https://www.anaconda.com/products/distribution][anaconda-distribution]
with your web browser.
2. Download the Python 3 installer for Windows.
3. Double-click the executable and install Python 3 using _MOST_ of the
   default settings. The only exception is to check the
   **Make Anaconda the default Python** option.

:::::::::::::::::::::::::

:::::::::::::::: solution

### MacOS

Checkout the [video tutorial][video-mac] or:

1. Open [https://www.anaconda.com/products/distribution][anaconda-distribution]
   with your web browser.
2. Download the Python 3 installer for OS X.
3. Install Python 3 using all of the defaults for installation.

:::::::::::::::::::::::::


:::::::::::::::: solution

### Linux

Note that the following installation steps require you to work from the shell.
If you run into any difficulties, please request help before the workshop begins.

1.  Open [https://www.anaconda.com/products/distribution][anaconda-distribution] with your web browser.
2.  Download the Python 3 installer for Linux.
3.  Install Python 3 using all of the defaults for installation.
    a.  Open a terminal window.
    b.  Navigate to the folder where you downloaded the installer
    c.  Type
    ```bash
    bash Anaconda3-
    ```
    and press tab.  The name of the file you just downloaded should appear.
    d.  Press enter.
    e.  Follow the text-only prompts.  When the license agreement appears (a colon
        will be present at the bottom of the screen) hold the down arrow until the
        bottom of the text. Type `yes` and press enter to approve the license. Press
        enter again to approve the default location for the files. Type `yes` and
        press enter to prepend Anaconda to your `PATH` (this makes the Anaconda
        distribution the default Python).

:::::::::::::::::::::::::

## Installing the required packages

[Conda](https://docs.conda.io/projects/conda/en/latest/) is the package management system associated with [Anaconda](https://anaconda.org) and runs on Windows, macOS and Linux.
Conda should already be available in your system once you installed Anaconda successfully. Conda thus works regardless of the operating system.
Make sure you have an up-to-date version of Conda running.
See [these instructions](https://docs.anaconda.com/anaconda/install/update-version/) for updating Conda if required.
{: .callout}

To create a conda environment called `dl_workshop` with the required packages, open a terminal (Mac/Linux) or Anaconda prompt (Windows) and type the command:
```bash
conda create --name dl_workshop python jupyter seaborn scikit-learn pandas
```

Activate the newly created environment:
```
conda activate dl_workshop
```

Install tensorflow using pip (python's package manager):
```bash
pip install tensorflow
```

Note that modern versions of Tensorflow make Keras available as a module.

[pip](https://pip.pypa.io/en/stable/) is the package management system for Python software packages.
It is integrated into your local Python installation and runs regardless of your operating system too.

::::::::::::::::::::::::::::::::::::::: discussion

### Python package installation troubleshooting



:::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::: solution

### Troubleshooting for Windows
It is possible that Windows users will run into version conflicts. If you are on Windows and get
errors running the command, you can try installing the packages using pip within a conda environment:

```bash
conda create -n dl_workshop python jupyter
conda activate dl_workshop
pip install tensorflow>=2.5 seaborn scikit-learn pandas
```

:::::::::::::::::::::::::

::::::::::::::::::: solution

### Troubleshooting for Macs with Apple silicon chip
Newer Macs (from 2020 onwards) often have a different kind of chip, manufactured by Apple instead of Intel.
This can lead to problems installing Tensorflow .
If you get errors running the installation command or conda hangs endlessly,
you can try installing Tensorflow for Mac with pip:

```bash
pip install tensorflow-macos
```

::::::::::::::::::::::::::::

## Starting Jupyter Lab

We will teach using Python in [Jupyter lab][jupyter], a
programming environment that runs in a web browser. Jupyter requires a reasonably
up-to-date browser, preferably a current version of Chrome, Safari, or Firefox
(note that Internet Explorer version 9 and below are *not* supported). If you
installed Python using Anaconda, Jupyter should already be on your system. If
you did not use Anaconda, use the Python package manager pip
(see the [Jupyter website][jupyter-install] for details.)

To start jupyter lab, open a terminal (Mac/Linux) or Anaconda prompt (Windows) and type the command:

```bash
jupyter lab
```

To start the Python interpreter without jupyter lab, open a terminal (Mac/Linux) or Anaconda prompt (Windows)
or git bash and type the command:

```bash
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

Alternatively you can use [Google colab](https://colab.research.google.com/). If you open a jupyter notebook here, the required packages are already pre-installed. Note that google colab uses jupyter notebook instead of jupyter lab.

## Downloading the required datasets

Download the [weather dataset prediction csv][weatherdata] and [BBQ labels][weatherbbqdata].

[anaconda]: https://www.anaconda.com/products/individual
[anaconda-distribution]: https://www.anaconda.com/products/distribution
[jupyter]: http://jupyter.org/
[jupyter-install]: http://jupyter.readthedocs.io/en/latest/install.html#optional-for-experienced-python-developers-installing-jupyter-with-pip
[python]: https://python.org
[video-mac]: https://www.youtube.com/watch?v=TcSAln46u9U
[video-windows]: https://www.youtube.com/watch?v=xxQ0mzZ8UvA
[penguindata]: https://zenodo.org/record/3960218/files/allisonhorst/palmerpenguins-v0.1.0.zip?download=1
[weatherdata]: https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1
[weatherbbqdata]: https://zenodo.org/record/4980359/files/weather_prediction_bbq_labels.csv?download=1


