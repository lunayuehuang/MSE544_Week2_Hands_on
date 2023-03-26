# MSE544_Week2_Hands_on

## Hands-on 2: Train a convolutional neural network (CNN) with MARCO data on Hyak

Authors: Ting Cao & [Ziyu Zhang](https://github.com/Ilxxll)

### Table of Content

- [Background & Goals](#background)
- [Train a CNN with MARCO data on Hyak](#train)
  - [Step 1: Get familiar with the convolutional neural network.](#step1)
  - [Step 2: Convert the training part into a python script file.](#step2)
  - [Step 3: Train the CNN on Hyak interactive node.](#step3)
  - [Step 4: Train the CNN as a batch job on Hyak](#step4)
  - [Step 5: Evaluate the model](#step5)
- [Submission](#submission)

## Background & Goals for this week's hands-on <a name="background"></a>

#### Convolutional Neural Network (CNN) & MARCO

CNN (Convolutional Neural Network) is a deep learning algorithm commonly used for image and video recognition tasks, that uses convolutional layers to automatically learn and extract features from input data.

MARCO is a dataset of protein crystal images, with four categories: Clear, Crystals, Other,
Precipitate. In this session we will use only a subset of it, with 20000 images (each
category has 5000 images).

You can learn what is CNN and MARCO from the fall quarter lecture in the [canvas page](https://canvas.uw.edu/courses/1631767/pages/week-2-using-hpc-and-github).

https://canvas.uw.edu/courses/1631767/pages/week-2-using-hpc-and-github

Additional resource: 

CNN: https://en.wikipedia.org/wiki/Convolutional_neural_network

MARCO: https://marco.ccr.buffalo.edu/about

#### Goals for this week's Hands-on

1. Train a convolutional neural network on Hyak platform, that can classify the images
from MARCO data into four categories: `Clear`, `Crystals`, `Other`, `Precipitate`.

2. Estimate the prediction accuracy of the trained model on a test dataset from MARCO.

## Train a CNN with MARCO data on Hyak <a name="train"></a>

### Before we start:

Please download: `crystal_image_processing.ipynb`, `marcodata.tar.gz`, `marco.py`, `script` from
Canvas

### Step 1: Get familiar with the convolutional neural network..<a name="step1"></a>

The python code is given in the jupyter notebook: `crystal_image_processing.ipynb`. Download it and read the code in it to get familiar with the code.

Download the compressed MARCO dataset `marcodata.tar.gz` from canvas, unzip it. You will see a folder named `marcodata`,with each subfolder under it contains the images for each category.

#### Optional：

You can run the notebook locally to get more familiar with the code. **The training process may take very long time if you run it on your local computer.**

##### Note for setting up python environment on your local machine to run the notebook:

1. Make sure your have `anacond` or `miniconda` installed first.
https://conda.io/projects/conda/en/latest/user-guide/install/index.html. 

2. Then create a virtual environment by command:

`conda create -n envname keras tensorflow scikit-learn pandas pillow scikit-image`

`conda install -c conda-forge scikit-image`

This is to create an environment named ‘envname’ with all the specified python packages installed. Activate the environment by command:  `conda activate envname.` Then you should be ready to work with your notebook in this environment.

### Step 2: Convert the training part into a python script file.<a name="step2"></a>

To train the CNN on Hyak, you will need to copy the code from the jupyter notebook and put them into a python script file. The code up to the training part has been transferred to the python script file `marco.py` for you.

