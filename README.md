# MSE544_Week2_Hands_on

## Hands-on 2: Train a convolutional neural network (CNN) with MARCO data on Hyak

Authors: Ting Cao & [Ziyu Zhang](https://github.com/Ilxxll)

### Table of Content

- [Background & Goals](#background)
- [Train a CNN with MARCO data on Hyak](#train)
  - [Step 1: Get familiar with the convolutional neural network.](#step1)
  - [Step 2: Convert the training part into a python script file.](#step2)
  - [Step 3: Configure Marco1 Environment](#step3)
  - [Step 4: Train the CNN as a batch job on Hyak](#step4)
  - [Step 5: Evaluate the model](#step5)
  - [Challenge: Train the CNN on Hyak interactive node.](#challenge)
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

Please download: `crystal_image_processing.ipynb`, `marcodata.tar.gz`, `marco.py`, `script` from Canvas

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

**Here you need to transferred the evaluate part to a script file `evaluate.py` by yourself.**

### Step 3: Train the CNN on Hyak interactive node.<a name="step3"></a>

a. Upload marcodata.tar.gz, marco.py to Hyak(should work under the directory named in your username under /gscratch/scrubbed on Hyak)

**Scrubbed administrators clean the file that have not been modifiled within 21 days. So if you want to keep your file and result, remember to download them to your local machine in time.**

b. Unzip the marcodata with command: `tar -xf ./marcodata.tar.gz`, you will see afolder named `marcodata`.

c. Get an interactive node on hyak:

You can use command:
- `srun -p compute -A stf --nodes=1 --ntasks-per-node=4 --time=2:00:00 --mem=100G --pty/bin/bash`

The above command will allocate a node from the stf partition.
  
Alternatively, you can use command:
- `srun -p ckpt -A stf --nodes=1 --ntasks-per-node=4 --time=2:00:00 --mem=100G --pty /bin/bash`

This allows you to use idle resources from other groups across the cluster using the checkpoint partition.

Additional resource about node setting: https://hyak.uw.edu/docs/compute/scheduling-jobs


d. Configure the python environment by the following commands:

- `module load foster/python/miniconda/3.8`
  - This is to load the preinstalled anaconda on Hyak.
- `conda create -n marco1 keras tensorflow scikit-learn pandas pillow`
  - This is to create a python environment named marco1 with all needed python packages installed.
- `conda init bash`
  - You will need to initiate conda if this is your first time using it on Hyak.
- `exec bash`
  - Restart bash to enable conda initiation.
  

e. Set the number of threads can be run at same time:
- `export OMP_NUM_THREADS=4` 

f. Activate the environment you created.
- `conda activate marco1`

g. run the python script.
First make sure in your current working directory, there is your marco.py and the folder named marcodata.

Here are 2 options for you to run the python script:
- Option1: `python3 marco.py > output`
  - The output from training process will be saved in the file named `output`.
- Option2: `nohup python3 marco.py > output &`
  - This will run python script in the background, which allow you to work on any other new command while the script is running.
  
h.The CNN model your constructed in marco.py will be saved under the folder `/models` in your current working directory, named `marco.h5`. As the training process going on, you can find the weights of your models saved under the same folder as well. They are named as `marco+number of epoch+validation accuracy+’.hdf5`

### Step 4: Train the CNN as a batch job on Hyak.<a name="step4"></a>

a. Prepare the python code and python environment after logging on to Hyak:

The steps are the same as above **step3.a,b,c**. No need to repeat if you already completed **step3.a,b,c**.

b. Change the parameter for the script:
- Make sure in your current working directory, there is your `marco.py` and the folder named `marcodata`.

- Prepare the slurm script in this current directory (download the script from canvas first). 
  - Replace the ‘chdir=’ in the script with the path of this directory. 
  - You may also need to set the ‘time=’ to a value in according to the number of epochs you specified to run in your marco.py.
  - (Hint: Use `vi` command)

c. Submit with the command:
- `sbatch script`

**Troubleshooting:**

When run it as batch job, if in the slurm output you see error:

Traceback (most recent call last):
File "marco.py", line 1, in <module>
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
ModuleNotFoundError: No module named 'tensorflow'

Try conda deactivate and activate your environment again. Then submit your job.
  
d. Check your submitted job in the queue:
- `squeue -u yourusername`
  - Change yourusername to your own user name

e. You will find the model and weights following same way in **step3.h** . The output information will be saved in a file named `output`.

### Step 5: Evaluate the model.<a name="step5"></a>

The final step is to evaluate the accuracy of the model you obtained from previous training steps. 

The code has been provided in the ‘Model Evaluation’ part in `crystal_image_processing.ipynb`.

a. To run it on Hyak you will need to prepare the code into a python script.

b. Then run it either on interactive node or submit it as batch job. Note that the df_test is the test dataset you prepared earlier along with the train and validation dataset, you should make sure this dataset can be correctly loaded when you evaluate the model.

## Submission. <a name="submission"></a>

Please submit:
1. Your python script 
2. The output file (it should be the output from your python code, not the slurm.out)
