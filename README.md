# PlanTT

This code repository is associated to the paper entitled *"PlanTT: a two-tower contrastive approach for the prediction of gene expression difference in plants"*, which was submitted to RECOMB 24.

<img width="762" alt="PlanTT drawio" src="https://github.com/AmiiThinks/PlanTT/assets/122919943/8cde1d26-4b48-4f9c-9724-0ccb8b07b845">


## Project Tree :deciduous_tree:
This section gives an overview of the project organization.
```
├── checkpoints                        -> Temporary model checkpoints during training.
├── records                            -> Record files generated from training a PlanTT model.
├── settings                           -> Conda environment settings and file with important directory paths.
├── src                                -> Code modules and functions required to run the experiments.
│   ├── data                           
│   │   ├── modules                    -> Objects dedicated to data storage and preprocessing. 
│   │   └── scripts                    -> Scripts used to create processed data.
│   ├── models                         -> Pytorch models (PlanTT and towers).
│   ├── optimization                   -> Training modules.
│   └── utils                          -> Other utilities such as custom metrics and loss functions.
├── models                             -> Weights of trained PlanTT models.
├── train_plantt.py                    -> Script to train a PlanTT model from scratch.
└── edit_sequence.py                   -> Gene editing program (see details further below).
```
Note that each Python file in the GitHub repository is accompanied with a description of its purpose. 

## Environment Setup :wrench:
The following section contains the procedure to correctly setup the environment needed for the project.  
The procedure was tested on a machine with ***Ubuntu 22.04.2 LTS operating system*** and ***Conda 23.3.1 software***.

#### 1. Clone the master branch of the GitHub repository :pencil:
From your terminal, move into the directory in which you would like to store the project and clone it.
```
git clone git@github.com:AmiiThinks/PlanTT.git
``` 

#### 2. Create a new conda environment named 'plantt' :snake:
Now that the repostiory is cloned, move to the ```settings``` directory and create a new conda environment
using the provided ```.yml``` file.
```
cd PlanTT/settings/
conda env create --file environment.yml
```
Once the environment is created, activate it and return to the root of the project.
```
conda activate plantt
cd ..
```

#### 3. Install [PyTorch](https://pytorch.org/get-started/locally/) :fire:
In addition to the packages available in the current environment, it is essential
to install the PyTorch package version specific to your device. To do so, first
open the installation [page](https://pytorch.org/get-started/locally/).

In the opened page, select your preferences. The preferences are as follows:
```
Pytorch build: Stable
Your OS: your_os
Package: Conda
Language: Python
Compute Platform: your_cuda_version
```
Copy the provided command, paste it in the terminal and press enter.

## Training PlanTT from scratch
It is possible to train a PlanTT model with your own data using the following script:
```
train_plantt.py
```
Pickle files containing respectively the training set and the validation set are required. The latter must match the following format:
```
with open(path, 'rb') as file:
        x_a, x_b, y = load(file)
```
where ```x_a``` and ```x_b``` are DNA sequences that are either one-hot encoded ```(N, 5, 3000)``` or separated into 6 chunks of tokenized 6-mers ```(N, 6, 3000)```, and ```y``` is a tensor of shape ```(N,)``` with rank expression differences of mRNA abundance differences. One-hot encoded sequences must respect the following convention:
```
"A": [1, 0, 0, 0, 0]
"C": [0, 1, 0, 0, 0]
"G": [0, 0, 1, 0, 0]
"T": [0, 0, 0, 1, 0]
"N": [0, 0, 0, 0, 1]
"X": [0, 0, 0, 0, 1]
```
All files recorded during the script are saved under the ```records``` directory in a folder named after the training start time.
Below, we list the possible arguments that can be given to the script and further provide a usage example.

### Argument
- ```--tower (str) - {'cnn', 'ddnabert', 'dnabert'}```:
    - Choice of the tower architecture used for the training.
- ```--training_data, -t_data (str)```:
    - Path leading to the pickle file with the training data.
- ```--validation_data, -v_data (str)```:
    - Path leading to the pickle file with the validation data.
- ```--tokens (bool)```:
    - If provided, the data is expected to contain tokenized 6-mers.
- ```--train_batch_size, -tbs (int)```:
    - Training batch size. Default to ```32```.
- ```--valid_batch_size, -vbs (int)```:
    - Validation batch size. Default to ```32```.
- ```--lr (float)```:
    - Initial learning rate. Default to ```5e-5```.
- ```--max_epochs, -max_e (int)```:
    - Maximum number of epochs. Default to ```200```.
- ```--patience (int)```:
    - Number of epochs without improvement allowed before stopping the training. Only the weights associated to the best validation RMSE are kept following the training. Default to ```20```.
- ```--milestones, -ms (list[int])```:
    - Epochs at which the learning rate is multiplied by a factor of gamma (ex. ```50 60 70```). When set to None, the learning rate is multiplied by a factor of gamma every ```15``` epochs, starting from epoch ```75th```. Default to ```None```.
- ```--gamma (float)```:
    - Constant multiplying the learning rate at each milestone. Default to ```0.75```.
- ```--weight_decay, -wd (float)```:
    - Weight decay (L2 penalty coefficient). Default to ```1e-2```.
- ```--dropout, -p (float) - [0, 1) ```:
    - Probability of the elements to be zeroed following convolution layers (Applies to ```plantt-cnn``` only). Default to ```0```.
- ```--freeze_method, -fm (str) - {'all', 'keep_last', 'None'}```:
    - Freeze method used if a pre-trained language model is selected. If ```all```, all layers are frozen. If ```keep_last```, all layers except the last one are frozen. If ```None```, all layers remain unfrozen. Default to ```None```.
- ```--device_id, -dev (int)```:
    - Cuda device ID. If none are provided, the script will use ```cuda``` as device. Otherwise, it will use ```cuda:device_id```. Default to ```None```.
- ```--memory_frac, -memory (float) - (0, 1]```:
    - Percentage of device allocated to the training. Default to ```1```.
- ```--seed (int)```:
    - Seed value used for training reproducibility. Default to ```1```.

### Usage example
Here are the commands that were used to train the ```PlanTT-CNN``` model stored under the ```models``` folder.
```
python train_plantt.py \
--tower cnn \
--training_data path1 \
--validation_data path2 \
--train_batch_size 32 \
--valid_batch_size 500 \
--lr 5e-5 \
--max_epochs 200 \
--patience 20 \
--gamma 0.75 \
--weight_decay 1e-2 \
--dropout 0
```


## Single-base gene editing
PlanTT can be used to generate a list of single-base modifications that can be applied to a gene to increase its transcript abundance. The file ```edit_sequence.py``` offers a program that uses a trained version of ```PlanTT-CNN``` to generate such list for any gene. The latter procedure requires the user to provide the promoter and terminator sequence of the gene of interest (see figure below).  

![features](https://github.com/AmiiThinks/PlanTT/assets/122919943/702f0a1a-706a-4a3f-863b-43507bc5df8d)


To run the program, simply enter the command ```python edit_sequence.py``` and fill the requested information in the terminal. An example of output is shown below for an experiment with a budget of ```4``` single-base edits and randomly generated terminator and promoter sequences.

Here is the terminal output:  

<img width="527" alt="program_output" src="https://github.com/AmiiThinks/PlanTT/assets/122919943/cad65217-d4fd-427b-a466-595688d158f2">


Here is the document ```edit_1.pdf``` mentioned in the last figure: 

<img width="815" alt="program_edit_screenshot" src="https://github.com/AmiiThinks/PlanTT/assets/122919943/96f8d6e8-89c8-4781-a47b-e7a446699d11">


