# PlanTT

This code repository is associated to the paper entitled *"PlanTT: a two-tower contrastive approach for the prediction of gene expression difference in plants"*, which was submitted to RECOMB 24.

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><span property="dct:title">PlanTT</span> is licensed under <a href="http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"></a></p>

<img width="1038" alt="PlanTT" src="https://github.com/Amii-Open-Source/PlanTT/assets/122919943/b61d08f8-df94-44fc-aae4-57cd57759663">


## Project Tree :deciduous_tree:
This section gives an overview of the project organization.
```
├── checkpoints                        -> Temporary model checkpoints during training.
├── data                               -> Data samples provided to try train_plantt.py
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
├── process_samples.py                 -> Script to process the raw data samples provided.
└── edit_sequence.py                   -> Gene editing program (see details further below).
```
Note that each Python file in the GitHub repository is accompanied with a description of its purpose. 

## Environment Setup :wrench:
The following section contains the procedure to setup the environment required for the project.  
The procedure was tested on a machine with the ***Ubuntu 22.04.2 LTS operating system***, the ***Conda 23.3.1 software***, and a single Nvidia RTX A6000 GPU.

#### 1. Clone the master branch of the GitHub repository :pencil:
From your terminal, move into the directory in which you would like to store the project and clone it.
```
git clone git@github.com:AmiiThinks/PlanTT.git
``` 

#### 2. Create a new conda environment named 'plantt' :seedling::
Now that the repostiory is cloned, move to the ```settings``` directory and create a new conda environment
using the provided ```.yml``` file.
```
cd PlanTT/settings/
```
```
conda env create --file environment.yml
```
Once the environment is created, activate it and return to the root of the project.
```
conda activate plantt
```
```
cd ..
```

#### 3. Install [PyTorch](https://pytorch.org/get-started/locally/) :fire:
In addition to the packages available in the current environment, it is essential
to install the PyTorch version specific to your device. To do so, first
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


#### 4. Process the raw data samples provided (optional) :dna:
In the ```data``` folder of the repository, we provided raw data samples into ```csv``` files. The latter can be processed to further try ```train_plantt.py``` script, which purpose is detailed below. The training and test sets contain ```2560``` and ```960``` pairs of synthetic dna sequences respectively. Sequences were generated randomly and their associated targets were sampled from a normal distribution with mean ```0``` and standard deviation ```500```. To process the raw data, simply run the following command:
```
python process_samples.py
```

## Training PlanTT from scratch (GPU required)
It is possible to train a PlanTT model with your own data using the following command:
```
python train_plantt.py
```
**Along with a GPU**, pickle files containing respectively the training set and the validation set are required. The latter must match the following format:
```
with open(path, 'rb') as file:
        x_a, x_b, y = load(file)
```
where ```x_a``` and ```x_b``` are DNA sequences that are either one-hot encoded ```(N, 5, 3000)``` or separated into 6 chunks of tokenized 6-mers ```(N, 6, 512)```, and ```y``` is a tensor of shape ```(N,)``` with expression differences between sequences contained in ```x_a``` and ```x_b```. One-hot encoded sequences must respect the following convention:
```
"A": [1, 0, 0, 0, 0]
"C": [0, 1, 0, 0, 0]
"G": [0, 0, 1, 0, 0]
"T": [0, 0, 0, 1, 0]
"N": [0, 0, 0, 0, 1]
"X": [0, 0, 0, 0, 1]
```
All files recorded during the script execution are saved under the ```records``` directory in a folder named after the training start time.
Below, we list the possible arguments that can be given to the script and further provide usage examples.

### Arguments
- ```--tower (str) - {'cnn', 'ddnabert', 'dnabert'}```:
    - Choice of the tower architecture used for the training. See ```src/models/README.md``` for more details.
- ```--training_data, -t_data (str)```:
    - Path leading to the pickle file with the training data.
- ```--valid_data, -v_data (str)```:
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
    - Number of epochs without improvement allowed before stopping the training. Only the weights associated to the best validation RMSE are kept at the end of the training. Default to ```20```.
- ```--milestones, -ms (list[int])```:
    - Epochs after which the learning rate is multiplied by a factor of ```gamma``` (ex. ```50 60 70```). When set to None, the learning rate is multiplied by a factor of ```gamma``` every ```15``` epochs, starting from epoch ```75th```. Default to ```None```.
- ```--gamma (float)```:
    - Constant multiplying the learning rate at each milestone. Default to ```0.75```.
- ```--weight_decay, -wd (float)```:
    - Weight decay (L2 penalty coefficient). Default to ```1e-2```.
- ```--dropout, -p (float) - [0, 1) ```:
    - Probability of the elements to be zeroed following convolution layers (Applies to ```plantt-cnn``` only). Default to ```0```.
- ```--freeze_method, -fm (str) - {'all', 'keep_last', 'None'}```:
    - Freeze method used if a pre-trained language model is selected. If ```all```, all layers are frozen. If ```keep_last```, all layers except the last one are frozen. If ```None```, all layers remain unfrozen. Default to ```None```.
- ```--device_id, -dev (int)```:
    - Cuda device ID. If none are provided, the script will use ```cuda:0``` as device. Otherwise, it will use ```cuda:device_id```. Default to ```None```.
- ```--memory_frac, -memory (float) - (0, 1]```:
    - Percentage of device allocated to the training. Default to ```1```.
- ```--seed (int)```:
    - Seed value used for training reproducibility. Default to ```1```.

### Usage examples
The following usage examples require to run ```train_plantt.py``` beforehand, as written in step 4 of the ***Environment Setup*** section.
Here we provide two examples, one with ```PlanTT-CNN``` and the other with ```PlanTT-DDNABERT```.
```
python train_plantt.py \
--tower cnn \
--training_data data/encoded_training_samples.pkl \
--valid_data data/encoded_validation_samples.pkl \
--train_batch_size 32 \
--valid_batch_size 32 \
--lr 5e-5 \
--max_epochs 20 \
--patience 15 \
--milestones 8 10 12 \
--gamma 0.75 \
--weight_decay 1e-2 \
--dropout 0 \
--memory_frac 1 
```

```
python train_plantt.py \
--tower ddnabert \
--training_data data/tokenized_training_samples.pkl \
--valid_data data/tokenized_validation_samples.pkl \
--tokens \
--train_batch_size 16 \
--valid_batch_size 16 \
--lr 5e-5 \
--max_epochs 20 \
--patience 15 \
--milestones 8 10 12 \
--gamma 0.75 \
--weight_decay 1e-2 \
--freeze_method keep_last \
--memory_frac 1 
```


## Single-base gene editing
PlanTT can be used to generate a list of single-base modifications that can be applied to a gene to increase its expression. The file ```edit_sequence.py``` offers a program that uses a trained version of ```PlanTT-CNN``` to generate such list for any gene. The weights of the model are stored in ```models/planttcnn.pt```.   

The gene editing procedure requires the user to provide the promoter and terminator sequences of the gene of interest (see figure below). 

<img width="720" alt="features" src="https://github.com/Amii-Open-Source/PlanTT/assets/122919943/19d95a3d-7492-48a3-838f-f09afb763028">


To run the program, simply enter the command ```python edit_sequence.py``` and fill the requested information in the terminal. An example of output is shown below for an experiment with a budget of ```4``` single-base edits, a batch size of ```1000```, and a randomly generated sequence used as the promoter and the terminator (see ```data/example_seq.txt```).

Here is the terminal output:  

<img width="525" alt="output" src="https://github.com/Amii-Open-Source/PlanTT/assets/122919943/019596ba-fbfb-4682-9dce-025576cd63c3">


Here is the document ```edit_2.pdf``` mentioned in the last figure: 


<img width="864" alt="edit_2" src="https://github.com/Amii-Open-Source/PlanTT/assets/122919943/5a419664-e12e-4a74-a4de-0a50b17ffedf">

