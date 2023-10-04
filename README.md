# PlanTT

This code repository is associated to the paper entitled *"PlanTT: a two-tower contrastive approach for the prediction of gene expression difference in plants"*, which was submitted to RECOMB 24.

![PlanTT](https://github.com/AmiiThinks/PlanTT/assets/122919943/374f0548-d9a3-4091-84d9-4baf368ec5b1)


## Project Tree :deciduous_tree:
This section gives an overview of the project organization.
```
├── checkpoints                        -> Temporary model checkpoints during training.
├── settings                           -> Conda environment settings and file with important directory paths.
├── src                                -> Code modules and functions required to run the experiments.
│   ├── data                           
│   │   ├── modules                    -> Objects dedicated to data storage and preprocessing. 
│   │   └── scripts                    -> Scripts used to create processed data.
│   ├── models                         -> Pytorch models (PlanTT and towers).
│   ├── optimization                   -> Training modules.
│   └── utils                          -> Other utilities such as custom metrics and loss functions.
├── trained_models                     -> Weights and other files associated to trained PlanTT models.
└── edit_sequence.py                   -> Gene editing program (see details further below).
```
Note that each Python file in the GitHub repository is accompanied with a description of its purpose. 

## Environment Setup :wrench:
The following section contains the procedure to correctly setup the environment needed for the project.  
The procedure was tested on a machine with ***Ubuntu 22.04.2 LTS operating system*** and ***Conda 23.3.1 software***.

#### 1. Open a terminal on your device :desktop_computer:

#### 2. Download Anaconda :arrow_down:
If you already have Anaconda on your device, you can skip that step.  
Otherwise, head to the download [page](https://www.anaconda.com/download/) of Anaconda on a web browser.  
Scroll down to the bottom of the page and copy the link of the appropriate installer for your device.  
Then, head back to you terminal and type the following line using the copied link:
```
wget copied_link
```
A ```.sh``` file starting with ```Anaconda``` should now be available in your current emplacement.
Using that file, which we refer to as ```anaconda_file.sh```, you can now proceed to the installation with the next command:
```
bash anaconda_file.sh
```
Once completed, refresh your terminal window as follow:
```
source ~/.bashrc
```
You should now see ```(base)``` appearing at the beginning of the line in your terminal.

#### 3. Clone the master branch of the GitHub repository :pencil:
From you terminal, move into the directory in which you would like to store the project and clone the project.
```
cd path_to_directory
git clone git@github.com:AmiiThinks/PlanTT.git
``` 

#### 4. Create a new conda environment named 'plantt' with Python 3.10 :snake:
Now that the repostiory is cloned, move to the ```settings``` directory and create a new conda environment
using the provided ```.yml``` file.
```
cd plantt/settings/
conda env create --file environment.yml
```
Once the environment is created, activate it and return to the root of the project.
```
conda activate plantt
cd ..
```

#### 5. Install [PyTorch](https://pytorch.org/get-started/locally/) :fire:
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
Copy the provided command, paste it in the terminal and press enter. It might take few minutes for the installation, don't worry!

#### 7. Celebrate the installation of the environment :partying_face:
You are all set! It is now time to go grab a coffee before running experiments.

## Single-base gene editing
PlanTT can be used to generate a list of single-base modifications that can be applied to a gene to increase its transcript abundance.  
The file ```edit_sequence.py``` offers a program that uses a trained version of ```PlanTT-CNN``` to generate such list for any gene.  
The latter procedure requires the user to provide the promoter and terminator sequence of the gene of interest (see figure below).  

![features](https://github.com/AmiiThinks/PlanTT/assets/122919943/702f0a1a-706a-4a3f-863b-43507bc5df8d)


To run the program, simply enter the command ```python edit_sequence.py``` and fill the requested information in the terminal.  
An example of output is shown below for an experiment with a budget of ```4``` single-base edits.

<img width="374" alt="program_output" src="https://github.com/AmiiThinks/nrc-ml-plant-genomics/assets/122919943/f287a467-52d2-4472-a1cd-4efdbe51b358">

Each edit proposed is presented in the format:   
```(old_nucleotide -> proposed_nucleotide, nucleotide_position)```.

