# NRC / Amii - Plant Genomic Project (Part 1) :seedling:

This code repository is associated to the paper entitled *"PlanTT: a two-tower contrastive approach for the prediction of gene expression difference in plants"*, which was submitted to RECOMB 24.

## Project Tree :deciduous_tree:
This section gives an overview of the project organization.
```
├── checkpoints                        -> Temporary model checkpoints during training.
├── data                               
│   ├── processed                      -> Training, validation and test data sets ready for the experiments.
│   │   ├── arabadopsis_medicago
│   │   ├── pea_faba_medicago
│   │   │   └── 6-mer_w_cls_sep        -> Tokenized 6-mer sequences with [CLS] and [SEP] tokens. 
│   │   └── sorghum_mays                        
│   ├── interim                        -> Datasets created from raw data but not ready for the experiments.
│   └── raw                            -> Raw data.
├── experiments                        -> Experiment scripts.
├── records                            -> Experiment records.
│   └── paper_results                  -> Experiment results associated to the paper for RECOMB 24.
├── settings                           -> Conda environment settings and files with important directory paths.
├── src                                -> Code modules and functions required to run the experiments.
│   ├── data                           
│   │   ├── modules                    -> Objects dedicated to data storage and preprocessing. 
│   │   └── scripts                    -> Scripts used to create interim and processed data.
│   ├── models                         -> Pytorch models (PlanTT and towers).
│   ├── optimization                   -> Models' training modules.
│   └── utils                          -> Other utilities such as custom metrics and loss functions.
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
git clone -b master --single-branch git@github.com:AmiiThinks/nrc-ml-plant-genomics.git
``` 

#### 4. Create a new conda environment named 'nrc-genomics' with Python 3.10 :snake:
Now that the repostiory is cloned, move to the ```settings``` directory and create a new conda environment
using the provided ```.yml``` file.
```
cd nrc-ml-plant-genomics/settings/
conda env create --file environment.yml
```
Once the environment is created, activate it and return to the root of the project.
```
conda activate nrc-genomics
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

#### 6. Download the data required to run the experiments :package:
Move into the ```data``` folder and download the zip file containing the data from Google Drive.
```
cd data
gdown 1lLB6LxzHM3vMfHFnC0_qNIY3a9qZdR6h
```
Once the file is downloaded, unzip it.
```
unzip data.zip
```
With all the files now available, you can delete the former zip file (optional) and return to the project root.
```
rm data.zip
cd ..
```

#### 7. Celebrate the installation of the environment :partying_face:
You are all set! It is now time to go grab a coffee before running experiments.


## Run experiments :scientist:
All main experiments can currently be executed with the following script:
```
experiments\ocm_evaluation.py
```
Below, we list the possible arguments that can be given to the script and further provide usage examples.

### Arguments

* ```--model (str) - {'plantt-cnn', 'plantt-ddnabert', 'plantt-dnabert', 'plantt-ct', 'pnas-cnn'}```:
  * Model used for the experiment.
* ```--data (str) - {'pfm', 'sb_zm'}```:
  * Data used for the 5-fold cross validation. If "pfm", the Pea, Faba and Medicago data is used. Otherwise, Sorghum bicolor and Zea mays data is used in the same way as in the PNAS paper. Default = ```pfm```.
* ```--head (str) - {'sum', 'mlp'}```:
  * Choice of head architecture for the experiment. Default = ```sum```.
* ```--train_batch_size, -trbs (int)```:
  * Training batch size. Default = ```32```.
* ```--test_batch_size, -tebs (int)```:
  * Test and validation batch size. Default = ```32```.
* ```--lr (float)```:
  * Initial learning rate. Default = ```1e-4```.
* ```--max_epochs, -epochs (int)```:
  * Maximum number of epochs. Default = ```200```.
* ```--patience (int)```:
  * Number of epochs without improvement allowed before stopping the training. Only weights associated to the best validation score are kept following the training. Default = ```20```. 
* ```--milestones (list[int])```:
  * Epochs at which the learning rate is multiplied by a factor of gamma (ex. ```50 60 70```). Default to ```None```. When set to None, the learning rate is multiplied by a factor of ```gamma``` every 15 epochs, starting from epoch 75th.
* ```--gamma (float)```:
  * Constant multiplying the learning rate at each milestone. Default = ```0.75```.
* ```--weight_decay, -wd (float)```:
  * Coefficient associated to the L2 penalty term in the loss function. Default = ```1e-2```.
* ```--dropout, -p (float) - [0, 1)```:
  * Probability of the elements to be zeroed following convolution layers. Applies to ```plantt-cnn``` and ```plantt-ct``` only. Default = ```0.0```.
* ```--head_dropout, -head_p (float) - [0, 1)```:
  * Probability of the elements to be zeroed in the head layers (Applies only if head = ```mlp```). Default = ```0.0```.
* ```--odd_head (bool)```:
  * If provided, the head will be modified to be an odd function (Applies only if head = ```mlp```). Default to ```False```.
* ```--feature_dropout, -p (float) - [0, 1]```:
  * Probability of replacing any nucleotide by X or N ([0 0 0 0 1]). Applies to ```plantt-cnn``` and ```plantt-ct``` only. Default = ```0.0```.
* ```--regression, -reg (bool)```:
  * If provided, model is trained to predict rank differences instead of binary classes. Default =```False```.
* ```--beta (float) -  [0, 1] or -1```:
  * Weight attributed to the scaled MSE in the multitask loss. Weight ```1 - beta``` is given to the BCE. Applies only if ```--regression``` parameter is also given. If ```beta = -1```, the unscaled MSE is used.
* ```--scale (bool)```:
  * If provided, the regression targets are transformed using standard scaling. The mean and std of the training set is employed for each fold. Applies only if ```--regression``` parameter is also given. Default = ```False```.
* ```--include_flip, -inc_flip (bool) ```:
  * If provided, flipped versions of the orthologs pairs are also included in the training sets. Default = ```False```.
* ``` --freeze_method, -fm (str) - {'all', 'keep_last'} or None```:
  * Freeze method used if a pre-trained language model is selected. If ```all```, all layers are frozen. If ```keep_last```, all layers except the last one are frozen. If not provided (```None```), all layers remain unfrozen. Default = ```None```.
* ```--nb_folds (int) - {1, 2, 3, 4, 5}```:
  * Number of cross-validation folds to execute. Default = ```5```.
* ```--training_cat (int) - {1, 2, 3} or None```:
  * If 0, only the observations with rank differences greater or equal to the 3rd quartile in each species pair are kept. If 1, only the observations with rank differences greater or equal than the 2nd quartile in each species pair are kept. If 2, only the observations with rank differences greater or equal than the 1st quartile in each species pair are kept. It not provided, all **training** data points are used. Default = ```None```.
* ```--validation_cat (int) - {1, 2, 3} or None```:
  * If 0, only the observations with rank differences greater or equal to the 3rd quartile in each species pair are kept. If 1, only the observations with rank differences greater or equal than the 2nd quartile in each species pair are kept. If 2, only the observations with rank differences greater or equal than the 1st quartile in each species pair are kept. It not provided, all **validation** data points are used. Default = ```None```.
* ```--test_cat (int) - {1, 2, 3} or None```:
  * If 0, only the observations with rank differences greater or equal to the 3rd quartile in each species pair are kept. If 1, only the observations with rank differences greater or equal than the 2nd quartile in each species pair are kept. If 2, only the observations with rank differences greater or equal than the 1st quartile in each species pair are kept. It not provided, all **test** data points are used. Default = ```None```.
* ```--dnabert_pretraining_folder, -dbp_folder (str)```:
  * Folder in which we can find the file containing weights obtained by pre-training ```plantt-dnabert``` on Arabadopsis and Medicago plant genome. Default = ```None```.
* ```--restart_from (str)```:
  * If the path of a past experiment is provided, pre-trained weights of each fold will be used to initialize PlanTT. Default = ```None```.
* ```--device_id, -dev (int)```:
  * Cuda device ID. Default = ```None```.
* ```--memory_frac, -memory (float) - [0, 1]```:
  * Percentage of device allocated to the experiment. Default = ```1```.
* ```--seed (int)```:
  * Seed value used for experiment reproducibility. Default = ```1```. 

### Usage examples
In this section, we provide arguments that can be copied and pasted in the terminal (from the project root) to run specific experiments
associated to the RECOMB 24 paper. Each of these were previously executed using a single **Nvidia RTX A6000 GPU** with **48G** of memory. The results obtained are shown below each experiment.

#### PNAS CNN :white_check_mark:
```
python experiments/ocm_evaluation.py \
--model pnas-cnn \
--data pfm \
--train_batch_size 32 \
--test_batch_size 500 \
--lr 5e-05 \
--max_epochs 200 \
--patience 20 \
--gamma 0.75 \
--weight_decay 0.01 \
--beta -1.0 \
--nb_folds 5 \
--memory_frac 1 \
--seed 1
```
| Fold | BAccuracy | F1-score | auROC |
|------|-----------|----------|-------|
|    0 |     0.615 |    0.629 | 0.664 |
|    1 |     0.603 |    0.613 | 0.652 |
|    2 |     0.583 |    0.631 | 0.617 |
|    3 |     0.589 |    0.629 | 0.644 |
|    4 |     0.611 |    0.580 | 0.665 |
| mean |     0.600 |    0.616 | 0.648 |
| std  |     0.014 |    0.022 | 0.020 |

#### PlanTT-CNN :white_check_mark:
```
python experiments/ocm_evaluation.py \
--model plantt-cnn \
--head sum \
--data pfm \
--train_batch_size 32 \
--test_batch_size 500 \
--lr 0.0001 \
--max_epochs 200 \
--patience 20 \
--gamma 0.75 \
--weight_decay 0.01 \
--regression \
--beta -1.0 \
--nb_folds 5 \
--memory_frac 1 \
--seed 1
```
| Fold | BAccuracy | F1-score | SpearmanR | R<sup>2</sup> |
|------|-----------|----------|-----------|----------|
|    0 |     0.659 |    0.639 |     0.459 |    0.272 |
|    1 |     0.654 |    0.632 |     0.463 |    0.280 |
|    2 |     0.661 |    0.650 |     0.479 |    0.288 |
|    3 |     0.655 |    0.630 |     0.465 |    0.269 |
|    4 |     0.658 |    0.642 |     0.455 |    0.247 |
| mean |     0.658 |    0.639 |     0.464 |    0.271 |
| std  |     0.003 |    0.008 |     0.009 |    0.016 |

#### PlanTT-DNABERT :white_check_mark:
```
python experiments/ocm_evaluation.py \
--model plantt-ddnabert \
--head sum \
--data pfm \
--train_batch_size 4 \
--test_batch_size 4 \
--lr 5e-05 \
--max_epochs 200 \
--patience 20 \
--gamma 0.75 \
--weight_decay 0.01 \
--dropout 0.0 \
--regression \
--beta -1.0 \
--nb_folds 5 \
--memory_frac 1 \
--seed 1
```
| Fold | BAccuracy | F1-score | SpearmanR | RSquared |
|------|-----------|----------|-----------|----------|
|    0 |     0.613 |    0.584 |     0.330 |    0.101 |
|    1 |     0.618 |    0.594 |     0.356 |    0.091 |
|    2 |     0.645 |    0.625 |     0.399 |    0.145 |
|    3 |     0.597 |    0.571 |     0.304 |    0.109 |
|    4 |     0.606 |    0.582 |     0.311 |    0.071 |
| mean |     0.616 |    0.591 |     0.340 |    0.103 |
| std  |     0.018 |    0.021 |     0.039 |    0.027 |

#### PlanTT-dDNABERT :white_check_mark:
```
python experiments/ocm_evaluation.py \
--model plantt-ddnabert \
--head sum \
--data pfm \
--train_batch_size 16 \
--test_batch_size 16 \
--lr 0.0001 \
--max_epochs 200 \
--patience 20 \
--gamma 0.75 \
--weight_decay 0.01 \
--dropout 0.0 \
--regression \
--beta -1.0 \
--nb_folds 5 \
--memory_frac 1 \
--seed 1
```
| Fold | BAccuracy | F1-score | SpearmanR | RR<sup>2</sup>|
|------|-----------|----------|-----------|----------|
|    0 |     0.672 |    0.646 |     0.466 |    0.208 |
|    1 |     0.650 |    0.627 |     0.449 |    0.211 |
|    2 |     0.642 |    0.612 |     0.420 |    0.201 |
|    3 |     0.648 |    0.628 |     0.407 |    0.148 |
|    4 |     0.655 |    0.633 |     0.426 |    0.177 |
| mean |     0.653 |    0.629 |     0.434 |    0.189 |
| std  |     0.012 |    0.012 |     0.024 |    0.026 |


### Instructions for PlanTT-DNABERT experiment with pre-training :weight_lifting:
In this section we provide:
- instructions to pre-train DNABERT6 masked language model on Arabadopsis and Medicago DNA sequences.
- instructions to train PlanTT-DNABERT with pre-trained DNABERT6 on Pea, Faba and Medicago dataset.

#### DNABERT pre-training
The following instructions run the pre-training of DNABERT6 on Arabadopsis and Medicago DNA sequences.
```
python experiments/plantbert_pretraining.py \
--folder_name dnabert_pretraining \
--batch_size 64 \
--lr 0.0001 \
--max_epochs 100 \
--patience 20 \
--weight_decay 0.01 \
--seed 1
```

#### PlantBERT fine-tuning
Once the pre-training is completed, run the following instructions to train PlanTT-DNABERT on the Pea, Faba and Medicago dataset.
```
python experiments/ocm_evaluation.py \
--model plantt-ddnabert \
--head sum \
--data pfm \
--train_batch_size 4 \
--test_batch_size 4 \
--lr 5e-05 \
--max_epochs 200 \
--patience 20 \
--gamma 0.75 \
--weight_decay 0.01 \
--dropout 0.0 \
--regression \
--beta -1.0 \
--nb_folds 5 \
--memory_frac 1 \
--dnabert_pretraining_folder dnabert_pretraining \
--seed 1
```

## Single-base gene editing
PlanTT can be used to generate a list of single-base modifications that can be applied to a gene to increase its transcript abundance.  
The file ```edit_sequence.py``` offers a program that uses a trained version of ```PlanTT-CNN``` to generate such list for any gene.  
The latter procedure requires the user to provide the promoter and terminator sequence of the gene of interest (see figure below).  

![features](https://github.com/AmiiThinks/nrc-ml-plant-genomics/assets/122919943/579e0429-751f-4320-81fb-d5d8858bd3c5)

To run the program, simply enter the command ```python edit_sequence.py``` and fill the requested information in the terminal.  
An example of output is shown below for an experiment with a budget of ```4``` single-base edits.

<img width="374" alt="program_output" src="https://github.com/AmiiThinks/nrc-ml-plant-genomics/assets/122919943/f287a467-52d2-4472-a1cd-4efdbe51b358">

Each edit proposed is presented in the format ```(old_nucleotide -> proposed_nucleotide, nucleotide_position)```.

