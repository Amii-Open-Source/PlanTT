#  Description of the contrast models 
Here, we describe the contrast models implemented within the files of this directory.

## Two-Tower contrast model for Plant genes (PlanTT)  
Inspired by Siamese Networks, PlanTT model is defined by the ```PlanTT``` class in ```plantt.py```.  
The architecture of the model follows the illustration below.

<img width="610" alt="plantt_sketch" src="https://github.com/Amii-Open-Source/PlanTT/assets/122919943/d09f8455-b09b-4d14-9520-25b1de638381">


The shared architecture of the towers, and the architecture of the head, are given via the ```tower``` and ```head``` argument of the ```PlanTT``` class respectively.
Any tower or head architecture created must inherit from the ```Tower``` and ```Head``` abstract classes defined in ```plantt.py```. Two head models are already proposed in ```head.py```.


## Washburn CNN
In ```cnn.py```, we provided a PyTorch implementation of the CNN contrast model previously proposed by [Washburn et al.](https://www.pnas.org/doi/10.1073/pnas.1814551116).   
The model is defined by the ```WCNN``` class.

# Description of PlanTT tower architectures
Here we describe the tower architectures implemented for PlanTT.

## CNN tower (for PlanTT-CNN)
The CNN tower is defined by the ```CNN1D``` class in ```cnn.py```.  
A summary of the architecture is presented in the following illustration.

<img width="725" alt="cnn_tower" src="https://github.com/Amii-Open-Source/PlanTT/assets/122919943/80c7d4e1-a02b-4e1a-a706-fe9f2a37bf9b">


## DNABERT-based tower (for PlanTT-DNABERT)
Two different tower architectures integrating [DNABERT](https://pubmed.ncbi.nlm.nih.gov/33538820/), a masked language model (MLM) pre-trained on the human genome, are proposed in ```mlm.py```. Precisely, the two architectures are defined by the ```DNABERT``` and ```DDNABERT``` classes. Each of them follows the figure below, but distinguishes itself by the version of DNABERT it uses.

<img width="838" alt="dnabert_tower" src="https://github.com/Amii-Open-Source/PlanTT/assets/122919943/b419a910-ede9-48c1-b955-1d931ce0c144">


- ```DNABERT``` integrates the [plain version](https://huggingface.co/zhihan1996/DNA_bert_6) of DNABERT6.

- ```DDNABERT``` integrates a [distilled version](https://huggingface.co/Peltarion/dnabert-minilm-small) of DNABERT6.

For ```DDNABERT```, embeddings shown in the figure above have size 384 instead of 768.




