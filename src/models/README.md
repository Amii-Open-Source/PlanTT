#  Description of ortholog contrast the models 
Here, we describe the ortholog contrast models implemented within the files of this directory.

## Two-Tower Orthologs Contrast for Plants (PlanTT)  
Inspired by Siamese Networks, PlanTT model is defined by the ```PlanTT``` class in ```plantt.py```.  
The architecture of the model follows the illustration below.

![PlanTT](https://github.com/AmiiThinks/nrc-ml-plant-genomics/assets/122919943/03937f79-68c5-471b-beef-bee916124780)

The shared architecture of the towers, and the architecture of the head, are given via the ```tower``` and ```head``` argument of the ```PlanTT``` class respectively.
Any tower or head architecture created must inherit from the ```Tower``` and ```Head``` abstract classes defined in ```plantt.py```. Two head models are already proposed in ```head.py```.


## PNAS CNN
In ```cnn.py```, we provided a PyTorch implementation of the CNN contrast model previously proposed by [Washburn et al.](https://www.pnas.org/doi/10.1073/pnas.1814551116).   
The model is defined by the ```PNASCNN``` class.

# Description of PlanTT tower architectures
Here we describe the tower architectures implemented for PlanTT.

## CNN tower (for PlanTT-CNN)
Inspired by the work of [Washburn et al.](https://www.pnas.org/doi/10.1073/pnas.1814551116), the CNN tower is defined by the ```CNN``` class in ```cnn.py```.  
A summary of the architecture is presented in the following illustration.

![cnn_tower](https://github.com/AmiiThinks/nrc-ml-plant-genomics/assets/122919943/3a89a174-6d10-4520-aaa0-b9bb0cb257ec)


## DNABERT-based tower (for PlanTT-DNABERT)
Two different tower architectures integrating [DNABERT](https://pubmed.ncbi.nlm.nih.gov/33538820/), a masked language model (MLM) pre-trained on the human genome, are proposed in ```mlm.py```. Precisely, the two architectures are defined by ```DNABERT``` and ```dDNABERT``` classes.   
Each of them follows the figure below, but distinguishes itself by the version of DNABERT it uses.

![dnabert_tower](https://github.com/AmiiThinks/nrc-ml-plant-genomics/assets/122919943/776895f5-d3c5-4c04-8688-ee4a03a0c315)

- ```DNABERT``` integrates the [plain version](https://huggingface.co/zhihan1996/DNA_bert_6) of DNABERT6.

- ```dDNABERT``` integrates a [distilled version](https://huggingface.co/Peltarion/dnabert-minilm-small) of DNABERT6. For this particular model, embeddings shown in the figure above have size 384 instead of 768.




