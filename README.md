# CasSampling
![](https://img.shields.io/badge/ECML--PKDD-2023-BLUE)
![](https://img.shields.io/badge/Python-3.9-blue)
![](https://img.shields.io/badge/torch-1.11.0-blue)

<img src="https://raw.githubusercontent.com/MartinHeinz/MartinHeinz/master/wave.gif" width="30px">This repo provides a  implementation of **CasSampling** as described in the paper:

> CasSampling: Exploring Efficient Cascade Graph Learning for Popularity Prediction

## Basic Usage

### Requirements

The code was tested with `python 3.9`, `torch 1.11.0`, `cudatoolkit 11.3`, and `cudnn 8.2.0`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name CasSampling python=3.9 

# activate environment
conda activate CasSampling

# install torch and other requirements
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia
pip install -r requirements.txt
```


### Run the code
```shell
cd ./preprocessing

##Preprocessing the data, Then transform the datasets to the format of ".pkl" command:
python utils.py
python preprocess_graph.py
#you can change the dataset, observation time, and parameter in config.py

# run CasSampling model
cd ./CasSampling_model
python run_CasSampling.py
```

## Datasets

See some sample cascades in `./data/twitter/dataset.txt`.

Weibo or Twitter Datasets download link: [Google Drive](https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view?usp=sharing) 

The datasets we used in the paper are come from:
- [Twitter](http://carl.cs.indiana.edu/data/#virality2013) (Weng *et al.*, [Virality Prediction and Community Structure in Social Network](https://www.nature.com/articles/srep02522), Scientific Report, 2013).You can also download Twitter dataset [here](https://github.com/Xovee/casflow) in here.
- [Weibo](https://github.com/CaoQi92/DeepHawkes) (Cao *et al.*, [DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades](https://dl.acm.org/doi/10.1145/3132847.3132973), CIKM, 2017). You can also download Weibo dataset [here](https://github.com/CaoQi92/DeepHawkes) in here.  
