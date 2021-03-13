# News_Recommandation-project-team27  
> This is the final project for ECE143 Group 27 in Winter 2021.  
> Team member: Qiyao Wu, Jiashun Wang, Rongxiang Zhang, Rasya Soeroso  
## Environment
### Training environment  
> ubuntu 16.04 with a NVIDIA GPU (12G)  
pytorch==1.5.0  
### Testing environment
> ubuntu 16.04 / windows 10
## Installation  
In order to set up the necessary environment:  
1. create a conda virtual environment with the help of [conda],  
``` bash
conda env create -n news
```  
2. activate the environment  
``` bash
conda activate news
```  
3. install the requirement  
``` bash
pip install -r requirement.txt
```  
4. install pytorch for the local machine, you should choose the suitable version according to your own machine  
> For more information, please check https://pytorch.org/  
5. create folders  
``` bash
cd NRMS
mkdir checkpoints  
mkdir data  
```  
6. download data and checkpoints to the corresponding folder  
> MIND data and glove embeddings we use in this project can be downloaded from here: https://drive.google.com/drive/folders/1BUk_CmsdBl2F7ihswvxDRcEUtSGFiSzf?usp=sharing  
All the checkpoints can be downloaded from here: https://drive.google.com/drive/folders/1lFQnAdY7gw3_6BLkrI1V8qQTvgSJKgvQ?usp=sharing
## Project structure
> + NRMS  
>   + checkpoints  
>     + DeepCross  
>     + NRMS  
>   + data  
>     + cf  
>     + glove  
>     + train  
>     + val  
>   + dataset  
>     + NewsDataset.py  
>     + NewsDatasetAll.py  
>     + NewsDatasetCategory.py  
>     + NewsDatasetDCN.py  
>   + models  
>     + attention.py  
>     + deep_cross.py  
>     + encoder.py  
>     + network.py  
>   + utils  
>     + cf.py  
>     + utils.py  
> 	+ news_recommandation.ipynb  
> + config.py    
> + glove.py  
> + main.py  
> + main_all.py    
> + main_all_resume.py  
> + main_category.py  
> + main_naive_deep_cross.py  
> + test.py  
> + train.py  
> + train_category.py  
> + train_deep_cross.py  
> + train_naive_deep_cross.py  
## File brief description
+ **checkpoints**: directory to save checkpoints while training and load the corresponding checkpoint configured in config.py while performing online inference  
+ **data**: directory to save data  
+ **dataset**: dataloaders for pytorch and MIND dataset, mainly dealing with data preprocessing  
+ **models**: Deep & cross, NRMS model implementation  
+ **utils**: some utility functions and basic collaborative filtering implementation  
+ **config.py**: model hyperparameter configuration  
+ **glove.py**: download and process word embeddings using Glove  
+ **train.py**: pytorch lightning file for training configuration, the loaded model is the basic NRMS  
+ **train_category.py**: pytorch lightning file for training configuration, the loaded model is the NRMS combined with abstract semantic features and attentive category features    
+ **train_deep_cross.py**: pytorch lightning file for training configuration, the loaded model is the NRMS combined with abstract semantic features  
+ **train_naive_deep_cross.py**: pytorch lightning file for training configuration, the loaded model is the basic Deep & cross model  
+ **main.py**: entry file for training model, the loaded model is the basic NRMS  
+ **main_category.py**: entry file for training model, the loaded model is the NRMS combined with abstract semantic features and attentive category features    
+ **main_all.py**: entry file for training model, the loaded model is the NRMS combined with abstract semantic features  
+ **main_naive_deep_cross.py**: entry file for training model, the loaded model is the basic Deep & cross model  
##Usage
+ For training the model, executing the following instructions (e.g. `main_category.py`). **Please make sure that your machine has a GPU, or you can change the configuration file.**  
``` bash
python [entry file]
```  
+ For online inference, you can run the notebook `news_recommandation.ipynb`. Details, thoughts, results and analysis are presented in the notebook as well.  

## Key Idea  
> We implement an additional attention trick and combine more features such as discrete categoty features and continous abstract embeddings in the model.  
> The idea of integrating those features into NRMS is derived from Deep & Cross and TF-IDF.  
> The intuition is that the raw data has bias, and the original NRMS model does not utilize the abundant category feature and abstract features mined in MIND dataset.
## Reference  
NRMS implementation (Wu et al., 2019): - Neural News Recommendation with Multi-Head Self-Attention.  
> If you have trouble in running the notebook or training file, feel free to email me: q6wu@ucsd.edu  
