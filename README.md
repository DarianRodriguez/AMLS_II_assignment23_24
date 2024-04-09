# README

# ELEC0134: Applied Machine Learning Systems II Project

## Project Description

![Samples Image](/figures/samples.png)


The Kaggle Cassava Leaf Disease Classification competition is a machine learning competition hosted on Kaggle, a platform for data science competitions. The goal of this competition is to develop machine learning models that can accurately classify images of cassava leaves into one of several disease categories or a healthy category.

Cassava is a major staple crop in many countries, particularly in Africa, and is an important source of food and income for millions of people. However, cassava plants are susceptible to various diseases, which can significantly reduce yields and threaten food security. Early and accurate detection of these diseases is crucial for effective disease management and crop protection.

The dataset contains 21,367 labeled images with the following Labels Mapping:

- **"0"**: Cassava Bacterial Blight (CBB)
- **"1"**: Cassava Brown Streak Disease (CBSD)
- **"2"**: Cassava Green Mottle (CGM)
- **"3"**: Cassava Mosaic Disease (CMD)
- **"4"**: Healthy

The dataset used in this project can be found on Kaggle: Cassava Leaf Disease Classification Dataset: https://www.kaggle.com/competitions/cassava-leaf-disease-classification

## Project Organization

The main.py calls the modules associated with the task


### Folder Structure 

* unet/: folder with generic U-Net implementation used for training, is extracted from https://github.com/jakeret/unet. 

* best_model.h5: Trained U-Net model

* helper_functions.py: contains all the functions to execute the main modules of the task: Preprocessing and data loader.

* models.py:  functions and classes associated with training the EfficientNetB3 model: focal loss, class for training, compute weights, etc.

* "Figures" folder: folder to save all the plots generated while executing the task.

### Other Files

* requirements.txt: list the packages that needs to be install for the environment.


## Setup

Create the environment and install the required libraries:

* numpy
* jupyter
* jupyterlab
* pandas
* matplotlib
* scikit-learn
* seaborn
* tensorflow
* keras
* opencv-python
* unet

Make sure to install file in unet folder: run the command pip install unet. If you want to run specific experiments on the model comment the necessary lines from model.py, by default the U-Net model will run as preprocessing step.