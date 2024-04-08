# Import Modules
from helper_functions import TFRecordDatasetLoader, Preprocessing
from models import plot_accuracy_epochs, ModelTrainer, prediction,misclassified_samples
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math

# =====================================================================================================================
# Load dataset

# Define dataset paths
TRAIN_TFREC_DIR = 'Datasets/train_tfrecords'
LABELS_MAP_PATH = 'Datasets/label_num_to_disease_map.json'
TRAIN_LABELS_PATH = 'Datasets/train.csv'

# Create instance of TFRecordDatasetLoader class
data_loader = TFRecordDatasetLoader(TRAIN_TFREC_DIR, LABELS_MAP_PATH, TRAIN_LABELS_PATH)

dataset = data_loader.load_tfrecord_dataset()

# Plot image distribution
data_loader.plot_image_distribution('class distribution')

# show image sample per class
data_loader.plot_images_per_class('class samples')

# get labels with class names abbreviated
abbrev_labels = data_loader.load_label_dictionary()


# ======================================================================================================================
# Data preprocessing

# Create object for preprocessing
preprocessor = Preprocessing(dataset)

# Preprocess steps
target_size = (512, 512) 
preprocessor.resize_dataset(target_size)
#preprocessor.normalize_data()

# Split dataset
train_dataset, valid_dataset, test_dataset = preprocessor.split_data(train_percentage=0.8, valid_percentage=0.1)

# Create a dictionary to store the datasets
datasets = {
    'train': train_dataset,
    'valid': valid_dataset,
    'test': test_dataset
}

# ======================================================================================================================
# Models Training

trainer = ModelTrainer(batch_size=64, num_classes=5, dataset = datasets, labels = abbrev_labels)

# Dictionary containing model fit parameters
params = {
    'epochs': 40,
    'verbose': 1
}

# Train the model
historic_data,model = trainer.train_efficientnet_transfer_learning(params,input_shape=(*target_size, 3))
plot_accuracy_epochs(historic_data,'accuracy',"EfficientNetB3 learning")

# Predict
print("Prediction")
predicted_labels,true_labels,images = prediction(trainer.test_set,model)

# Report Results
print("Report")
trainer.report_multi_results(predicted_labels,true_labels,"EfficientNet Report")

# Plot some misclassified samples for error analysis
misclassified_samples(predicted_labels,true_labels,images,"B3")