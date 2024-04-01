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
target_size = (512, 512) # original 512x 512
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

trainer = ModelTrainer(batch_size=20, num_classes=5, dataset = datasets, labels = abbrev_labels)

# Dictionary containing model fit parameters
params = {
    'epochs': 2,
    'verbose': 1
}

# Train the model
historic_data,model = trainer.train_efficientnet_transfer_learning(params,input_shape=(*target_size, 3))
plot_accuracy_epochs(historic_data,'accuracy',"EfficientNetB3 learning")

# Predict
print("Prediction")
predicted_labels,true_labels,images = prediction(trainer.valid_set,model)

# Report Results
print("Report")
trainer.report_multi_results(predicted_labels,true_labels,"EfficientNet Report")


# Evaluate model on validation set
#loss, accuracy = model.evaluate(trainer.valid_set)
#print("Validation Loss:", loss)
#print("Validation Accuracy:", accuracy)

# Plot some misclassified samples for error analysis
misclassified_samples(predicted_labels,true_labels,images,"B3")



#data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A
#model_A = A(args...)                 # Build model object.
#acc_A_train = model_A.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
#acc_A_test = model_A.test(args...)   # Test model based on the test set.
#Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task B
#model_B = B(args...)
#acc_B_train = model_B.train(args...)
#acc_B_test = model_B.test(args...)
#Clean up memory/GPU etc...




# ======================================================================================================================
## Print out your results with following format:
#print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                        acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'