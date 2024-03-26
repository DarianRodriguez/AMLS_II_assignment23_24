import os
import json
import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class TFRecordDatasetLoader:
    """
    This class is responsible for loading data from TFRecord files and visualizing the distribution of classes.

    """

    def __init__(self, tfrecord_dir, labels_map_path, train_labels_path):
        """
        Initialize the TFRecordDatasetLoader instance.

        Args:
            tfrecord_dir (str): The directory containing TFRecord files.
            labels_map_path (str): The file path to the labels map.
            train_labels_path (str): The file path to the training labels CSV file.
        """
                
        self.tfrecord_dir = tfrecord_dir
        self.labels_map_path = labels_map_path
        self.train_labels_path = train_labels_path
        self.label_dict = None
        self.dataset = None

    def get_feature_description(self):
        """Defines the expected format for parsing TFRecord data."""

        return {
            'image': tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
            'image_name': tf.io.FixedLenFeature([], tf.string),
        }
    
    def _parse_function(self, record):
        """
        Extracts features from a single TFRecord record.

        Args:
            record: A single TFRecord record.

        Returns:
            image: The decoded image tensor.
            label: The label tensor.
            name: The name of the image.
        """
        features = tf.io.parse_single_example(record, self.get_feature_description())
        image = tf.image.decode_jpeg(features['image'], channels=3)
        label = features['target']
        name = features['image_name']
        return image, label, name
    
    def load_tfrecord_dataset(self):
        """
        Loads the TFRecord dataset from the specified directory.

        Returns:
            tf.data.Dataset: The loaded TFRecord dataset.
        """

        tfrecord_files = [os.path.join(self.tfrecord_dir, file) for file in os.listdir(self.tfrecord_dir) if file.endswith('.tfrec')]

        # Create a dataset from your TFRecord files.
        self.dataset = tf.data.TFRecordDataset(tfrecord_files)
        self.dataset = self.dataset.map(self._parse_function)

        return  self.dataset
    
    def load_label_dictionary(self):
        """ Loads the label dictionary from the specified JSON file."""

        with open(self.labels_map_path, 'r') as json_file:
            self.label_dict = json.load(json_file)

        abbreviated_label_dict = {}

        # Extract only the abbreviations from class labels
        for key, value in self.label_dict .items():
            abbreviation = value.split('(')[-1].split(')')[0].strip()
            abbreviated_label_dict[key] = abbreviation

        return abbreviated_label_dict
    
    def plot_image_distribution(self,filename:str):
        """  Plots the distribution of images across different classes."""

        df_info = pd.read_csv(self.train_labels_path)
        label_counts = df_info['label'].value_counts()

        # Load abbreviations dictionary
        abbreviations = self.load_label_dictionary()

        # Map label indices to abbreviations
        value_labels = [abbreviations[str(value)] for value in label_counts.index]

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        label_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('Image Count')
        ax.set_title('Class Distribution')
        ax.set_xticklabels(value_labels, rotation=0)
        
        # Add percentage labels on top of each bar
        total = label_counts.sum()
        for i, v in enumerate(label_counts):
            percentage = "{:.2f}%".format(100 * v / total)
            ax.annotate(percentage, xy=(i, v), ha='center', va='bottom')

        # Define the path where you want to save the plot
        folder_path = "./figures"

        # Save plot in the figures folder
        plt.savefig(f"{folder_path}/{filename}.png")

        # Explicitly close the figure
        plt.close()


    def plot_images_per_class(self,filename:str):

        """
        Plot images per class and save the plot as an image file.

        Args:
            filename (str): The name of the file to save the plot.

        Returns:
            None
        """
            
        # Load dict labels with abbreviations
        dict_labels = self.load_label_dictionary()

        class_labels = len(dict_labels)
        num_cols = 3
        num_rows = math.ceil(class_labels/ num_cols) 

        # Shuffle the dataset for randomness 
        shuffled_dataset = self.dataset.shuffle(buffer_size=500)

        # Create subplot
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))


        for class_label in range(class_labels):

            filtered_dataset = shuffled_dataset.filter(lambda image, label, image_name: label == class_label)

            # Take one element from the filtered dataset
            image, label, image_name = next(iter(filtered_dataset))

            # Calculate subplot index
            row_index = class_label // num_cols # floor division
            col_index = class_label % num_cols
        
            # Extract image label
            label_name = dict_labels[str(class_label)]
            
            # Plot the image
            axes[row_index, col_index].imshow(image)
            axes[row_index, col_index].set_title(f"Class {label.numpy()} ({label_name})")
            axes[row_index, col_index].axis('off')

            
        # If there are fewer subplots than needed, hide the last subplot
        if class_labels < num_rows * num_cols:
            empty_row_index = class_labels // num_cols
            empty_col_index = class_labels % num_cols
            axes[empty_row_index, empty_col_index].axis('off')

        # Define the path where you want to save the plot
        folder_path = "./figures"

        # Save plot in the figures folder
        plt.savefig(f"{folder_path}/{filename}.png")

        # Explicitly close the figure
        plt.close()