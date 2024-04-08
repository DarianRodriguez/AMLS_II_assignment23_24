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
        fig, ax = plt.subplots(figsize=(8, 5))
        label_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('Image Count')
        ax.set_title('Class Distribution')
        ax.set_xticklabels(value_labels, rotation=0,fontsize=12)
        
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


        
class Preprocessing:
    """
    Class containing methods for preprocessing dataset
    """

    def __init__(self,dataset):

        # Initialize class variables here
        self.dataset = dataset
        self.num_samples = sum(1 for _ in self.dataset) # count samples in dataset


    def split_data(self, train_percentage=0.8, valid_percentage=0.1, seed=None):

        """
        Split the dataset into training, validation, and test sets based on the provided percentages.

        Args:
            train_percentage (float): Percentage of the dataset to be used for training. Default is 0.8.
            valid_percentage (float): Percentage of the dataset to be used for validation. Default is 0.1.
            seed (int or None): Seed for shuffling the dataset. If None, shuffling will not be performed. Default is None.

        Returns:
            tuple: A tuple containing three datasets - training, validation, and test datasets.
        """

        # Shuffle dataset
        if seed is not None:
            self.dataset = self.dataset.shuffle(buffer_size=1000, seed=seed)
        else:
            self.dataset = self.dataset.shuffle(buffer_size=1000)


        # Calculate dataset sizes
        num_train_samples = int(train_percentage * self.num_samples)
        num_valid_samples = int(valid_percentage * self.num_samples)
        num_test_samples =self.num_samples - num_train_samples - num_valid_samples


        # Split dataset
        train_dataset = self.dataset.take(num_train_samples)
        remaining_dataset = self.dataset.skip(num_train_samples)
        valid_dataset = remaining_dataset.take(num_valid_samples)
        test_dataset = remaining_dataset.skip(num_valid_samples)

        return train_dataset, valid_dataset, test_dataset
    
    def resize_dataset(self, target_size):

        """
        Resize the dataset images to a target size.

        Args:
            target_size (tuple): Target size for resizing images, specified as (height, width).

        Returns:
            None
        """

        data = self.dataset

        # Define a function to resize images
        def resize_image(image, label, image_name):
            resized_image = tf.image.resize(image, target_size)
            return resized_image, label, image_name
        
        # Apply the resize function to each element in the training dataset
        self.dataset = data.map(resize_image)

    
    def normalize_data(self):
        """
        Normalize image data in the dataset to the range [0, 1].

        Returns:
            None
        """

        def normalize_image(image, label, image_name):
            # convert tensor from uint 8 to float 32 image
            norm_image = tf.cast(image, tf.float32) / 255.0
            return norm_image,label, image_name
        
        # Normalize
        self.dataset = self.dataset.map(normalize_image)


    def plot_distribution(self,data, title:str, abbrev_labels:dict,filename:str):

        # Initialize a dictionary to store counts for each class label
        class_counts = {}

        # Iterate through the dataset and count occurrences of each class label
        for image, label, _ in data:
            label_ref = label.numpy()
            class_counts[label_ref ] = class_counts.get(label_ref , 0) + 1

        labels = list(class_counts.keys())
        counts = list(class_counts.values())
        labels,sorted_counts =  zip(*sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)) # sort by counts
        sorted_labels = [str(label) for label in labels] # convert labels to string to manage order

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_container = ax.bar(sorted_labels, sorted_counts)

        # Calculate total count
        total_count = sum(sorted_counts)

        # Calculate percentages
        percentages = [count / total_count * 100 for count in sorted_counts]

        #get labels abbrev
        abb_labels = [abbrev_labels[abbrev] for abbrev in sorted_labels]
        print(abb_labels)

        # Plot number of each bar
        ax.bar_label(bar_container, labels=["{:.2f}%".format(p) for p in percentages], fontsize=12)
        ax.set_xticklabels(abb_labels, rotation=0)
        ax.set_ylabel('Image Count')
        ax.set_title(title)

        # Define the path where you want to save the plot
        folder_path = "./figures"

        # Save plot in the figures folder
        plt.savefig(f"{folder_path}/{filename}.png")

        # Explicitly close the figure
        plt.close()