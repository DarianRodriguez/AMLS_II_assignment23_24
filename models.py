import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from keras.applications import EfficientNetB4,EfficientNetB3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.models import Model
from keras.optimizers import Adam
from keras import layers
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils import compute_class_weight



from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

def sparse_categorical_focal_loss(gamma=2.0, from_logits=False, class_weights=None):
  """
  Sparse Categorical Focal Loss function with alpha for class weights and gamma.

  Args:
    gamma: Focusing parameter (float).
    from_logits: Whether predictions are logits or probabilities (bool).
    class_weights: List of class weights (length must be 5) (optional).

  Returns:
    A loss function that can be used during model compilation.
  """

  def loss(y_true, y_pred):

    epsilon = K.epsilon()

    # Clip prediction values to avoid overflow/underflow
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)

    # Calculate crossentropy loss
    ce = K.sparse_categorical_crossentropy(y_true, y_pred)

    if not from_logits:
      # If predictions are probabilities, apply softmax
      y_pred = K.softmax(y_pred)

    # Calculate modulating factor
    pt = K.sum(K.cast(y_true, 'float32') * y_pred, axis=-1)
    focal_factor = K.pow(1 - pt, gamma)

    # Apply class weights (if provided) if not all classes equally important
    if class_weights:
      if len(class_weights) != 5:
        raise ValueError("Class weights must have length 5 for 5 classes!")
      
      class_weights_tensor = K.constant(class_weights)
      # Print shapes for debugging
      print("Shape of class_weights_tensor:", K.shape(class_weights_tensor))
      print("Shape of y_true:", K.shape(y_true))
      print(y_true)
      print(class_weights_tensor)
        
      focal_factor *= class_weights_tensor[y_true]

    # Apply focal loss formula with weighted modulating factor
    loss = K.mean(focal_factor * ce)
    return loss

  return loss



class ModelTrainer:

    def __init__(self, batch_size, labels:dict, dataset, num_classes=5):

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.class_labels_dict = labels 
        self.class_names = list(self.class_labels_dict.values())

        # Reconfigure the datasets to have only image and label
        self.valid_set = dataset['valid'].map(lambda image, label, _: (image, label)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.train_set = dataset['train'].map(lambda image, label, _: (image, label)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.test_set = dataset['test'].map(lambda image, label, _: (image, label)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        dataset_ytrain = dataset['train'].map(lambda image, label,_: label)
        self.ytrain =  [label.numpy() for label in dataset_ytrain]

        #test_set = dataset['test'].map(lambda image, label, image_name: (image))
        #test_set = dataset['valid'].map(lambda image, label, image_name: (image))
        #self.test_images = test_set.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        #self.test_images_list = list(test_set)
        #self.test_labels= list(dataset['valid'].map(lambda image, label, image_name: (label)).as_numpy_iterator())



    def train_efficientnet_transfer_learning(self, params, input_shape=(512, 512, 3)):

        base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_shape)

        # Freeze base model layers
        base_model.trainable = False

        # Add layers on top of base model
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = Dense(5, activation="softmax")(x)


        # Unfreeze some layers
        for layer in base_model.layers[-5:]:
            layer.trainable = True

        #loss_fn = sparse_categorical_focal_loss(gamma=2, from_logits=True, class_weights = class_weights_v1)


        # Define the model
        model = Model(inputs=base_model.input, outputs=outputs)

        # Compile the model
        #model.compile(optimizer=Adam(),
        #            loss=SparseCategoricalCrossentropy(),
        #            metrics=['accuracy'])

        # Use original class distribution as class weights
        #class_distribution = {3: 0.61, 4: 0.12, 2: 0.11, 1: 0.10, 0: 0.05}
        #class_weights = {3: 1 / 0.61, 4: 1 / 0.12, 2: 1 / 0.11, 1: 1 / 0.10, 0: 1 / 0.05}
        class_weights = compute_weights(self.ytrain)
        print(class_weights)

        # Define the callback
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.1, 
                                    patience=5, 
                                    min_lr=0.0001)

        optimizer = Adam(learning_rate=0.01) 

        # Compile the model
        model.compile(optimizer=Adam(),
                    loss=SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

        # Train the model 
        historic_data = model.fit(self.train_set, validation_data = self.valid_set,class_weight=class_weights, **params) #callbacks=[reduce_lr]

        return historic_data,model
    
    def report_multi_results(self,y_pred,y_true,filename:str):
        
        """
        Generate report for the Classifier.

        Args:
            y_pred (array-like): Predicted labels generated by the classifier.
            y_real (array-like): Ground truth (real) labels.
            labels(str): xticks labels
            name (str) : filename
        """

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy:", accuracy)

        # Calculate other metrics
        f1score = f1_score(y_true, y_pred,average='weighted')
        precision = precision_score(y_true, y_pred,average='weighted')
        recall = recall_score(y_true, y_pred,average='weighted')
        print(f"F1-Score of the Classifier: {f1score:.2f}")
        print(f"precision of the Classifier: {precision:.2f}")
        print(f"recall of the Classifier: {recall:.2f}")

        # Generate classification report
        report = classification_report(y_true, y_pred,target_names=self.class_names)
        print("Classification Report:\n")
        print(report)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Create a heatmap for the confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        # Define the path where you want to save the plot
        folder_path = "./figures"

        # Save plot in the figures folder
        plt.savefig(f"{folder_path}/{filename}.png")
        
        # Explicitly close the figure
        plt.close()


def plot_accuracy_epochs(history,metric, filename:str):
    """
    Generate plot of the model accuracy across the epochs

    Arguments:
        history: history object from model training
        filename (str) : specify the file name

    Returns:
    """
    
    plt.figure(figsize=(20,5))

    # Create a list of the epochs
    epochs = range(1,len(history.history['loss'])+1)

    plt.plot(epochs,history.history[metric], label='Training Accuracy',marker='o')
    plt.plot(epochs,history.history['val_'+ metric], label='Validation Accuracy',marker='o')
    plt.xticks(epochs)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Define the path where you want to save the plot
    folder_path = "./figures"

    # Save plot in the figures folder
    plt.savefig(f"{folder_path}/{filename}.png")

    # Explicitly close the figure
    plt.close()

def prediction(data, model):

    true_labels, predicted_labels = [], []
    images = []  # List to store images

    # Iterate over the validation dataset
    for image, label in data:

        # Make predictions for the current batch
        predicted_probabilities = model.predict(image,verbose=0)
        # Convert predictions to class labels
        batch_predicted_labels = np.argmax(predicted_probabilities, axis=1)
        # Append true and predicted labels to the lists
        true_labels.extend(label.numpy())
        predicted_labels.extend(batch_predicted_labels)
        images.extend(image.numpy())
        
    return predicted_labels,true_labels,images

def misclassified_samples(predicted_labels,true_labels,images,filename):

    num_samples = 6
    # Find misclassified indexes
    misclassified_indexes = np.where(np.array(predicted_labels) != np.array(true_labels))[0]

    misclassified_indexes_list = misclassified_indexes[:num_samples]

    num_cols = 3
    num_rows = math.ceil(num_samples / num_cols) 

    # Create subplot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

    # Show misclassified images
    for index, misclassified_index in enumerate(misclassified_indexes_list):
        # Extract misclassified image from the dataset
        image = images[misclassified_index]
        
        # Make prediction
        predicted_label = predicted_labels[misclassified_index]
        true_label = true_labels[misclassified_index]

        # Calculate subplot index
        row_index = index // num_cols # floor division
        col_index = index % num_cols

        # Visualize the misclassified image
        axes[row_index, col_index].imshow(image.astype(np.uint8))
        axes[row_index, col_index].set_title(f'True label: {true_label}, Predicted label: {predicted_label}')
        axes[row_index, col_index].axis('off')

    # If there are fewer subplots than needed, hide the last subplot
    if num_samples < num_rows * num_cols:
        empty_row_index = num_samples // num_cols
        empty_col_index = num_samples % num_cols
        axes[empty_row_index, empty_col_index].axis('off')
    
    # Define the path where you want to save the plot
    folder_path = "./figures"

    # Save plot in the figures folder
    plt.savefig(f"{folder_path}/misclassification_{filename}.png")

    # Explicitly close the figure
    plt.close()

def compute_weights(y_train):

    #y_train = list(dataset['train'].map(lambda image, label, _: (label)))
    class_weights = compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(y_train),
                                            y = y_train                                                
                                        )
    class_weights = dict(zip(np.unique(y_train), class_weights))

    return class_weights