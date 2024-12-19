import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from typing import Dict, Any, List, Tuple

class VideoContentClassifier:
    """
    Custom machine learning model for video content classification
    
    This class demonstrates a sophisticated approach to training 
    a deep learning model specifically tailored for video content analysis.
    
    Key Design Principles:
    1. Transfer Learning: Build upon pre-trained architectures
    2. Flexible Training: Support multiple dataset configurations
    3. Comprehensive Evaluation: Detailed model performance metrics
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 10,
                 model_save_dir: str = None):
        """
        Initialize the video content classification model
        
        :param input_shape: Expected input image dimensions
        :param num_classes: Number of content categories to classify
        :param model_save_dir: Directory to save trained models
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Set up model saving directory
        self.model_save_dir = model_save_dir or os.path.join(
            os.path.dirname(__file__), '..', 'models'
        )
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Initialize the model architecture
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """
        Construct a custom convolutional neural network for video content classification
        
        :return: Compiled TensorFlow/Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Convolutional layers with increasing complexity
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Flatten and dense layers
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        
        # Output layer with softmax activation
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with appropriate loss and metrics
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, 
                     data_dir: str, 
                     validation_split: float = 0.2,
                     batch_size: int = 32) -> Dict[str, tf.data.Dataset]:
        """
        Prepare training and validation datasets
        
        :param data_dir: Directory containing training images
        :param validation_split: Proportion of data to use for validation
        :param batch_size: Number of images per training batch
        :return: Dictionary of training and validation datasets
        """
        # Data augmentation to improve model generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Load validation data
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return {
            'train': train_generator,
            'validation': validation_generator
        }
    
    def train(self, 
              data_dir: str, 
              epochs: int = 50, 
              validation_split: float = 0.2,
              batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the video content classification model
        
        :param data_dir: Directory containing training images
        :param epochs: Number of training iterations
        :param validation_split: Proportion of data to use for validation
        :param batch_size: Number of images per training batch
        :return: Training history and model performance metrics
        """
        # Prepare datasets
        datasets = self.prepare_data(
            data_dir, 
            validation_split=validation_split, 
            batch_size=batch_size
        )
        
        # Configure model checkpointing
        checkpoint_path = os.path.join(
            self.model_save_dir, 
            'best_model_{epoch:02d}.h5'
        )
        
        # Callbacks for improved training
        callbacks = [
            ModelCheckpoint(
                checkpoint_path, 
                save_best_only=True, 
                monitor='val_accuracy'
            ),
            EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            datasets['train'],
            validation_data=datasets['validation'],
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Evaluate final model performance
        evaluation = self.model.evaluate(datasets['validation'])
        
        return {
            'history': history.history,
            'final_accuracy': evaluation[1],
            'final_loss': evaluation[0],
            'best_model_path': checkpoint_path
        }
    
    def save_model(self, filename: str = None):
        """
        Save the trained model
        
        :param filename: Optional custom filename
        """
        if not filename:
            filename = os.path.join(
                self.model_save_dir, 
                'video_content_classifier.h5'
            )
        
        self.model.save(filename)
    
    def load_model(self, filename: str = None):
        """
        Load a pre-trained model
        
        :param filename: Path to saved model weights
        """
        if not filename:
            filename = os.path.join(
                self.model_save_dir, 
                'video_content_classifier.h5'
            )
        
        if os.path.exists(filename):
            self.model = tf.keras.models.load_model(filename)
        else:
            raise FileNotFoundError(f"No model found at {filename}")

def train_video_content_model(
    data_directory: str, 
    num_classes: int = 10, 
    epochs: int = 50
) -> Dict[str, Any]:
    """
    Convenience function to train a video content classification model
    
    :param data_directory: Path to training image dataset
    :param num_classes: Number of video content categories
    :param epochs: Number of training iterations
    :return: Training results and model performance metrics
    """
    classifier = VideoContentClassifier(num_classes=num_classes)
    return classifier.train(data_directory, epochs=epochs)
