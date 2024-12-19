import os
import numpy as np
import tensorflow as tf

class ModelValidator:
    """
    Simple, practical model validation class
    
    Focuses on providing clear, actionable insights about model performance
    """
    
    def __init__(self, model_path):
        """
        Initialize validator with a specific model
        
        :param model_path: Path to saved TensorFlow/Keras model
        """
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load model safely
        self.model = tf.keras.models.load_model(model_path)
    
    def validate_model(self, test_data):
        """
        Perform basic model validation
        
        :param test_data: Validation dataset
        :return: Dictionary of performance metrics
        """
        # Compute performance metrics
        results = self.model.evaluate(test_data)
        
        return {
            'loss': results[0],
            'accuracy': results[1]
        }
    
    def predict_sample(self, sample_image):
        """
        Make a prediction on a single sample image
        
        :param sample_image: Preprocessed image for prediction
        :return: Prediction results
        """
        prediction = self.model.predict(sample_image)
        return np.argmax(prediction)

def validate_video_content_model(model_path, test_data):
    """
    Convenience function for model validation
    
    :param model_path: Path to saved model
    :param test_data: Test dataset
    :return: Performance metrics
    """
    validator = ModelValidator(model_path)
    return validator.validate_model(test_data)
