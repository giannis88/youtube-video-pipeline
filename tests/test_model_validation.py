import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Ensure we can import from src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_validator import ModelValidator

class TestModelValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Create a simple test model for validation
        """
        # Create a dummy model for testing
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Flatten()(inputs)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        
        cls.test_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        cls.test_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save test model
        cls.model_path = os.path.join(
            os.path.dirname(__file__), 
            'test_video_classifier.h5'
        )
        cls.test_model.save(cls.model_path)
    
    def setUp(self):
        """
        Prepare test data for each test method
        """
        # Generate dummy test data
        self.test_data = (
            np.random.random((100, 224, 224, 3)),  # Input images
            tf.keras.utils.to_categorical(
                np.random.randint(10, size=(100, 1)), 
                num_classes=10
            )  # Categorical labels
        )
    
    def test_model_loading(self):
        """
        Verify model can be loaded correctly
        """
        validator = ModelValidator(self.model_path)
        self.assertIsNotNone(validator.model, "Model should be loaded successfully")
    
    def test_model_validation(self):
        """
        Test basic model validation performance
        """
        validator = ModelValidator(self.model_path)
        results = validator.validate_model(self.test_data)
        
        # Check results have expected keys
        self.assertIn('loss', results)
        self.assertIn('accuracy', results)
        
        # Basic sanity checks on results
        self.assertGreaterEqual(results['accuracy'], 0)
        self.assertLessEqual(results['accuracy'], 1)
    
    def test_prediction_sample(self):
        """
        Test prediction on a single sample
        """
        validator = ModelValidator(self.model_path)
        sample_image = np.random.random((1, 224, 224, 3))
        
        prediction = validator.predict_sample(sample_image)
        
        # Check prediction is a valid class index
        self.assertGreaterEqual(prediction, 0)
        self.assertLess(prediction, 10)
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up test model file
        """
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)

if __name__ == '__main__':
    unittest.main()
