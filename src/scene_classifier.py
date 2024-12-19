import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from typing import List, Dict, Any

class SceneClassifier:
    """
    Advanced scene classification using pre-trained deep learning models
    
    This class provides intelligent scene understanding by leveraging
    transfer learning and pre-trained neural networks.
    """
    
    def __init__(self, model_cache_dir: str = None):
        """
        Initialize scene classification model
        
        :param model_cache_dir: Directory to cache downloaded model weights
        """
        # Set up model caching
        if model_cache_dir:
            os.makedirs(model_cache_dir, exist_ok=True)
            tf.keras.utils.get_file(
                origin='https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                cache_dir=model_cache_dir
            )
        
        # Load pre-trained InceptionV3 model
        self.model = InceptionV3(weights='imagenet')
        
        # Scene classification thresholds
        self.confidence_threshold = 0.3
    
    def classify_video_scenes(self, video_path: str, sample_rate: int = 30) -> List[Dict[str, Any]]:
        """
        Classify scenes throughout the video
        
        :param video_path: Path to the input video
        :param sample_rate: Process every nth frame
        :return: List of scene classification results
        """
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Scene classification results
        scene_classifications = []
        
        for frame_no in range(0, total_frames, sample_rate):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Preprocess frame for InceptionV3
            preprocessed_frame = self._preprocess_frame(frame)
            
            # Predict scene
            predictions = self.model.predict(preprocessed_frame)
            decoded_predictions = decode_predictions(predictions, top=3)[0]
            
            # Filter predictions above confidence threshold
            filtered_predictions = [
                {
                    'label': pred[1],
                    'confidence': float(pred[2]),
                    'timestamp': frame_no / fps
                }
                for pred in decoded_predictions
                if float(pred[2]) >= self.confidence_threshold
            ]
            
            # Add to results if any meaningful predictions
            if filtered_predictions:
                scene_classifications.append({
                    'frame_number': frame_no,
                    'timestamp': frame_no / fps,
                    'predictions': filtered_predictions
                })
        
        cap.release()
        return scene_classifications
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess video frame for InceptionV3 model
        
        :param frame: Input video frame
        :return: Preprocessed frame ready for prediction
        """
        # Resize frame to InceptionV3 input size
        resized_frame = cv2.resize(frame, (299, 299))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to array and expand dimensions
        frame_array = img_to_array(rgb_frame)
        frame_array = np.expand_dims(frame_array, axis=0)
        
        # Preprocess for InceptionV3
        return preprocess_input(frame_array)
    
    def analyze_scene_composition(self, video_path: str) -> Dict[str, Any]:
        """
        Provide a comprehensive analysis of scene composition
        
        :param video_path: Path to the input video
        :return: Dictionary of scene composition insights
        """
        scene_classifications = self.classify_video_scenes(video_path)
        
        # Aggregate scene insights
        scene_summary = {
            'total_classified_scenes': len(scene_classifications),
            'scene_types': {},
            'temporal_distribution': {}
        }
        
        # Categorize and count scene types
        for scene in scene_classifications:
            for prediction in scene['predictions']:
                label = prediction['label']
                
                # Count scene type occurrences
                if label not in scene_summary['scene_types']:
                    scene_summary['scene_types'][label] = 0
                scene_summary['scene_types'][label] += 1
                
                # Track temporal distribution
                timestamp = scene['timestamp']
                time_bucket = int(timestamp / 60)  # Bucket by minute
                
                if time_bucket not in scene_summary['temporal_distribution']:
                    scene_summary['temporal_distribution'][time_bucket] = []
                
                scene_summary['temporal_distribution'][time_bucket].append(label)
        
        return scene_summary

def analyze_video_scenes(video_path: str) -> Dict[str, Any]:
    """
    Convenience function to analyze video scenes
    
    :param video_path: Path to the video file
    :return: Comprehensive scene analysis
    """
    classifier = SceneClassifier()
    return classifier.analyze_scene_composition(video_path)
