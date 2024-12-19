import os
import logging
from typing import Dict, Any, Optional

# Import our previously created modules
from feature_extractor import analyze_video_features
from scene_classifier import analyze_video_scenes
from recommendation_engine import generate_video_recommendations
from error_handler import ErrorHandler, validate_video_file

class VideoPipelineOrchestrator:
    """
    Comprehensive video processing pipeline orchestrator
    
    This class serves as the central coordinator for our video analysis ecosystem,
    demonstrating how different machine learning and computer vision modules 
    can work together to provide rich, multi-dimensional video insights.
    
    Key Design Principles:
    1. Modular Architecture: Each analysis step is independent
    2. Error Resilience: Robust error handling and fallback mechanisms
    3. Extensible: Easy to add new analysis modules
    """
    
    def __init__(self, 
                 output_directory: str = None, 
                 log_level: int = logging.INFO):
        """
        Initialize the video processing pipeline
        
        :param output_directory: Directory to store analysis results
        :param log_level: Logging verbosity level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set output directory
        self.output_directory = output_directory or os.path.join(
            os.path.dirname(__file__), '..', 'output'
        )
        os.makedirs(self.output_directory, exist_ok=True)
    
    @ErrorHandler.retry(max_retries=3)
    def process_video(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Comprehensive video processing workflow
        
        This method demonstrates a complete video analysis pipeline:
        1. Validate video file integrity
        2. Extract low-level visual features
        3. Classify video scenes
        4. Generate intelligent recommendations
        
        :param video_path: Path to the input video file
        :return: Comprehensive video analysis dictionary
        """
        # Step 1: Validate Video File
        if not validate_video_file(video_path):
            self.logger.error(f"Invalid video file: {video_path}")
            return None
        
        try:
            # Step 2: Feature Extraction
            self.logger.info(f"Extracting visual features for {video_path}")
            feature_analysis = analyze_video_features(video_path)
            
            # Step 3: Scene Classification
            self.logger.info(f"Classifying scenes in {video_path}")
            scene_analysis = analyze_video_scenes(video_path)
            
            # Step 4: Generate Recommendations
            self.logger.info(f"Generating recommendations for {video_path}")
            recommendations = generate_video_recommendations(
                video_path, feature_analysis, scene_analysis
            )
            
            # Combine all analysis results
            comprehensive_analysis = {
                'video_path': video_path,
                'feature_analysis': feature_analysis,
                'scene_analysis': scene_analysis,
                'recommendations': recommendations
            }
            
            # Save analysis results
            self._save_analysis_results(comprehensive_analysis)
            
            return comprehensive_analysis
        
        except Exception as e:
            self.logger.error(f"Comprehensive video analysis failed: {e}")
            return None
    
    def _save_analysis_results(self, analysis_data: Dict[str, Any]):
        """
        Save analysis results to output directory
        
        :param analysis_data: Comprehensive video analysis dictionary
        """
        try:
            # Generate unique filename based on video name
            video_filename = os.path.basename(analysis_data['video_path'])
            output_filename = f"{os.path.splitext(video_filename)[0]}_analysis.json"
            output_path = os.path.join(self.output_directory, output_filename)
            
            # Use error handler to manage file writing
            import json
            with open(output_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            self.logger.info(f"Analysis results saved to {output_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {e}")
    
    def batch_process_videos(self, input_directory: str):
        """
        Process all videos in a given directory
        
        :param input_directory: Directory containing video files
        """
        # Support multiple video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        
        # Track processing statistics
        processed_videos = 0
        failed_videos = 0
        
        # Iterate through videos
        for filename in os.listdir(input_directory):
            # Check file extension
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(input_directory, filename)
                
                try:
                    analysis_result = self.process_video(video_path)
                    
                    if analysis_result:
                        processed_videos += 1
                    else:
                        failed_videos += 1
                
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")
                    failed_videos += 1
        
        # Log processing summary
        self.logger.info(
            f"Batch Processing Summary: "
            f"Total Videos: {processed_videos + failed_videos}, "
            f"Processed: {processed_videos}, "
            f"Failed: {failed_videos}"
        )

def run_video_pipeline(
    video_path: str, 
    output_directory: str = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to run the entire video processing pipeline
    
    :param video_path: Path to the input video
    :param output_directory: Optional output directory for results
    :return: Comprehensive video analysis
    """
    pipeline = VideoPipelineOrchestrator(output_directory)
    return pipeline.process_video(video_path)
