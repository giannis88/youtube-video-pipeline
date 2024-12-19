import os
import time
import logging
import json
import google.generativeai as genai
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import ffmpeg

# Import our new video analysis module
from video_analysis import analyze_video

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='video_pipeline.log'
)

class GeminiAPIManager:
    def __init__(self, api_keys):
        """
        Manage multiple Gemini API keys for rate limit mitigation
        
        :param api_keys: List of API keys
        """
        self.api_keys = api_keys
        self.key_usage_count = {key: 0 for key in api_keys}

    def get_next_key(self):
        """
        Rotate to the next API key, selecting the least used
        
        :return: Selected API key
        """
        # Find the key with the least usage
        least_used_key = min(self.key_usage_count, key=self.key_usage_count.get)
        
        # Update usage count
        self.key_usage_count[least_used_key] += 1
        
        return least_used_key

class VideoProcessor:
    def __init__(self, input_dir, output_dir, api_keys):
        """
        Initialize video processing pipeline
        
        :param input_dir: Directory to watch for new videos
        :param output_dir: Directory to store processed videos
        :param api_keys: List of Gemini API keys
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Initialize API Manager
        self.api_manager = GeminiAPIManager(api_keys)
        
        # Configure Gemini API
        try:
            api_key = self.api_manager.get_next_key()
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logging.info(f"Gemini API initialized with key: {api_key[:10]}...")
        except Exception as e:
            logging.error(f"Gemini API initialization failed: {e}")
            raise

    def generate_youtube_metadata(self, video_analysis):
        """
        Generate YouTube metadata using Gemini AI
        
        :param video_analysis: Dictionary of video analysis results
        :return: Dictionary of metadata suggestions
        """
        try:
            # Construct a detailed prompt using video analysis
            metadata_prompt = f"""
            Generate YouTube metadata for a video with these characteristics:
            - Duration: {video_analysis['duration']:.2f} seconds
            - Scene Changes: {video_analysis['scene_change_count']} 
            - Motion Intensity: {video_analysis['motion_intensity']['mean']:.2f}
            - Brightness Variation: {video_analysis['brightness_profile']['mean']:.2f}

            Provide:
            1. An engaging video title (max 60 chars)
            2. A compelling description (max 300 chars)
            3. 5-7 relevant tags
            """
            
            # Generate metadata using Gemini
            metadata_response = self.model.generate_content(metadata_prompt)
            
            # Parse the response (you might need more sophisticated parsing)
            metadata_lines = metadata_response.text.split('\n')
            
            return {
                'title': metadata_lines[0].strip() if len(metadata_lines) > 0 else "Untitled Video",
                'description': metadata_lines[1].strip() if len(metadata_lines) > 1 else "No description generated",
                'tags': [tag.strip() for tag in metadata_lines[2:7] if tag.strip()] if len(metadata_lines) > 2 else []
            }
        except Exception as e:
            logging.error(f"Metadata generation failed: {e}")
            return {
                'title': f"Video {time.time()}",
                'description': "Automatically generated video",
                'tags': []
            }

    def process_video(self, video_path):
        """
        Process the input video
        
        :param video_path: Path to the input video
        :return: Path to the processed video
        """
        try:
            # Perform advanced video analysis
            video_analysis = analyze_video(video_path)
            
            if not video_analysis:
                logging.error(f"Video analysis failed for {video_path}")
                return None
            
            # Generate YouTube metadata
            youtube_metadata = self.generate_youtube_metadata(video_analysis)
            
            # Generate output filename
            output_filename = f"processed_{os.path.basename(video_path)}"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # FFmpeg processing - resize and optimize
            (
                ffmpeg
                .input(video_path)
                .filter('scale', 1280, 720)  # Resize to 720p
                .output(output_path, 
                        vcodec='libx264',  # Use H.264 codec
                        crf=23,  # Balanced quality
                        preset='medium'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Store video analysis results
            analysis_path = os.path.join(self.output_dir, f"{output_filename}_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump({
                    'video_analysis': video_analysis,
                    'youtube_metadata': youtube_metadata
                }, f, indent=2)
            
            logging.info(f"Processed video: {output_path}")
            
            return output_path
        
        except Exception as e:
            logging.error(f"Video processing error for {video_path}: {e}")
            return None

# ... (rest of the script remains the same as in the previous version)
