import os
import time
import logging
import random
import google.generativeai as genai
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
import ffmpeg

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
        self.current_key_index = 0
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

    def analyze_video(self, video_path):
        """
        Perform basic video analysis
        
        :param video_path: Path to the input video
        :return: Dictionary of video insights
        """
        try:
            # Open video using OpenCV
            video = cv2.VideoCapture(video_path)
            
            # Extract basic video metadata
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            # Generate insights using Gemini
            insights_prompt = f"""
            Analyze a video with the following characteristics:
            - Filename: {os.path.basename(video_path)}
            - Duration: {duration:.2f} seconds
            - Frames: {total_frames}
            - FPS: {fps}

            Provide:
            1. A brief thematic description
            2. Potential YouTube title suggestions (3 options)
            3. Key topics or themes
            """
            
            # Retry mechanism with key rotation
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    insights_response = self.model.generate_content(insights_prompt)
                    break
                except Exception as e:
                    logging.warning(f"API call failed (Attempt {attempt+1}): {e}")
                    # Rotate to next API key
                    new_key = self.api_manager.get_next_key()
                    genai.configure(api_key=new_key)
                    self.model = genai.GenerativeModel('gemini-pro')
            
            return {
                'filename': os.path.basename(video_path),
                'duration': duration,
                'total_frames': total_frames,
                'fps': fps,
                'ai_insights': insights_response.text
            }
        except Exception as e:
            logging.error(f"Video analysis error for {video_path}: {e}")
            return None
        finally:
            video.release()

    def process_video(self, video_path):
        """
        Process the input video
        
        :param video_path: Path to the input video
        :return: Path to the processed video
        """
        try:
            # Analyze video first
            video_insights = self.analyze_video(video_path)
            
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
            
            logging.info(f"Processed video: {output_path}")
            
            # Store insights
            insights_path = os.path.join(self.output_dir, f"{output_filename}_insights.txt")
            with open(insights_path, 'w') as f:
                f.write(str(video_insights))
            
            return output_path
        
        except Exception as e:
            logging.error(f"Video processing error for {video_path}: {e}")
            return None

class VideoHandler(FileSystemEventHandler):
    def __init__(self, processor):
        """
        Initialize file system event handler
        
        :param processor: VideoProcessor instance
        """
        self.processor = processor

    def on_created(self, event):
        """
        Handle new file creation events
        
        :param event: File system event
        """
        if not event.is_directory:
            # Check for video file extensions
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
            if any(event.src_path.lower().endswith(ext) for ext in video_extensions):
                logging.info(f"New video detected: {event.src_path}")
                self.processor.process_video(event.src_path)

def main():
    """
    Main entry point for the video processing pipeline
    """
    # Configuration 
    INPUT_DIR = r'C:\Users\giova\youtube-video-pipeline\input'
    OUTPUT_DIR = r'C:\Users\giova\youtube-video-pipeline\output'
    
    # Gemini API Keys
    GEMINI_API_KEYS = [
        'AIzaSyABilhc1fzLxbvmT0M1RZCN2DyOO-M3DMw',
        'AIzaSyCAWHiK2MB8UlLY2OFViG9Z8q1yEtPu8LU',
        'AIzaSyC6d_bp4gfbXfqexEV5wr8CJgqY3bxxNeA'
    ]

    # Ensure directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize processor
    processor = VideoProcessor(INPUT_DIR, OUTPUT_DIR, GEMINI_API_KEYS)
    
    # Setup file system watcher
    event_handler = VideoHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    
    try:
        logging.info("Starting video processing pipeline...")
        observer.start()
        
        # Keep main thread running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logging.info("Stopping video processing pipeline...")
        observer.stop()
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    
    finally:
        observer.join()

if __name__ == '__main__':
    main()
