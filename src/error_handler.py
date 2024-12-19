import logging
import traceback
from typing import Optional, Any, Callable

class VideoProcessingError(Exception):
    """Base exception for video processing errors"""
    pass

class RetryableError(VideoProcessingError):
    """Errors that can be potentially resolved by retrying"""
    pass

class ErrorHandler:
    """
    Centralized error handling and logging for video processing pipeline
    """
    
    @staticmethod
    def log_error(error: Exception, context: Optional[dict] = None):
        """
        Log detailed error information
        
        :param error: Exception to log
        :param context: Additional context about the error
        """
        logger = logging.getLogger(__name__)
        
        # Log basic error details
        logger.error(f"Error occurred: {type(error).__name__}")
        logger.error(f"Error message: {str(error)}")
        
        # Log stack trace
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())
        
        # Log additional context if provided
        if context:
            logger.error("Error Context:")
            for key, value in context.items():
                logger.error(f"{key}: {value}")

    @staticmethod
    def retry(
        func: Callable, 
        max_retries: int = 3, 
        retriable_exceptions: tuple = (RetryableError,),
        initial_wait: float = 1.0,
        backoff_factor: float = 2.0
    ):
        """
        Retry decorator for handling retriable errors
        
        :param func: Function to retry
        :param max_retries: Maximum number of retry attempts
        :param retriable_exceptions: Tuple of exceptions that trigger a retry
        :param initial_wait: Initial wait time between retries
        :param backoff_factor: Factor to increase wait time between retries
        :return: Decorated function
        """
        import time
        
        def wrapper(*args, **kwargs):
            retries = 0
            wait_time = initial_wait
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except retriable_exceptions as e:
                    retries += 1
                    
                    # Log retry attempt
                    logging.warning(
                        f"Retry attempt {retries}/{max_retries} for {func.__name__}. "
                        f"Error: {type(e).__name__}"
                    )
                    
                    # Backoff strategy
                    if retries < max_retries:
                        time.sleep(wait_time)
                        wait_time *= backoff_factor
                    else:
                        # Re-raise the last exception if all retries fail
                        ErrorHandler.log_error(e, {
                            'function': func.__name__,
                            'args': args,
                            'kwargs': kwargs
                        })
                        raise
        
        return wrapper

def validate_video_file(video_path: str) -> bool:
    """
    Validate video file integrity
    
    :param video_path: Path to the video file
    :return: True if video is valid, False otherwise
    """
    import os
    import cv2
    
    # Check file exists
    if not os.path.exists(video_path):
        logging.error(f"Video file does not exist: {video_path}")
        return False
    
    # Check file is not empty
    if os.path.getsize(video_path) == 0:
        logging.error(f"Video file is empty: {video_path}")
        return False
    
    # Try to open video with OpenCV
    try:
        video = cv2.VideoCapture(video_path)
        
        # Check if video can be opened
        if not video.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return False
        
        # Check total frames
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logging.error(f"No frames in video: {video_path}")
            return False
        
        return True
    
    except Exception as e:
        logging.error(f"Error validating video: {e}")
        return False
    finally:
        # Ensure video capture is released
        video.release()
