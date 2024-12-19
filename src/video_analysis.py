import cv2
import numpy as np
import logging
from typing import Dict, List, Optional

class AdvancedVideoAnalyzer:
    def __init__(self, video_path: str):
        """
        Initialize advanced video analysis
        
        :param video_path: Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.logger = logging.getLogger(__name__)
        
        # Basic video metadata
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps
        
        # Analysis containers
        self.scene_changes = []
        self.motion_levels = []
        self.brightness_levels = []
        
    def detect_scene_changes(self, threshold: float = 0.3) -> List[float]:
        """
        Detect significant scene changes
        
        :param threshold: Sensitivity of scene change detection
        :return: List of timestamps with scene changes
        """
        scene_changes = []
        prev_frame = None
        
        for frame_no in range(0, self.total_frames, max(1, int(self.fps / 2))):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Compute frame difference
                diff = cv2.absdiff(prev_frame, gray)
                non_zero_count = np.count_nonzero(diff)
                change_ratio = non_zero_count / (diff.shape[0] * diff.shape[1])
                
                if change_ratio > threshold:
                    scene_changes.append(frame_no / self.fps)
            
            prev_frame = gray
        
        self.scene_changes = scene_changes
        return scene_changes
    
    def analyze_motion(self) -> List[float]:
        """
        Analyze motion intensity throughout the video
        
        :return: List of motion intensity levels
        """
        motion_levels = []
        prev_frame = None
        
        for frame_no in range(0, self.total_frames, max(1, int(self.fps / 2))):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Compute motion magnitude
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_intensity = np.mean(magnitude)
                motion_levels.append(motion_intensity)
            
            prev_frame = gray
        
        self.motion_levels = motion_levels
        return motion_levels
    
    def analyze_brightness(self) -> List[float]:
        """
        Analyze brightness levels throughout the video
        
        :return: List of brightness levels
        """
        brightness_levels = []
        
        for frame_no in range(0, self.total_frames, max(1, int(self.fps / 2))):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Convert to grayscale and compute average brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_levels.append(brightness)
        
        self.brightness_levels = brightness_levels
        return brightness_levels
    
    def generate_video_profile(self) -> Dict[str, any]:
        """
        Generate a comprehensive video profile
        
        :return: Dictionary with video analysis results
        """
        # Detect scene changes
        scene_changes = self.detect_scene_changes()
        
        # Analyze motion
        motion_levels = self.analyze_motion()
        
        # Analyze brightness
        brightness_levels = self.analyze_brightness()
        
        # Close video capture
        self.cap.release()
        
        return {
            'filename': self.video_path,
            'duration': self.duration,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'scene_changes': scene_changes,
            'motion_intensity': {
                'mean': np.mean(motion_levels),
                'max': np.max(motion_levels),
                'min': np.min(motion_levels)
            },
            'brightness_profile': {
                'mean': np.mean(brightness_levels),
                'max': np.max(brightness_levels),
                'min': np.min(brightness_levels)
            },
            'analysis_metrics': {
                'scene_change_count': len(scene_changes),
                'high_motion_frames': sum(1 for m in motion_levels if m > np.mean(motion_levels)),
                'brightness_variation': np.std(brightness_levels)
            }
        }

def analyze_video(video_path: str) -> Optional[Dict[str, any]]:
    """
    Convenience function to analyze a video
    
    :param video_path: Path to the video file
    :return: Video profile or None if analysis fails
    """
    try:
        analyzer = AdvancedVideoAnalyzer(video_path)
        return analyzer.generate_video_profile()
    except Exception as e:
        logging.error(f"Video analysis failed for {video_path}: {e}")
        return None
