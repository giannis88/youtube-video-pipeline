import cv2
import numpy as np
from typing import Dict, List, Tuple

class VideoFeatureExtractor:
    """
    Advanced feature extraction for video content analysis
    
    This class provides deep insights into video characteristics
    by analyzing visual and temporal patterns.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize feature extractor for a specific video
        
        :param video_path: Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Core video metadata
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps
        
        # Feature storage
        self.color_histogram = []
        self.edge_density = []
        self.motion_vectors = []
        
    def extract_color_histogram(self, sample_rate: int = 10) -> List[np.ndarray]:
        """
        Extract color distribution across video frames
        
        :param sample_rate: Process every nth frame
        :return: List of color histograms
        """
        histograms = []
        for frame_no in range(0, self.total_frames, sample_rate):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Convert to HSV for more meaningful color representation
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Compute histogram for each color channel
            hist = cv2.calcHist([hsv_frame], [0, 1], None, [180, 256], [0, 180, 0, 256])
            histograms.append(hist)
        
        self.color_histogram = histograms
        return histograms
    
    def compute_edge_density(self, sample_rate: int = 10) -> List[float]:
        """
        Measure the complexity of frames through edge detection
        
        :param sample_rate: Process every nth frame
        :return: List of edge density values
        """
        edge_densities = []
        
        for frame_no in range(0, self.total_frames, sample_rate):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect edges using Canny
            edges = cv2.Canny(gray, 100, 200)
            
            # Compute edge density
            density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            edge_densities.append(density)
        
        self.edge_density = edge_densities
        return edge_densities
    
    def detect_camera_movement(self, sample_rate: int = 10) -> List[Dict[str, float]]:
        """
        Analyze camera movement and scene dynamics
        
        :param sample_rate: Process every nth frame
        :return: List of camera movement characteristics
        """
        movements = []
        prev_frame = None
        
        for frame_no in range(0, self.total_frames, sample_rate):
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
                
                # Compute flow magnitude and angle
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                movements.append({
                    'avg_motion': np.mean(magnitude),
                    'motion_variation': np.std(magnitude),
                    'dominant_direction': np.mean(angle)
                })
            
            prev_frame = gray
        
        self.motion_vectors = movements
        return movements
    
    def generate_video_complexity_profile(self) -> Dict[str, Any]:
        """
        Create a comprehensive profile of video complexity
        
        :return: Dictionary of video complexity metrics
        """
        # Ensure feature extraction is complete
        if not self.color_histogram:
            self.extract_color_histogram()
        
        if not self.edge_density:
            self.compute_edge_density()
        
        if not self.motion_vectors:
            self.detect_camera_movement()
        
        return {
            'total_frames': self.total_frames,
            'duration': self.duration,
            'fps': self.fps,
            'color_complexity': {
                'variance': np.var([np.sum(hist) for hist in self.color_histogram]),
                'unique_color_distribution': len(set(tuple(map(tuple, hist.flatten())) for hist in self.color_histogram))
            },
            'visual_complexity': {
                'avg_edge_density': np.mean(self.edge_density),
                'edge_density_variance': np.var(self.edge_density)
            },
            'camera_dynamics': {
                'avg_motion': np.mean([m['avg_motion'] for m in self.motion_vectors]),
                'motion_variation': np.mean([m['motion_variation'] for m in self.motion_vectors])
            }
        }

def analyze_video_features(video_path: str) -> Dict[str, Any]:
    """
    Convenience function to extract video features
    
    :param video_path: Path to the video file
    :return: Comprehensive video feature profile
    """
    extractor = VideoFeatureExtractor(video_path)
    return extractor.generate_video_complexity_profile()
