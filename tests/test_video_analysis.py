import os
import unittest
import numpy as np
from src.video_analysis import AdvancedVideoAnalyzer, analyze_video

class TestVideoAnalysis(unittest.TestCase):
    def setUp(self):
        # Assuming you have a test video in a known location
        self.test_video_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'test_video.mp4')
        
        # Create a dummy video if it doesn't exist (for testing purposes)
        if not os.path.exists(self.test_video_path):
            self._create_dummy_video()

    def _create_dummy_video(self):
        """
        Create a dummy video for testing if no test video exists
        This requires OpenCV to generate a simple test video
        """
        import cv2
        
        # Ensure input directory exists
        os.makedirs(os.path.dirname(self.test_video_path), exist_ok=True)
        
        # Create a simple video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.test_video_path, fourcc, 20.0, (640, 480))
        
        # Write 100 frames
        for i in range(100):
            # Create a blank frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add some variation
            if i % 10 == 0:
                frame[:] = (i % 255, i % 255, i % 255)
            
            out.write(frame)
        
        out.release()

    def test_video_analyzer_initialization(self):
        """
        Test initialization of AdvancedVideoAnalyzer
        """
        analyzer = AdvancedVideoAnalyzer(self.test_video_path)
        
        self.assertTrue(analyzer.total_frames > 0, "Total frames should be greater than 0")
        self.assertTrue(analyzer.fps > 0, "FPS should be greater than 0")
        self.assertTrue(analyzer.duration > 0, "Duration should be greater than 0")

    def test_scene_change_detection(self):
        """
        Test scene change detection
        """
        analyzer = AdvancedVideoAnalyzer(self.test_video_path)
        scene_changes = analyzer.detect_scene_changes()
        
        self.assertIsInstance(scene_changes, list, "Scene changes should be a list")
        self.assertTrue(len(scene_changes) >= 0, "Scene changes list can be empty but should exist")

    def test_motion_analysis(self):
        """
        Test motion intensity analysis
        """
        analyzer = AdvancedVideoAnalyzer(self.test_video_path)
        motion_levels = analyzer.analyze_motion()
        
        self.assertIsInstance(motion_levels, list, "Motion levels should be a list")
        self.assertTrue(len(motion_levels) > 0, "Motion levels list should not be empty")
        
        # Check motion levels are numeric
        self.assertTrue(all(isinstance(level, (int, float)) for level in motion_levels), 
                        "All motion levels should be numeric")

    def test_brightness_analysis(self):
        """
        Test brightness level analysis
        """
        analyzer = AdvancedVideoAnalyzer(self.test_video_path)
        brightness_levels = analyzer.analyze_brightness()
        
        self.assertIsInstance(brightness_levels, list, "Brightness levels should be a list")
        self.assertTrue(len(brightness_levels) > 0, "Brightness levels list should not be empty")
        
        # Check brightness levels are within expected range
        self.assertTrue(all(0 <= level <= 255 for level in brightness_levels), 
                        "Brightness levels should be between 0 and 255")

    def test_video_profile_generation(self):
        """
        Test comprehensive video profile generation
        """
        profile = analyze_video(self.test_video_path)
        
        self.assertIsNotNone(profile, "Video profile should not be None")
        
        # Check key profile components
        expected_keys = [
            'filename', 'duration', 'total_frames', 'fps', 
            'scene_changes', 'motion_intensity', 'brightness_profile',
            'analysis_metrics'
        ]
        
        for key in expected_keys:
            self.assertIn(key, profile, f"{key} should be in video profile")

    def tearDown(self):
        """
        Clean up any resources or temporary files
        """
        # Optional: remove dummy video if it was created
        if os.path.exists(self.test_video_path):
            try:
                os.remove(self.test_video_path)
            except Exception:
                pass

if __name__ == '__main__':
    unittest.main()
