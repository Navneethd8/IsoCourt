import numpy as np
from typing import List, Dict, Tuple

class BadmintonPoseDetector:
    """
    Detects badminton-specific poses using MediaPipe landmarks.
    
    MediaPipe Pose Landmark Indices:
    - 11, 12: Left/Right Shoulder
    - 13, 14: Left/Right Elbow
    - 15, 16: Left/Right Wrist
    - 23, 24: Left/Right Hip
    - 25, 26: Left/Right Knee
    - 27, 28: Left/Right Ankle
    """
    
    def __init__(self, 
                 overhead_threshold: float = 0.15,
                 stance_width_threshold: float = 0.25,
                 arm_asymmetry_threshold: float = 0.20):
        """
        Args:
            overhead_threshold: Minimum vertical distance (normalized) for overhead motion
            stance_width_threshold: Minimum horizontal distance (normalized) for wide stance
            arm_asymmetry_threshold: Minimum difference in arm positions for racket detection
        """
        self.overhead_threshold = overhead_threshold
        self.stance_width_threshold = stance_width_threshold
        self.arm_asymmetry_threshold = arm_asymmetry_threshold
    
    def _get_landmark_coords(self, landmarks: List[Dict], idx: int) -> Tuple[float, float, float]:
        """Extract x, y, z coordinates from a landmark."""
        if idx < len(landmarks):
            lm = landmarks[idx]
            return lm['x'], lm['y'], lm['z']
        return 0.0, 0.0, 0.0
    
    def detect_overhead_motion(self, landmarks: List[Dict]) -> Tuple[bool, float]:
        """
        Detect if arms are raised overhead (typical for smash, clear, serve).
        
        Returns:
            (is_overhead, confidence_score)
        """
        if not landmarks or len(landmarks) < 17:
            return False, 0.0
        
        # Get shoulder and wrist positions
        left_shoulder_x, left_shoulder_y, _ = self._get_landmark_coords(landmarks, 11)
        right_shoulder_x, right_shoulder_y, _ = self._get_landmark_coords(landmarks, 12)
        left_wrist_x, left_wrist_y, _ = self._get_landmark_coords(landmarks, 15)
        right_wrist_x, right_wrist_y, _ = self._get_landmark_coords(landmarks, 16)
        
        # Calculate average shoulder height
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        
        # Check if either wrist is significantly above shoulders
        # (y decreases as you go up in image coordinates)
        left_overhead = (left_shoulder_y - left_wrist_y) > self.overhead_threshold
        right_overhead = (right_shoulder_y - right_wrist_y) > self.overhead_threshold
        
        # Calculate confidence based on how far above shoulders
        left_distance = max(0, left_shoulder_y - left_wrist_y)
        right_distance = max(0, right_shoulder_y - right_wrist_y)
        max_distance = max(left_distance, right_distance)
        
        # Normalize confidence (0.0 to 1.0)
        confidence = min(1.0, max_distance / (self.overhead_threshold * 3))
        
        is_overhead = left_overhead or right_overhead
        return is_overhead, confidence if is_overhead else 0.0
    
    def detect_wide_stance(self, landmarks: List[Dict]) -> Tuple[bool, float]:
        """
        Detect wide badminton stance (feet apart).
        
        Returns:
            (is_wide_stance, confidence_score)
        """
        if not landmarks or len(landmarks) < 29:
            return False, 0.0
        
        # Get ankle positions (more stable than feet)
        left_ankle_x, left_ankle_y, _ = self._get_landmark_coords(landmarks, 27)
        right_ankle_x, right_ankle_y, _ = self._get_landmark_coords(landmarks, 28)
        
        # Get hip positions for normalization
        left_hip_x, left_hip_y, _ = self._get_landmark_coords(landmarks, 23)
        right_hip_x, right_hip_y, _ = self._get_landmark_coords(landmarks, 24)
        
        # Calculate stance width
        ankle_width = abs(left_ankle_x - right_ankle_x)
        hip_width = abs(left_hip_x - right_hip_x)
        
        # Normalize by hip width (typical body proportion)
        if hip_width > 0.01:  # Avoid division by zero
            normalized_stance = ankle_width / hip_width
        else:
            normalized_stance = ankle_width
        
        # Badminton players typically have stance 2-3x hip width
        is_wide = normalized_stance > 1.5
        
        # Confidence based on how wide the stance is
        confidence = min(1.0, (normalized_stance - 1.0) / 2.0)
        
        return is_wide, max(0.0, confidence) if is_wide else 0.0
    
    def detect_racket_holding_pose(self, landmarks: List[Dict]) -> Tuple[bool, float]:
        """
        Detect asymmetric arm positions (one arm extended, typical when holding racket).
        
        Returns:
            (is_racket_pose, confidence_score)
        """
        if not landmarks or len(landmarks) < 17:
            return False, 0.0
        
        # Get wrist and elbow positions
        left_elbow_x, left_elbow_y, _ = self._get_landmark_coords(landmarks, 13)
        right_elbow_x, right_elbow_y, _ = self._get_landmark_coords(landmarks, 14)
        left_wrist_x, left_wrist_y, _ = self._get_landmark_coords(landmarks, 15)
        right_wrist_x, right_wrist_y, _ = self._get_landmark_coords(landmarks, 16)
        
        # Calculate arm extension (elbow to wrist distance)
        left_extension = np.sqrt((left_wrist_x - left_elbow_x)**2 + 
                                 (left_wrist_y - left_elbow_y)**2)
        right_extension = np.sqrt((right_wrist_x - right_elbow_x)**2 + 
                                  (right_wrist_y - right_elbow_y)**2)
        
        # Check for asymmetry (one arm more extended than the other)
        asymmetry = abs(left_extension - right_extension)
        
        is_asymmetric = asymmetry > self.arm_asymmetry_threshold
        
        # Confidence based on degree of asymmetry
        confidence = min(1.0, asymmetry / (self.arm_asymmetry_threshold * 2))
        
        return is_asymmetric, confidence if is_asymmetric else 0.0
    
    def calculate_badminton_score(self, landmarks_list: List[List[Dict]]) -> float:
        """
        Calculate overall badminton pose score from multiple frames.
        
        Args:
            landmarks_list: List of landmark sets from multiple frames
            
        Returns:
            score: 0.0 to 1.0, where higher means more likely badminton
        """
        if not landmarks_list:
            return 0.0
        
        overhead_scores = []
        stance_scores = []
        racket_scores = []
        
        for landmarks in landmarks_list:
            if landmarks:  # Only process if landmarks detected
                overhead_detected, overhead_conf = self.detect_overhead_motion(landmarks)
                stance_detected, stance_conf = self.detect_wide_stance(landmarks)
                racket_detected, racket_conf = self.detect_racket_holding_pose(landmarks)
                
                overhead_scores.append(overhead_conf)
                stance_scores.append(stance_conf)
                racket_scores.append(racket_conf)
        
        # If no valid detections, return low score
        if not overhead_scores:
            return 0.0
        
        # Calculate average scores
        avg_overhead = np.mean(overhead_scores) if overhead_scores else 0.0
        avg_stance = np.mean(stance_scores) if stance_scores else 0.0
        avg_racket = np.mean(racket_scores) if racket_scores else 0.0
        
        # Weighted combination
        # Overhead motion is strongest indicator (50%)
        # Racket pose is second (30%)
        # Wide stance is supporting evidence (20%)
        final_score = (
            0.5 * avg_overhead +
            0.3 * avg_racket +
            0.2 * avg_stance
        )
        
        return min(1.0, final_score)
    
    def is_badminton_video(self, landmarks_list: List[List[Dict]], 
                          threshold: float = 0.35) -> Tuple[bool, float, Dict[str, float]]:
        """
        Determine if video contains badminton based on pose analysis.
        
        Args:
            landmarks_list: List of landmark sets from video frames
            threshold: Minimum score to classify as badminton
            
        Returns:
            (is_badminton, confidence, details)
        """
        score = self.calculate_badminton_score(landmarks_list)
        
        # Calculate individual component scores for debugging
        overhead_scores = []
        stance_scores = []
        racket_scores = []
        
        for landmarks in landmarks_list:
            if landmarks:
                _, overhead_conf = self.detect_overhead_motion(landmarks)
                _, stance_conf = self.detect_wide_stance(landmarks)
                _, racket_conf = self.detect_racket_holding_pose(landmarks)
                
                overhead_scores.append(overhead_conf)
                stance_scores.append(stance_conf)
                racket_scores.append(racket_conf)
        
        details = {
            "overall_score": score,
            "overhead_score": np.mean(overhead_scores) if overhead_scores else 0.0,
            "stance_score": np.mean(stance_scores) if stance_scores else 0.0,
            "racket_score": np.mean(racket_scores) if racket_scores else 0.0,
            "frames_analyzed": len([l for l in landmarks_list if l])
        }
        
        return score >= threshold, score, details
