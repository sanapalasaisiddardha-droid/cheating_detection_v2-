"""
Stage 3: Pose Estimation and Head Pose Calculation
Uses YOLOv8-Pose for body keypoints and solvePnP for head orientation.
"""

import cv2
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO


# COCO keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class PoseEstimator:
    """Body pose estimation using YOLOv8-Pose."""

    def __init__(self, config: dict):
        """
        Initialize the pose estimator.

        Args:
            config: Configuration dictionary with pose settings
        """
        self.config = config

        # Determine device
        device = config["pipeline"]["device"]
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Load YOLOv8-Pose model
        model_name = config["pose"]["model"]
        self.model = YOLO(model_name)
        self.model.to(self.device)

        self.keypoint_confidence = config["pose"]["keypoint_confidence"]

    def estimate(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Estimate body pose for all persons in frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of pose dictionaries with keypoints and bboxes
        """
        results = self.model(frame, verbose=False)
        pose_results = []

        if results and len(results) > 0:
            result = results[0]

            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()

                for i in range(len(keypoints_data)):
                    # Each person's keypoints: shape (17, 3) for x, y, confidence
                    kpts = keypoints_data[i]

                    pose_result = {
                        "keypoints": kpts,  # (17, 3) array: x, y, conf for each point
                        "bbox": boxes[i],
                        "keypoint_names": KEYPOINT_NAMES,
                    }
                    pose_results.append(pose_result)

        return pose_results


class HeadPoseEstimator:
    """Head pose estimation using solvePnP with facial keypoints."""

    def __init__(self, config: dict):
        """
        Initialize the head pose estimator.

        Args:
            config: Configuration dictionary with head_pose settings
        """
        self.config = config
        self.yaw_threshold = config["head_pose"]["yaw_looking_threshold"]
        self.keypoint_confidence = config["pose"]["keypoint_confidence"]

        # 3D model points for a generic face (in mm)
        # These define the geometry of a typical face
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-350.0, -50.0, -135.0),   # Left ear
            (350.0, -50.0, -135.0),    # Right ear
        ], dtype=np.float64)

    def estimate_head_pose(
        self,
        keypoints: np.ndarray,
        frame_shape: Tuple[int, int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Estimate head pose from COCO keypoints using solvePnP.

        Args:
            keypoints: Array of shape (17, 3) with x, y, confidence for each COCO keypoint
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            Dictionary with yaw, pitch, roll, direction_vector, is_looking_sideways
            or None if estimation fails
        """
        # Extract face keypoints from COCO format
        # Indices: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        left_ear = keypoints[3]
        right_ear = keypoints[4]

        # Check confidence for all required points
        face_points = [nose, left_eye, right_eye, left_ear, right_ear]
        for pt in face_points:
            if pt[2] < self.keypoint_confidence:
                return None

        # Build 2D image points array
        image_points = np.array([
            (nose[0], nose[1]),
            (left_eye[0], left_eye[1]),
            (right_eye[0], right_eye[1]),
            (left_ear[0], left_ear[1]),
            (right_ear[0], right_ear[1]),
        ], dtype=np.float64)

        # Check for valid coordinates
        if np.any(image_points < 0) or np.any(np.isnan(image_points)):
            return None

        # Build camera matrix
        height, width = frame_shape[:2]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP (use SQPNP which works with 4+ points, unlike ITERATIVE which needs 6+)
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP
            )

            if not success:
                return None

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Extract Euler angles from rotation matrix
            yaw, pitch, roll = self._rotation_matrix_to_euler(rotation_matrix)

            # Compute head direction unit vector
            yaw_rad = math.radians(yaw)
            pitch_rad = math.radians(pitch)

            direction_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
            direction_y = -math.sin(pitch_rad)

            # Normalize to unit vector
            magnitude = math.sqrt(direction_x**2 + direction_y**2)
            if magnitude > 0:
                direction_x /= magnitude
                direction_y /= magnitude

            direction_vector = np.array([direction_x, direction_y])

            # Determine if looking sideways
            is_looking_sideways = abs(yaw) > self.yaw_threshold

            return {
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "direction_vector": direction_vector,
                "is_looking_sideways": is_looking_sideways,
                "rotation_vector": rotation_vector,
                "translation_vector": translation_vector,
            }

        except cv2.error:
            return None

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract Euler angles (yaw, pitch, roll) from rotation matrix.

        Args:
            R: 3x3 rotation matrix

        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)

        if sy > 1e-6:
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(-R[1, 2], R[1, 1])
            roll = 0

        # Convert to degrees
        yaw = math.degrees(yaw)
        pitch = math.degrees(pitch)
        roll = math.degrees(roll)

        return yaw, pitch, roll
