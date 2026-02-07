"""
Stage 5: Behavioral Feature Extraction
Extract per-student and pairwise interaction features.
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional


class BehaviorFeatureExtractor:
    """Extracts behavioral features from pose and head data."""

    def __init__(self, config: dict):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration dictionary with feature thresholds
        """
        self.config = config

        # Load thresholds from config
        self.mutual_gaze_dot_threshold = config["features"]["mutual_gaze_dot_threshold"]
        self.hand_face_distance = config["features"]["hand_face_distance"]
        self.hand_neighbor_distance = config["features"]["hand_neighbor_distance"]
        self.wrist_elevation_threshold = config["features"]["wrist_elevation_threshold"]

        # Baseline positions for body lean detection
        self.baseline_positions: Dict[int, Tuple[float, float]] = {}

    def extract_student_features(
        self,
        detection: Dict[str, Any],
        pose_data: Dict[str, Any],
        head_pose: Dict[str, Any],
        seat_position: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Extract per-student behavioral features.

        Args:
            detection: Detection dictionary with bbox, track_id, center
            pose_data: Pose data with keypoints
            head_pose: Head pose data with yaw, direction_vector
            seat_position: The student's assigned seat position

        Returns:
            Dictionary of student features
        """
        track_id = detection["track_id"]
        keypoints = pose_data["keypoints"]

        # Extract keypoint positions
        nose = keypoints[0][:2] if keypoints[0][2] > 0.3 else None
        left_wrist = keypoints[9][:2] if keypoints[9][2] > 0.3 else None
        right_wrist = keypoints[10][:2] if keypoints[10][2] > 0.3 else None
        left_shoulder = keypoints[5][:2] if keypoints[5][2] > 0.3 else None
        right_shoulder = keypoints[6][:2] if keypoints[6][2] > 0.3 else None

        # Hand near face detection
        hand_near_face = False
        if nose is not None:
            if left_wrist is not None:
                dist_left = np.linalg.norm(np.array(left_wrist) - np.array(nose))
                if dist_left < self.hand_face_distance:
                    hand_near_face = True
            if right_wrist is not None:
                dist_right = np.linalg.norm(np.array(right_wrist) - np.array(nose))
                if dist_right < self.hand_face_distance:
                    hand_near_face = True

        # Wrist elevation detection (elevated = lower Y value = higher in image)
        wrist_elevated = False
        if left_shoulder is not None and right_shoulder is not None:
            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

            if left_wrist is not None:
                if left_wrist[1] < avg_shoulder_y - self.wrist_elevation_threshold:
                    wrist_elevated = True
            if right_wrist is not None:
                if right_wrist[1] < avg_shoulder_y - self.wrist_elevation_threshold:
                    wrist_elevated = True

        # Shoulder angle (body orientation)
        shoulder_angle = 0.0
        if left_shoulder is not None and right_shoulder is not None:
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            shoulder_angle = math.degrees(math.atan2(dy, dx))

        # Update baseline position for body lean tracking
        body_center = detection["center"]
        if track_id not in self.baseline_positions:
            self.baseline_positions[track_id] = body_center

        return {
            "track_id": track_id,
            "head_yaw": head_pose["yaw"],
            "head_direction": head_pose["direction_vector"],
            "is_looking_sideways": head_pose["is_looking_sideways"],
            "hand_near_face": hand_near_face,
            "wrist_positions": {
                "left": tuple(left_wrist) if left_wrist is not None else None,
                "right": tuple(right_wrist) if right_wrist is not None else None,
            },
            "wrist_elevated": wrist_elevated,
            "shoulder_angle": shoulder_angle,
            "body_center": body_center,
            "seat_position": seat_position,
        }

    def extract_pair_features(
        self,
        student_a: Dict[str, Any],
        student_b: Dict[str, Any],
        seat_a: Tuple[float, float],
        seat_b: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Extract pairwise interaction features between two students.

        Args:
            student_a: Student A's features
            student_b: Student B's features
            seat_a: Student A's seat position
            seat_b: Student B's seat position

        Returns:
            Dictionary of pair features
        """
        # Ensure consistent pair ordering
        id_a = student_a["track_id"]
        id_b = student_b["track_id"]
        pair_id = tuple(sorted([id_a, id_b]))

        # Get head directions
        dir_a = student_a["head_direction"]
        dir_b = student_b["head_direction"]

        # Mutual gaze detection (heads facing each other)
        # Negative dot product means vectors point at each other
        mutual_gaze = False
        if dir_a is not None and dir_b is not None:
            dot_product = np.dot(dir_a, dir_b)
            mutual_gaze = dot_product < self.mutual_gaze_dot_threshold

        # Direction from A to B (normalized)
        pos_a = np.array(seat_a)
        pos_b = np.array(seat_b)
        dir_a_to_b = pos_b - pos_a
        distance = np.linalg.norm(dir_a_to_b)
        if distance > 0:
            dir_a_to_b = dir_a_to_b / distance

        # Direction from B to A
        dir_b_to_a = -dir_a_to_b

        # Directional gaze check disabled (gaze_toward_threshold removed)
        a_looking_at_b = False
        b_looking_at_a = False

        # Whispering detection (looking at + hand near face)
        a_whispering_toward_b = a_looking_at_b and student_a["hand_near_face"]
        b_whispering_toward_a = b_looking_at_a and student_b["hand_near_face"]

        # Paper showing detection (wrist elevated + reaching toward neighbor)
        a_showing_paper = self._check_paper_showing(student_a, seat_b)
        b_showing_paper = self._check_paper_showing(student_b, seat_a)

        # Current distance between students
        current_pos_a = np.array(student_a["body_center"])
        current_pos_b = np.array(student_b["body_center"])
        current_distance = np.linalg.norm(current_pos_a - current_pos_b)

        # Baseline distance
        baseline_distance = np.linalg.norm(pos_a - pos_b)

        # Proximity delta (negative = moved closer together)
        proximity_delta = current_distance - baseline_distance

        return {
            "pair_id": pair_id,
            "mutual_gaze": mutual_gaze,
            "a_looking_at_b": a_looking_at_b,
            "b_looking_at_a": b_looking_at_a,
            "a_whispering_toward_b": a_whispering_toward_b,
            "b_whispering_toward_a": b_whispering_toward_a,
            "a_showing_paper": a_showing_paper,
            "b_showing_paper": b_showing_paper,
            "proximity_delta": proximity_delta,
            "current_distance": current_distance,
            "baseline_distance": baseline_distance,
            "student_a": student_a,
            "student_b": student_b,
        }

    def _check_paper_showing(
        self,
        student: Dict[str, Any],
        neighbor_seat: Tuple[float, float]
    ) -> bool:
        """
        Check if student appears to be showing paper to neighbor.

        Args:
            student: Student's features
            neighbor_seat: Neighbor's seat position

        Returns:
            True if showing paper behavior detected
        """
        if not student["wrist_elevated"]:
            return False

        neighbor_pos = np.array(neighbor_seat)
        wrist_positions = student["wrist_positions"]

        for side in ["left", "right"]:
            wrist = wrist_positions.get(side)
            if wrist is not None:
                wrist_pos = np.array(wrist)
                dist = np.linalg.norm(wrist_pos - neighbor_pos)
                if dist < self.hand_neighbor_distance:
                    return True

        return False
