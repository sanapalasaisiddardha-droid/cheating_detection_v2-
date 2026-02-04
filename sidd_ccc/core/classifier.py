"""
Stage 6: Temporal Classification
Sliding window analysis with multi-criteria filtering.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class TemporalClassifier:
    """Classifies cheating behavior using temporal analysis."""

    def __init__(self, config: dict):
        """
        Initialize the temporal classifier.

        Args:
            config: Configuration dictionary with classifier settings
        """
        self.config = config

        # Calculate window parameters in frames
        sample_fps = config["pipeline"]["sample_fps"]
        self.window_size = int(config["classifier"]["window_size_seconds"] * sample_fps)
        self.window_stride = int(config["classifier"]["window_stride_seconds"] * sample_fps)

        # Scoring weights
        self.weights = config["classifier"]["weights"]

        # Filter thresholds
        self.min_suspicious_windows = config["classifier"]["min_suspicious_windows"]
        self.min_confidence = config["classifier"]["min_confidence"]
        self.require_mutual = config["classifier"]["require_mutual"]
        self.teacher_suppress_distance = config["classifier"]["teacher_suppress_distance"]

        # Feature buffer: pair_id -> list of frame features
        self.pair_feature_buffer: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)

    def add_frame(
        self,
        pair_features: List[Dict[str, Any]],
        frame_idx: int,
        timestamp: float
    ):
        """
        Add frame features to the temporal buffer.

        Args:
            pair_features: List of pair feature dictionaries
            frame_idx: Current frame index
            timestamp: Current timestamp in seconds
        """
        for pf in pair_features:
            pair_id = pf["pair_id"]

            # Attach frame metadata
            feature_entry = {
                **pf,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
            }

            self.pair_feature_buffer[pair_id].append(feature_entry)

    def classify_all(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Run classification on all pairs.

        Returns:
            Dictionary mapping pair_id to classification results
        """
        results = {}

        for pair_id, features in self.pair_feature_buffer.items():
            if len(features) < self.window_size:
                # Not enough data for classification
                results[pair_id] = {
                    "is_cheating": False,
                    "confidence": 0.0,
                    "num_suspicious_windows": 0,
                    "evidence_windows": [],
                    "behavior_summary": {},
                    "reason": "insufficient_data"
                }
                continue

            # Generate sliding windows
            scored_windows = self._generate_and_score_windows(features)

            # Apply multi-criteria filter
            results[pair_id] = self._apply_filter(pair_id, features, scored_windows)

        return results

    def _generate_and_score_windows(
        self,
        features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate sliding windows and score each one.

        Args:
            features: List of per-frame features for a pair

        Returns:
            List of scored window dictionaries
        """
        scored_windows = []
        n_frames = len(features)

        # Slide window
        for start_idx in range(0, n_frames - self.window_size + 1, self.window_stride):
            end_idx = start_idx + self.window_size
            window_frames = features[start_idx:end_idx]

            # Score the window
            score, behavior_counts = self._score_window(window_frames)

            # Get window timestamps
            start_time = window_frames[0]["timestamp"]
            end_time = window_frames[-1]["timestamp"]

            # Track teacher distance during window
            teacher_distances = [
                f.get("teacher_distance", float('inf'))
                for f in window_frames
            ]
            min_teacher_distance = min(teacher_distances)

            scored_windows.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": start_time,
                "end_time": end_time,
                "score": score,
                "behavior_counts": behavior_counts,
                "min_teacher_distance": min_teacher_distance,
            })

        return scored_windows

    def _score_window(
        self,
        window_frames: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, int]]:
        """
        Calculate weighted score for a window.

        Args:
            window_frames: List of frame features in the window

        Returns:
            Tuple of (weighted_score, behavior_counts)
        """
        n = len(window_frames)

        # Count occurrences of each behavior
        counts = {
            "mutual_gaze": 0,
            "sustained_head_turn": 0,
            "whispering_posture": 0,
            "body_lean": 0,
            "hand_activity": 0,
            "paper_movement": 0,
            "proximity": 0,
        }

        for frame in window_frames:
            # Mutual gaze
            if frame.get("mutual_gaze", False):
                counts["mutual_gaze"] += 1

            # Sustained head turn (either person looking at the other)
            if frame.get("a_looking_at_b", False) or frame.get("b_looking_at_a", False):
                counts["sustained_head_turn"] += 1

            # Whispering posture
            if frame.get("a_whispering_toward_b", False) or frame.get("b_whispering_toward_a", False):
                counts["whispering_posture"] += 1

            # Body lean (using shoulder angle deviation from typical)
            student_a = frame.get("student_a", {})
            student_b = frame.get("student_b", {})
            shoulder_a = abs(student_a.get("shoulder_angle", 0))
            shoulder_b = abs(student_b.get("shoulder_angle", 0))
            if shoulder_a > 15 or shoulder_b > 15:  # More than 15 degrees tilt
                counts["body_lean"] += 1

            # Hand activity (either person with hand near face)
            if student_a.get("hand_near_face", False) or student_b.get("hand_near_face", False):
                counts["hand_activity"] += 1

            # Paper movement (showing paper)
            if frame.get("a_showing_paper", False) or frame.get("b_showing_paper", False):
                counts["paper_movement"] += 1

            # Proximity (moved closer than baseline)
            if frame.get("proximity_delta", 0) < 0:
                counts["proximity"] += 1

        # Calculate weighted score
        weighted_score = 0.0
        for behavior, weight in self.weights.items():
            if behavior in counts:
                ratio = counts[behavior] / n
                weighted_score += weight * ratio

        return weighted_score, counts

    def _apply_filter(
        self,
        pair_id: Tuple[int, int],
        features: List[Dict[str, Any]],
        scored_windows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply multi-criteria filter to determine if cheating.

        Args:
            pair_id: The pair identifier
            features: All features for this pair
            scored_windows: List of scored windows

        Returns:
            Classification result dictionary
        """
        # Find suspicious windows (score >= 0.45)
        suspicious_windows = [w for w in scored_windows if w["score"] >= 0.45]

        # Get peak score
        peak_score = max((w["score"] for w in scored_windows), default=0.0)

        # Check if mutual gaze occurred in >30% of any window
        has_mutual = False
        for w in scored_windows:
            mutual_ratio = w["behavior_counts"]["mutual_gaze"] / self.window_size
            if mutual_ratio > 0.3:
                has_mutual = True
                break

        # Check teacher suppression
        teacher_suppressed = False
        for w in suspicious_windows:
            if w["min_teacher_distance"] < self.teacher_suppress_distance:
                teacher_suppressed = True
                break

        # Apply multi-criteria filter
        is_cheating = True
        fail_reason = None

        # Criterion 1: Minimum number of suspicious windows
        if len(suspicious_windows) < self.min_suspicious_windows:
            is_cheating = False
            fail_reason = "insufficient_suspicious_windows"

        # Criterion 2: Peak confidence threshold
        if peak_score < self.min_confidence:
            is_cheating = False
            fail_reason = "below_confidence_threshold"

        # Criterion 3: Require mutual behavior (if enabled)
        if self.require_mutual and not has_mutual:
            is_cheating = False
            fail_reason = "no_mutual_behavior"

        # Criterion 4: Teacher suppression
        if teacher_suppressed:
            is_cheating = False
            fail_reason = "teacher_nearby"

        # Sort evidence windows by score
        evidence_windows = sorted(suspicious_windows, key=lambda w: w["score"], reverse=True)[:5]

        # Build behavior summary
        total_counts = {}
        for w in scored_windows:
            for behavior, count in w["behavior_counts"].items():
                total_counts[behavior] = total_counts.get(behavior, 0) + count

        return {
            "is_cheating": is_cheating,
            "confidence": peak_score,
            "num_suspicious_windows": len(suspicious_windows),
            "evidence_windows": evidence_windows,
            "behavior_summary": total_counts,
            "has_mutual": has_mutual,
            "teacher_suppressed": teacher_suppressed,
            "fail_reason": fail_reason,
        }
