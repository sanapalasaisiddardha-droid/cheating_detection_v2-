"""
Stage 7: Alert Generation
Generate human-readable alerts and save evidence.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple


class AlertGenerator:
    """Generates alerts and evidence from classification results."""

    def __init__(self, config: dict, spatial):
        """
        Initialize the alert generator.

        Args:
            config: Configuration dictionary
            spatial: SpatialAnalyzer instance for seat labels
        """
        self.config = config
        self.spatial = spatial

        # Confidence tier thresholds
        self.tier_very_high = config["alerts"]["tier_very_high"]
        self.tier_high = config["alerts"]["tier_high"]
        self.tier_medium = config["alerts"]["tier_medium"]
        self.max_evidence_frames = config["alerts"]["max_evidence_frames"]

    def generate_alerts(
        self,
        classifications: Dict[Tuple[int, int], Dict[str, Any]],
        frame_buffer: Dict[int, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Generate alert dictionaries from classification results.

        Args:
            classifications: Classification results per pair
            frame_buffer: Buffer of stored frames for evidence

        Returns:
            List of alert dictionaries
        """
        raw_alerts = []

        for pair_id, result in classifications.items():
            if not result["is_cheating"]:
                continue

            track_id_a, track_id_b = pair_id
            confidence = result["confidence"]

            # Get seat labels
            label_a = self.spatial.get_seat_label(track_id_a)
            label_b = self.spatial.get_seat_label(track_id_b)

            # Skip self-pairs (same physical seat)
            if label_a == label_b:
                continue

            # Determine confidence tier
            if confidence >= self.tier_very_high:
                tier = "VERY_HIGH"
            elif confidence >= self.tier_high:
                tier = "HIGH"
            elif confidence >= self.tier_medium:
                tier = "MEDIUM"
            else:
                tier = "LOW"

            # Extract evidence timestamps
            timestamps = []
            for window in result.get("evidence_windows", []):
                timestamps.append((window["start_time"], window["end_time"]))

            # Build behavior description
            behaviors = self._build_behavior_descriptions(result["behavior_summary"])

            alert = {
                "pair": pair_id,
                "seat_labels": tuple(sorted([label_a, label_b])),
                "tier": tier,
                "confidence": confidence,
                "timestamps": timestamps,
                "behaviors": behaviors,
                "num_suspicious_windows": result["num_suspicious_windows"],
                "evidence_windows": result.get("evidence_windows", []),
            }
            raw_alerts.append(alert)

        # Deduplicate: keep only the highest-confidence alert per seat pair
        best_by_seat_pair = {}
        for alert in raw_alerts:
            key = alert["seat_labels"]
            if key not in best_by_seat_pair or alert["confidence"] > best_by_seat_pair[key]["confidence"]:
                best_by_seat_pair[key] = alert

        alerts = list(best_by_seat_pair.values())

        # Sort by confidence (highest first)
        alerts.sort(key=lambda a: a["confidence"], reverse=True)

        return alerts

    def _build_behavior_descriptions(self, behavior_summary: Dict[str, int]) -> List[str]:
        """
        Build human-readable behavior descriptions.

        Args:
            behavior_summary: Dictionary of behavior counts

        Returns:
            List of behavior description strings
        """
        descriptions = []

        behavior_names = {
            "mutual_gaze": "Mutual gaze (looking at each other)",
            "headturn_hand_on_face": "Head turned sideways with hand on face",
            "sustained_head_turn": "Sustained head turn toward peer",
            "whispering_posture": "Whispering posture detected",
            "body_lean": "Body leaning toward peer",
            "hand_activity": "Hand activity near face",
            "paper_movement": "Paper/material sharing gesture",
            "proximity": "Moved closer than normal",
        }

        # Sort by count and include most significant behaviors
        sorted_behaviors = sorted(
            behavior_summary.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for behavior, count in sorted_behaviors:
            if count > 0 and behavior in behavior_names:
                descriptions.append(f"{behavior_names[behavior]} ({count} instances)")

        return descriptions

    def save_evidence_frames(
        self,
        alerts: List[Dict[str, Any]],
        frame_buffer: Dict[int, Dict[str, Any]],
        output_dir: str
    ):
        """
        Save annotated evidence frames for each alert.

        Args:
            alerts: List of alert dictionaries
            frame_buffer: Buffer of stored frames with detection data
            output_dir: Directory to save evidence frames
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, alert in enumerate(alerts):
            pair_id = alert["pair"]
            labels = alert["seat_labels"]

            # Create subdirectory for this pair
            pair_dir = os.path.join(output_dir, f"pair_{labels[0]}_{labels[1]}")
            os.makedirs(pair_dir, exist_ok=True)

            # Find frames closest to evidence timestamps
            evidence_count = 0
            for window in alert.get("evidence_windows", []):
                if evidence_count >= self.max_evidence_frames:
                    break

                # Find a frame from this window
                target_time = (window["start_time"] + window["end_time"]) / 2

                # Find closest frame in buffer by timestamp
                closest_frame_idx = None
                min_diff = float('inf')
                for frame_idx, frame_data in frame_buffer.items():
                    if isinstance(frame_data, dict) and "timestamp" in frame_data:
                        diff = abs(frame_data["timestamp"] - target_time)
                    else:
                        # Fallback for old format
                        diff = abs(frame_idx - int(target_time * 5))
                    if diff < min_diff:
                        min_diff = diff
                        closest_frame_idx = frame_idx

                if closest_frame_idx is not None and closest_frame_idx in frame_buffer:
                    frame_data = frame_buffer[closest_frame_idx]

                    # Handle both old (just frame) and new (dict with frame+detections) format
                    if isinstance(frame_data, dict):
                        frame = frame_data["frame"].copy()
                        detections = frame_data.get("detections", [])
                        actual_timestamp = frame_data.get("timestamp", target_time)
                    else:
                        frame = frame_data.copy()
                        detections = []
                        actual_timestamp = target_time

                    # Add annotation overlay with bounding boxes
                    self._annotate_evidence_frame(
                        frame,
                        alert,
                        window["score"],
                        actual_timestamp,
                        detections,
                        pair_id
                    )

                    # Save frame
                    filename = f"evidence_{evidence_count:02d}_t{actual_timestamp:.1f}s.jpg"
                    filepath = os.path.join(pair_dir, filename)
                    cv2.imwrite(filepath, frame)

                    evidence_count += 1

    def _annotate_evidence_frame(
        self,
        frame: np.ndarray,
        alert: Dict[str, Any],
        score: float,
        timestamp: float,
        detections: List[Dict[str, Any]] = None,
        pair_id: tuple = None
    ):
        """
        Add annotations to an evidence frame with bounding boxes.

        Args:
            frame: Frame to annotate (modified in place)
            alert: Alert dictionary
            score: Window score
            timestamp: Timestamp
            detections: List of detections in this frame
            pair_id: Tuple of track IDs for the flagged pair
        """
        h, w = frame.shape[:2]
        labels = alert["seat_labels"]

        # Draw bounding boxes around the cheating pair
        if detections and pair_id:
            for det in detections:
                tid = det["track_id"]
                bbox = det["bbox"]
                x1, y1, x2, y2 = map(int, bbox)

                if tid in pair_id:
                    # Red box for flagged students
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # Get label for this student
                    idx = pair_id.index(tid)
                    label = labels[idx] if idx < len(labels) else f"ID:{tid}"

                    # Draw label
                    cv2.rectangle(frame, (x1, y1-25), (x1+len(label)*12+10, y1), (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1+5, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw line connecting the pair
            pair_centers = []
            for det in detections:
                if det["track_id"] in pair_id:
                    pair_centers.append(det["center"])
            if len(pair_centers) == 2:
                pt1 = (int(pair_centers[0][0]), int(pair_centers[0][1]))
                pt2 = (int(pair_centers[1][0]), int(pair_centers[1][1]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

        # Draw semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 120), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Add header text
        text1 = f"CHEATING DETECTED: {labels[0]} & {labels[1]}"
        text2 = f"Confidence: {score:.0%} | Time: {timestamp:.1f}s | Tier: {alert['tier']}"

        cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)

        # Add behavior indicators at bottom
        behaviors = alert.get("behaviors", [])[:3]  # Top 3 behaviors
        if behaviors:
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h-70), (w, h), (0, 0, 80), -1)
            cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

            y_pos = h - 50
            for behavior in behaviors:
                # Truncate long behavior text
                short_behavior = behavior.split("(")[0].strip()[:50]
                cv2.putText(frame, f"* {short_behavior}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1)
                y_pos += 20

    def save_json_log(
        self,
        all_frame_data: List[Dict[str, Any]],
        output_path: str
    ):
        """
        Save per-frame JSONL log for audit trail.

        Args:
            all_frame_data: List of per-frame data dictionaries
            output_path: Path to output file
        """
        with open(output_path, 'w') as f:
            for entry in all_frame_data:
                # Convert numpy types to Python types
                clean_entry = self._convert_to_serializable(entry)
                f.write(json.dumps(clean_entry) + '\n')

    def _convert_to_serializable(self, obj):
        """Convert numpy types and other non-serializable objects."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
