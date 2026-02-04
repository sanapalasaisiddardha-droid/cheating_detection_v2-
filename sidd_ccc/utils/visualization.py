"""
Visualization Utilities
Drawing functions for video annotation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Any


def draw_detection(
    frame: np.ndarray,
    bbox: np.ndarray,
    track_id: int,
    color: Tuple[int, int, int],
    label: Optional[str] = None
):
    """
    Draw bounding box and label for a detection.

    Args:
        frame: Frame to draw on (modified in place)
        bbox: Bounding box [x1, y1, x2, y2]
        track_id: Track identifier
        color: BGR color tuple
        label: Optional label string (uses track_id if not provided)
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Prepare label text
    if label:
        text = f"ID:{track_id} {label}"
    else:
        text = f"ID:{track_id}"

    # Draw label background
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - 10),
        (x1 + text_width + 4, y1),
        color,
        -1
    )

    # Draw label text
    cv2.putText(
        frame,
        text,
        (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


def draw_head_pose_arrow(
    frame: np.ndarray,
    nose_point: Tuple[float, float],
    direction_vector: np.ndarray,
    length: int = 60,
    color: Tuple[int, int, int] = (0, 255, 0)
):
    """
    Draw arrow indicating head direction.

    Args:
        frame: Frame to draw on (modified in place)
        nose_point: Starting point (nose position)
        direction_vector: 2D unit direction vector
        length: Arrow length in pixels
        color: BGR color tuple (default green)
    """
    if direction_vector is None or nose_point is None:
        return

    start_x, start_y = int(nose_point[0]), int(nose_point[1])
    end_x = int(start_x + direction_vector[0] * length)
    end_y = int(start_y + direction_vector[1] * length)

    # Draw arrow
    cv2.arrowedLine(
        frame,
        (start_x, start_y),
        (end_x, end_y),
        color,
        2,
        tipLength=0.3
    )


def draw_pair_line(
    frame: np.ndarray,
    center_a: Tuple[float, float],
    center_b: Tuple[float, float],
    color: Tuple[int, int, int],
    thickness: int = 2
):
    """
    Draw line connecting two persons.

    Args:
        frame: Frame to draw on (modified in place)
        center_a: Center of person A
        center_b: Center of person B
        color: BGR color tuple
        thickness: Line thickness
    """
    pt_a = (int(center_a[0]), int(center_a[1]))
    pt_b = (int(center_b[0]), int(center_b[1]))

    cv2.line(frame, pt_a, pt_b, color, thickness)


def draw_info_overlay(
    frame: np.ndarray,
    timestamp: float,
    frame_idx: int,
    alerts_active: List[Any]
):
    """
    Draw information overlay at the top of the frame.

    Args:
        frame: Frame to draw on (modified in place)
        timestamp: Current timestamp in seconds
        frame_idx: Current frame index
        alerts_active: List of active alerts (for count display)
    """
    h, w = frame.shape[:2]

    # Draw semi-transparent bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 35), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Format timestamp
    minutes = int(timestamp // 60)
    seconds = timestamp % 60
    time_str = f"{minutes:02d}:{seconds:05.2f}"

    # Draw text
    info_text = f"Time: {time_str} | Frame: {frame_idx}"
    if alerts_active:
        info_text += f" | Alerts: {len(alerts_active)}"

    cv2.putText(
        frame,
        info_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


def annotate_evidence_frame(
    frame: np.ndarray,
    pair_detections: List[dict],
    pair_features: dict,
    score: float,
    timestamp: float
) -> np.ndarray:
    """
    Create fully annotated evidence frame.

    Args:
        frame: Original frame
        pair_detections: Detection dicts for the pair
        pair_features: Pair features dictionary
        score: Confidence score
        timestamp: Timestamp in seconds

    Returns:
        Annotated frame copy
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw red bounding boxes for the pair
    for det in pair_detections:
        draw_detection(annotated, det["bbox"], det["track_id"], (0, 0, 255))

    # Draw connecting line between the pair
    if len(pair_detections) >= 2:
        draw_pair_line(
            annotated,
            pair_detections[0]["center"],
            pair_detections[1]["center"],
            (0, 0, 255),
            3
        )

    # Draw header overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 100), -1)
    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

    # Add evidence text
    text1 = f"EVIDENCE FRAME - Score: {score:.2f}"
    text2 = f"Time: {timestamp:.2f}s"

    cv2.putText(annotated, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Add behavior indicators
    behaviors = []
    if pair_features.get("mutual_gaze"):
        behaviors.append("MUTUAL GAZE")
    if pair_features.get("a_looking_at_b") or pair_features.get("b_looking_at_a"):
        behaviors.append("LOOKING AT PEER")
    if pair_features.get("a_whispering_toward_b") or pair_features.get("b_whispering_toward_a"):
        behaviors.append("WHISPERING")
    if pair_features.get("a_showing_paper") or pair_features.get("b_showing_paper"):
        behaviors.append("SHOWING PAPER")

    # Draw behavior labels
    y_offset = h - 30
    for behavior in behaviors:
        cv2.putText(
            annotated,
            behavior,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
        y_offset -= 20

    return annotated


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    confidence_threshold: float = 0.3
):
    """
    Draw body skeleton from COCO keypoints.

    Args:
        frame: Frame to draw on (modified in place)
        keypoints: Array of shape (17, 3) with x, y, confidence
        color: BGR color tuple
        confidence_threshold: Minimum confidence to draw a keypoint
    """
    # COCO skeleton connections
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

    # Draw connections
    for start_idx, end_idx in skeleton:
        if (keypoints[start_idx][2] > confidence_threshold and
            keypoints[end_idx][2] > confidence_threshold):

            start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))

            cv2.line(frame, start_pt, end_pt, color, 2)

    # Draw keypoints
    for i, kpt in enumerate(keypoints):
        if kpt[2] > confidence_threshold:
            pt = (int(kpt[0]), int(kpt[1]))
            cv2.circle(frame, pt, 4, color, -1)
