"""
Visualization Utilities
Drawing functions for video annotation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


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


def draw_info_overlay(
    frame: np.ndarray,
    timestamp: float,
    frame_idx: int,
    alerts_active: list
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


def generate_id_mapping_snapshot(
    frame: np.ndarray,
    detections: List[dict],
    spatial,
    output_path: str
):
    """
    Generate a student ID mapping snapshot showing ALL known seated students
    using their stored seat positions, not just current-frame detections.

    Args:
        frame: The video frame to annotate
        detections: List of detection dicts from that frame
        spatial: SpatialAnalyzer instance with seat/label data
        output_path: Path to save the snapshot image
    """
    snapshot = frame.copy()
    h, w = snapshot.shape[:2]

    num_students = len(spatial.seats)

    # Draw header
    overlay = snapshot.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, snapshot, 0.3, 0, snapshot)

    title = f"Student ID Mapping - {num_students} Students Detected"
    cv2.putText(snapshot, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Draw ALL known seats using stored seat positions
    for tid, seat_pos in spatial.seats.items():
        label = spatial.get_seat_label(tid)
        stability = spatial.get_stability_score(tid)
        cx, cy = int(seat_pos[0]), int(seat_pos[1])

        # Color by stability
        if stability >= 0.8:
            color = (0, 255, 0)      # Green
        elif stability >= 0.6:
            color = (0, 200, 255)    # Yellow
        else:
            color = (0, 100, 255)    # Orange

        # Draw circle marker at seat position
        cv2.circle(snapshot, (cx, cy), 8, color, -1)
        cv2.circle(snapshot, (cx, cy), 8, (255, 255, 255), 2)

        # Draw label
        text = f"{label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_x = cx - tw // 2
        label_y = cy - 15

        cv2.rectangle(snapshot, (label_x - 2, label_y - th - 2),
                      (label_x + tw + 2, label_y + 2), color, -1)
        cv2.putText(snapshot, text, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw legend at bottom-left
    legend_y = h - 80
    overlay2 = snapshot.copy()
    cv2.rectangle(overlay2, (0, legend_y - 10), (280, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.6, snapshot, 0.4, 0, snapshot)

    cv2.putText(snapshot, "Legend:", (10, legend_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.circle(snapshot, (20, legend_y + 35), 5, (0, 255, 0), -1)
    cv2.putText(snapshot, "High Stability (>0.8)", (35, legend_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.circle(snapshot, (20, legend_y + 55), 5, (0, 200, 255), -1)
    cv2.putText(snapshot, "Medium Stability (0.6-0.8)", (35, legend_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(output_path, snapshot)
    return num_students
