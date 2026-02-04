"""
Stage 4: Spatial Analysis
Seat assignment, neighbor graph construction, and teacher identification.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import DBSCAN


class SpatialAnalyzer:
    """Analyzes spatial relationships between tracked persons."""

    def __init__(self, config: dict):
        """
        Initialize the spatial analyzer.

        Args:
            config: Configuration dictionary with spatial settings
        """
        self.config = config

        # Get thresholds from config
        self.seat_warmup_frames = config["spatial"]["seat_warmup_frames"]
        self.neighbor_max_distance = config["spatial"]["neighbor_max_distance"]
        self.close_proximity_distance = config["spatial"]["close_proximity_distance"]
        self.teacher_movement_threshold = config["spatial"]["teacher_movement_threshold"]
        self.teacher_aspect_ratio = config["spatial"]["teacher_aspect_ratio"]
        self.min_track_length = config["tracking"]["min_track_length"]

        # Track position history: track_id -> list of (center_x, center_y)
        self.track_positions: Dict[int, List[Tuple[float, float]]] = {}

        # Track aspect ratio history for teacher detection
        self.track_aspect_ratios: Dict[int, List[float]] = {}

        # Finalized seats: track_id -> (seat_x, seat_y)
        self.seats: Dict[int, Tuple[float, float]] = {}

        # Seat labels: track_id -> "R1_C2" etc
        self.seat_labels: Dict[int, str] = {}

        # Neighbor graph: track_id -> [neighbor_ids]
        self.neighbors: Dict[int, List[int]] = {}

        # Teacher ID (None if not identified)
        self.teacher_id: Optional[int] = None

        # Warmup flag
        self.is_warmup = True

        # Count of processed frames (not raw frame index)
        self.processed_frame_count = 0

    def update(self, detections: List[Dict[str, Any]], frame_idx: int):
        """
        Update spatial analysis with new detections.

        Args:
            detections: List of detection dictionaries
            frame_idx: Current frame index
        """
        self.processed_frame_count += 1

        # Accumulate positions during warmup
        for det in detections:
            track_id = det["track_id"]
            center = det["center"]
            aspect_ratio = det["aspect_ratio"]

            if track_id not in self.track_positions:
                self.track_positions[track_id] = []
                self.track_aspect_ratios[track_id] = []

            self.track_positions[track_id].append(center)
            self.track_aspect_ratios[track_id].append(aspect_ratio)

        # After warmup (enough processed frames), lock seat positions
        # Use processed_frame_count to ensure we have enough observations
        min_required_frames = max(self.min_track_length + 5, 20)
        if self.processed_frame_count >= min_required_frames and self.is_warmup:
            self._assign_seats()
            self._detect_teacher()
            self._build_neighbor_graph()
            self.is_warmup = False

    def _assign_seats(self):
        """Assign seat positions based on median positions during warmup."""
        # Filter tracks with enough history
        valid_tracks = {
            tid: positions
            for tid, positions in self.track_positions.items()
            if len(positions) >= self.min_track_length
        }

        # Calculate median position for each track
        for track_id, positions in valid_tracks.items():
            positions_array = np.array(positions)
            seat_x = np.median(positions_array[:, 0])
            seat_y = np.median(positions_array[:, 1])
            self.seats[track_id] = (seat_x, seat_y)

        # Generate seat labels using DBSCAN clustering
        self._generate_seat_labels()

    def _generate_seat_labels(self):
        """Generate row/column labels for seats using DBSCAN clustering."""
        if not self.seats:
            return

        track_ids = list(self.seats.keys())
        positions = np.array([self.seats[tid] for tid in track_ids])

        # Cluster Y-coordinates to find rows
        # Use smaller eps (35) to properly separate desk rows in classroom
        y_coords = positions[:, 1].reshape(-1, 1)
        row_clustering = DBSCAN(eps=35, min_samples=1).fit(y_coords)
        row_labels = row_clustering.labels_

        # Get unique rows sorted by Y coordinate (top to bottom)
        unique_rows = sorted(set(row_labels), key=lambda r: np.mean(y_coords[row_labels == r]))
        row_mapping = {old: new for new, old in enumerate(unique_rows)}

        # Assign labels
        for i, track_id in enumerate(track_ids):
            row_idx = row_mapping[row_labels[i]]

            # Find column within this row
            same_row_mask = row_labels == row_labels[i]
            same_row_indices = np.where(same_row_mask)[0]
            same_row_x = [(idx, positions[idx, 0]) for idx in same_row_indices]
            same_row_x.sort(key=lambda x: x[1])  # Sort by X coordinate

            col_idx = next(j for j, (idx, _) in enumerate(same_row_x) if idx == i)

            label = f"R{row_idx + 1}_C{col_idx + 1}"
            self.seat_labels[track_id] = label

    def _detect_teacher(self):
        """Identify the teacher based on movement and aspect ratio."""
        teacher_candidates = []

        for track_id, positions in self.track_positions.items():
            if len(positions) < self.min_track_length:
                continue

            # Calculate total cumulative displacement
            positions_array = np.array(positions)
            displacements = np.diff(positions_array, axis=0)
            distances = np.linalg.norm(displacements, axis=1)
            total_displacement = np.sum(distances)

            # Calculate average aspect ratio
            aspect_ratios = self.track_aspect_ratios.get(track_id, [])
            avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 0

            # Check teacher criteria
            is_mobile = total_displacement > self.teacher_movement_threshold
            is_standing = avg_aspect_ratio > self.teacher_aspect_ratio

            if is_mobile or is_standing:
                teacher_candidates.append((track_id, total_displacement, avg_aspect_ratio))

        # Select the most mobile person as teacher
        if teacher_candidates:
            # Sort by displacement (most mobile first)
            teacher_candidates.sort(key=lambda x: x[1], reverse=True)
            self.teacher_id = teacher_candidates[0][0]

            # Remove teacher from seats
            if self.teacher_id in self.seats:
                del self.seats[self.teacher_id]
            if self.teacher_id in self.seat_labels:
                del self.seat_labels[self.teacher_id]

    def _build_neighbor_graph(self):
        """Build neighbor graph based on seat proximity."""
        self.neighbors = {tid: [] for tid in self.seats}

        track_ids = list(self.seats.keys())
        positions = {tid: np.array(self.seats[tid]) for tid in track_ids}

        # Compute pairwise distances
        for i, tid_a in enumerate(track_ids):
            for j, tid_b in enumerate(track_ids):
                if i >= j:  # Skip self and already computed pairs
                    continue

                distance = np.linalg.norm(positions[tid_a] - positions[tid_b])

                if distance < self.neighbor_max_distance:
                    self.neighbors[tid_a].append(tid_b)
                    self.neighbors[tid_b].append(tid_a)

    def get_seat_label(self, track_id: int) -> str:
        """Get the seat label for a track ID."""
        return self.seat_labels.get(track_id, "Unknown")

    def is_teacher(self, track_id: int) -> bool:
        """Check if a track ID is the teacher."""
        return track_id == self.teacher_id

    def get_neighbors(self, track_id: int) -> List[int]:
        """Get list of neighbor track IDs for a given track."""
        return self.neighbors.get(track_id, [])

    def get_teacher_position(self, detections: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
        """
        Get the teacher's current position from detections.

        Args:
            detections: List of current frame detections

        Returns:
            Teacher's center position or None if not visible
        """
        if self.teacher_id is None:
            return None

        for det in detections:
            if det["track_id"] == self.teacher_id:
                return det["center"]

        return None
