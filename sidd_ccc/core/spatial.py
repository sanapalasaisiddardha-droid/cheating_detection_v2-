"""
Stage 4: Spatial Analysis
Seat assignment and neighbor graph construction.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Any
from sklearn.cluster import KMeans


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
        self.neighbor_max_distance = config["spatial"]["neighbor_max_distance"]
        self.min_track_length = config["tracking"]["min_track_length"]

        # Track position history: track_id -> list of (center_x, center_y)
        self.track_positions: Dict[int, List[Tuple[float, float]]] = {}

        # Track aspect ratio history for standing detection
        self.track_aspect_ratios: Dict[int, List[float]] = {}

        # Finalized seats: track_id -> (seat_x, seat_y)
        self.seats: Dict[int, Tuple[float, float]] = {}

        # Seat labels: track_id -> "R1_C2" etc
        self.seat_labels: Dict[int, str] = {}

        # Neighbor graph: track_id -> [neighbor_ids]
        self.neighbors: Dict[int, List[int]] = {}

        # Warmup flag
        self.is_warmup = True

        # Count of processed frames (not raw frame index)
        self.processed_frame_count = 0

        # Fixed grid layout
        self.num_rows = config["spatial"].get("num_rows", 4)

        # Standing detection threshold
        self.standing_aspect_ratio = config["spatial"].get("standing_aspect_ratio", 1.4)

        # ID stability tracking
        self.position_jump_threshold = config["spatial"]["position_jump_threshold"]
        self.reassignment_interval = config["spatial"]["reassignment_interval"]
        self.stability_scores: Dict[int, float] = {}
        self.flagged_ids: Set[int] = set()
        self.last_reassignment_frame = 0

        # ID merging: multiple tracker IDs -> one canonical student
        self.id_merge_map: Dict[int, int] = {}  # merged_id -> canonical_id
        self.seat_merge_distance = config["spatial"].get("seat_merge_distance", 80)

        # Stale ID pruning: remove IDs not seen in N processed frames
        self.last_seen_frame: Dict[int, int] = {}  # canonical_id -> last processed_frame_count
        self.stale_track_frames = config["spatial"].get("stale_track_frames", 500)

    def update(self, detections: List[Dict[str, Any]], frame_idx: int):
        """
        Update spatial analysis with new detections.

        Args:
            detections: List of detection dictionaries
            frame_idx: Current frame index
        """
        self.processed_frame_count += 1

        # Accumulate positions and check for jumps
        for det in detections:
            track_id = det["track_id"]
            center = det["center"]
            aspect_ratio = det["aspect_ratio"]

            # Resolve to canonical ID if this was merged
            canonical = self.resolve_id(track_id)

            # Check for position jump (log only first time per ID)
            was_flagged = canonical in self.flagged_ids
            if self._check_position_jump(canonical, center) and not was_flagged:
                print(f"[SPATIAL] Warning: Position jump detected for ID {canonical}")

            if canonical not in self.track_positions:
                self.track_positions[canonical] = []
                self.track_aspect_ratios[canonical] = []

            self.track_positions[canonical].append(center)
            self.track_aspect_ratios[canonical].append(aspect_ratio)

            # Track when this ID was last seen
            self.last_seen_frame[canonical] = self.processed_frame_count

            # Update stability score
            self.stability_scores[canonical] = self._compute_stability_score(canonical)

        # Initial warmup assignment
        min_required_frames = max(self.min_track_length + 5, 20)
        if self.processed_frame_count >= min_required_frames and self.is_warmup:
            self._assign_seats()
            self._build_neighbor_graph()
            self.is_warmup = False
            self.last_reassignment_frame = self.processed_frame_count

        # Periodic re-validation (after warmup)
        elif not self.is_warmup:
            frames_since_reassignment = self.processed_frame_count - self.last_reassignment_frame
            if frames_since_reassignment >= self.reassignment_interval:
                self._reassign_seats()
                self.last_reassignment_frame = self.processed_frame_count

    def _assign_seats(self):
        """Assign seat positions based on median positions during warmup."""
        valid_tracks = {
            tid: positions
            for tid, positions in self.track_positions.items()
            if len(positions) >= self.min_track_length
        }

        for track_id, positions in valid_tracks.items():
            positions_array = np.array(positions)
            seat_x = np.median(positions_array[:, 0])
            seat_y = np.median(positions_array[:, 1])
            self.seats[track_id] = (seat_x, seat_y)

        # Merge IDs that occupy the same physical seat
        self._merge_nearby_seats()

        # Generate seat labels using DBSCAN clustering
        self._generate_seat_labels()

    def _generate_seat_labels(self):
        """Generate row/column labels using K-means with fixed row count."""
        if not self.seats:
            return

        max_columns = self.config["spatial"].get("max_columns", 8)

        track_ids = list(self.seats.keys())
        positions = np.array([self.seats[tid] for tid in track_ids])

        # Use K-means on Y-coordinates with fixed number of rows
        n_clusters = min(self.num_rows, len(track_ids))
        y_coords = positions[:, 1].reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(y_coords)
        row_labels = kmeans.labels_

        # Sort cluster centers by Y (top to bottom)
        center_order = np.argsort(kmeans.cluster_centers_.flatten())
        row_mapping = {old: new for new, old in enumerate(center_order)}

        # Force-merge excess seats in rows that exceed max_columns
        for cluster_id in range(n_clusters):
            cluster_mask = row_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) <= max_columns:
                continue

            # Sort by X position, merge closest pairs until within limit
            cluster_x = [(idx, positions[idx, 0]) for idx in cluster_indices]
            cluster_x.sort(key=lambda x: x[1])

            while len(cluster_x) > max_columns:
                # Find the closest adjacent pair
                min_gap = float('inf')
                min_idx = 0
                for k in range(len(cluster_x) - 1):
                    gap = cluster_x[k + 1][1] - cluster_x[k][1]
                    if gap < min_gap:
                        min_gap = gap
                        min_idx = k

                # Merge the closer pair (keep the one with more observations)
                idx_a, _ = cluster_x[min_idx]
                idx_b, _ = cluster_x[min_idx + 1]
                tid_a = track_ids[idx_a]
                tid_b = track_ids[idx_b]

                len_a = len(self.track_positions.get(tid_a, []))
                len_b = len(self.track_positions.get(tid_b, []))

                if len_a >= len_b:
                    canonical, to_merge = tid_a, tid_b
                    cluster_x.pop(min_idx + 1)
                else:
                    canonical, to_merge = tid_b, tid_a
                    cluster_x.pop(min_idx)

                self.id_merge_map[to_merge] = canonical
                self.seats.pop(to_merge, None)

            if len(cluster_indices) > max_columns:
                merged_count = len(cluster_indices) - max_columns
                row_idx = row_mapping[cluster_id]
                print(f"[SPATIAL] Force-merged {merged_count} excess IDs in row {row_idx + 1}")

        # Regenerate with updated seats
        track_ids = list(self.seats.keys())
        if not track_ids:
            return
        positions = np.array([self.seats[tid] for tid in track_ids])

        n_clusters = min(self.num_rows, len(track_ids))
        y_coords = positions[:, 1].reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(y_coords)
        row_labels = kmeans.labels_

        center_order = np.argsort(kmeans.cluster_centers_.flatten())
        row_mapping = {old: new for new, old in enumerate(center_order)}

        # Assign labels
        for i, track_id in enumerate(track_ids):
            row_idx = row_mapping[row_labels[i]]

            # Find column within this row (sort by X)
            same_row_mask = row_labels == row_labels[i]
            same_row_indices = np.where(same_row_mask)[0]
            same_row_x = [(idx, positions[idx, 0]) for idx in same_row_indices]
            same_row_x.sort(key=lambda x: x[1])

            col_idx = next(j for j, (idx, _) in enumerate(same_row_x) if idx == i)

            label = f"R{row_idx + 1}_C{col_idx + 1}"
            self.seat_labels[track_id] = label

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

    def _check_position_jump(self, track_id: int, new_position: Tuple[float, float]) -> bool:
        """
        Check if position jumped suspiciously (possible ID swap).
        Returns True if jump detected.
        """
        if track_id not in self.track_positions or len(self.track_positions[track_id]) == 0:
            return False

        last_pos = self.track_positions[track_id][-1]
        distance = np.linalg.norm(np.array(new_position) - np.array(last_pos))

        if distance > self.position_jump_threshold:
            self.flagged_ids.add(track_id)
            return True
        return False

    def _compute_stability_score(self, track_id: int) -> float:
        """
        Compute stability score based on position variance.
        High variance = low stability = possible ID issues.
        """
        positions = self.track_positions.get(track_id, [])
        if len(positions) < 5:
            return 0.0

        positions_array = np.array(positions[-20:])  # Use last 20 positions
        variance = np.var(positions_array, axis=0).sum()

        # Normalize: low variance = high score
        # Typical seated student variance < 500, high movement > 2000
        score = max(0.0, min(1.0, 1.0 - (variance / 3000)))
        return score

    def resolve_id(self, track_id: int) -> int:
        """Resolve a track ID to its canonical ID (follows merge chain)."""
        seen = set()
        while track_id in self.id_merge_map:
            if track_id in seen:
                break
            seen.add(track_id)
            track_id = self.id_merge_map[track_id]
        return track_id

    def _merge_nearby_seats(self):
        """
        Merge tracker IDs that occupy the same physical seat.
        Keeps the ID with the most position observations as canonical.
        """
        merged = set()
        track_ids = list(self.seats.keys())

        for i, tid_a in enumerate(track_ids):
            if tid_a in merged:
                continue
            for j in range(i + 1, len(track_ids)):
                tid_b = track_ids[j]
                if tid_b in merged:
                    continue

                pos_a = np.array(self.seats[tid_a])
                pos_b = np.array(self.seats[tid_b])
                dist = np.linalg.norm(pos_a - pos_b)

                if dist < self.seat_merge_distance:
                    # Keep the ID with more observations
                    len_a = len(self.track_positions.get(tid_a, []))
                    len_b = len(self.track_positions.get(tid_b, []))

                    if len_a >= len_b:
                        canonical, to_merge = tid_a, tid_b
                    else:
                        canonical, to_merge = tid_b, tid_a

                    self.id_merge_map[to_merge] = canonical
                    merged.add(to_merge)

        # Remove merged IDs from seats
        for mid in merged:
            self.seats.pop(mid, None)

        if merged:
            print(f"[SPATIAL] Merged {len(merged)} duplicate IDs -> {len(self.seats)} unique students")

    def _reassign_seats(self):
        """
        Re-validate seat assignments using recent position data.
        Prunes stale IDs not seen recently and merges duplicates.
        """
        min_stability = self.config["spatial"].get("min_stability_score", 0.7)

        # Step 1: Prune stale IDs (not seen in last N frames)
        stale_ids = []
        for tid in list(self.seats.keys()):
            last_seen = self.last_seen_frame.get(tid, 0)
            if self.processed_frame_count - last_seen > self.stale_track_frames:
                stale_ids.append(tid)
        for tid in stale_ids:
            self.seats.pop(tid, None)
        if stale_ids:
            print(f"[SPATIAL] Pruned {len(stale_ids)} stale IDs -> {len(self.seats)} active students")

        # Step 2: Update seat positions for active canonical IDs
        for track_id, positions in self.track_positions.items():
            # Skip IDs that were already merged into another canonical
            if track_id in self.id_merge_map:
                continue

            if len(positions) < self.min_track_length:
                continue

            # Skip IDs not seen recently
            last_seen = self.last_seen_frame.get(track_id, 0)
            if self.processed_frame_count - last_seen > self.stale_track_frames:
                continue

            stability = self.stability_scores.get(track_id, 0.0)
            if stability < min_stability:
                continue

            # Use recent positions (last 30) for median
            recent_positions = positions[-30:]
            positions_array = np.array(recent_positions)
            new_seat_x = np.median(positions_array[:, 0])
            new_seat_y = np.median(positions_array[:, 1])

            # Only update if seat moved significantly (avoid noise)
            if track_id in self.seats:
                old_seat = self.seats[track_id]
                dist = np.linalg.norm(np.array([new_seat_x, new_seat_y]) - np.array(old_seat))
                if dist < 30:  # Less than 30px change, skip
                    continue

            self.seats[track_id] = (new_seat_x, new_seat_y)

        # Step 3: Merge any duplicate IDs
        self._merge_nearby_seats()

        # Step 4: Rebuild labels and neighbor graph
        self._generate_seat_labels()
        self._build_neighbor_graph()

    def get_flagged_ids(self) -> Set[int]:
        """Return set of IDs flagged for suspicious position jumps."""
        return self.flagged_ids.copy()

    def get_stability_score(self, track_id: int) -> float:
        """Get stability score for a track ID."""
        return self.stability_scores.get(track_id, 0.0)

    def get_seat_label(self, track_id: int) -> str:
        """Get the seat label for a track ID."""
        canonical = self.resolve_id(track_id)
        return self.seat_labels.get(canonical, "Unknown")

    def is_standing(self, track_id: int) -> bool:
        """Check if a track ID currently has a standing aspect ratio."""
        track_id = self.resolve_id(track_id)
        ratios = self.track_aspect_ratios.get(track_id, [])
        if not ratios:
            return False
        # Use recent aspect ratios (last 5 frames) to determine standing
        recent = ratios[-5:]
        avg_ratio = sum(recent) / len(recent)
        return avg_ratio > self.standing_aspect_ratio

    def get_neighbors(self, track_id: int) -> List[int]:
        """Get list of neighbor track IDs for a given track."""
        canonical = self.resolve_id(track_id)
        return self.neighbors.get(canonical, [])

