"""
Stage 2: Person Detection and Tracking
Uses YOLOv8 for detection and BoT-SORT for multi-object tracking.
"""

import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO


class PersonDetector:
    """Person detection using YOLOv8 with integrated tracking."""

    def __init__(self, config: dict):
        """
        Initialize the person detector.

        Args:
            config: Configuration dictionary with detection settings
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

        # Load YOLO model (auto-downloads on first run)
        model_name = config["detection"]["model"]
        self.model = YOLO(model_name)
        self.model.to(self.device)

        # Store detection parameters
        self.confidence = config["detection"]["confidence"]
        self.iou_threshold = config["detection"]["iou_threshold"]
        self.classes = config["detection"]["classes"]
        self.imgsz = config["detection"]["imgsz"]

        # Store tracking parameters
        self.tracker = config["tracking"]["tracker"]

    def detect_and_track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and track persons in a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detection dictionaries with bbox, track_id, etc.
        """
        # Run detection with tracking
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False
        )

        detections = []

        # Process results (list with one element per image)
        if results and len(results) > 0:
            result = results[0]

            # Check if boxes exist and have track IDs
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes

                # Get all data as numpy arrays
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                for i in range(len(xyxy)):
                    bbox = xyxy[i]
                    x1, y1, x2, y2 = bbox

                    # Calculate center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Calculate aspect ratio (height / width)
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = height / width if width > 0 else 0

                    detection = {
                        "bbox": bbox,
                        "track_id": int(track_ids[i]),
                        "confidence": float(confidences[i]),
                        "center": (center_x, center_y),
                        "aspect_ratio": aspect_ratio,
                        "width": width,
                        "height": height,
                    }
                    detections.append(detection)

        return detections

    def reset(self):
        """Reset the tracker by reloading the model."""
        model_name = self.config["detection"]["model"]
        self.model = YOLO(model_name)
        self.model.to(self.device)
