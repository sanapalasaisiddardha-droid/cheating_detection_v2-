"""
Stage 1: Video Preprocessing
Extracts frames from video at configured FPS, normalizes brightness.
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Dict, Any


class VideoPreprocessor:
    """Preprocesses video frames with CLAHE enhancement and controlled sampling."""

    def __init__(self, video_path: str, config: dict):
        """
        Initialize the video preprocessor.

        Args:
            video_path: Path to the input video file
            config: Configuration dictionary with preprocessing settings
        """
        self.video_path = video_path
        self.config = config

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Read video properties
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.native_fps if self.native_fps > 0 else 0

        # Get config values
        self.sample_fps = config["pipeline"]["sample_fps"]
        self.resize_width = config["preprocessing"]["resize_width"]
        self.resize_height = config["preprocessing"]["resize_height"]

        # Calculate sample interval (process every Nth frame)
        self.sample_interval = max(1, int(self.native_fps / self.sample_fps))

        # Setup CLAHE if enabled
        self.clahe_enabled = config["preprocessing"]["clahe_enabled"]
        if self.clahe_enabled:
            clip_limit = config["preprocessing"]["clahe_clip_limit"]
            grid_size = config["preprocessing"]["clahe_grid_size"]
            self.clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(grid_size, grid_size)
            )

    def process(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Process video frames using a generator pattern.

        Yields:
            Tuple of (frame_index, timestamp_seconds, processed_frame)
        """
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Only process frames at the sample interval
            if frame_idx % self.sample_interval == 0:
                # Resize frame
                processed = cv2.resize(frame, (self.resize_width, self.resize_height))

                # Apply CLAHE in LAB color space to preserve color
                if self.clahe_enabled:
                    processed = self._apply_clahe(processed)

                # Calculate timestamp
                timestamp = frame_idx / self.native_fps

                yield frame_idx, timestamp, processed

            frame_idx += 1

        # Release video capture when done
        self.cap.release()

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L channel in LAB color space.
        This prevents color distortion while enhancing contrast.

        Args:
            frame: BGR input frame

        Returns:
            Enhanced BGR frame
        """
        # Convert BGR to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Split into L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel only
        l_enhanced = self.clahe.apply(l_channel)

        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def get_info(self) -> Dict[str, Any]:
        """
        Get video information.

        Returns:
            Dictionary with video metadata
        """
        return {
            "fps": self.native_fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "width": self.original_width,
            "height": self.original_height,
            "sample_fps": self.sample_fps,
            "sample_interval": self.sample_interval,
        }

    def __del__(self):
        """Ensure video capture is released."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
