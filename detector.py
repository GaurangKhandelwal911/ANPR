"""
Number plate detection module for the ANPR system.

Uses OpenCV Haar Cascade classifier to detect license plate regions
in video frames. Designed for easy future upgrade to YOLOv8.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


class PlateDetector:
    """
    Detects number plate regions in images using Haar Cascade.

    Future scope: Replace detect() internals with YOLOv8 inference
    while keeping the same public interface.
    """

    def __init__(
        self,
        cascade_path: str = "haarcascade_plate.xml",
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (60, 20),
    ):
        """
        Load the Haar Cascade classifier for plate detection.

        Args:
            cascade_path: Path to haarcascade_plate.xml.
            scale_factor: Parameter for detectMultiScale (image pyramid step).
            min_neighbors: Minimum neighbors for detection grouping.
            min_size: Minimum plate width and height in pixels.

        Raises:
            FileNotFoundError: If cascade XML is not found.
            RuntimeError: If cascade fails to load.
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade = self._load_cascade(cascade_path)

    def _load_cascade(self, cascade_path: str) -> cv2.CascadeClassifier:
        """
        Load Haar cascade from project file or OpenCV built-in data.

        Args:
            cascade_path: Preferred local cascade file path.

        Returns:
            Loaded CascadeClassifier instance.

        Raises:
            FileNotFoundError: If no cascade file is available.
            RuntimeError: If classifier fails to initialize.
        """
        # Resolve path relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(script_dir, cascade_path)

        paths_to_try = [local_path]

        # Fallback to OpenCV bundled Russian plate cascade
        opencv_cascade = os.path.join(
            cv2.data.haarcascades, "haarcascade_russian_plate_number.xml"
        )
        paths_to_try.append(opencv_cascade)

        for path in paths_to_try:
            if os.path.exists(path):
                if not hasattr(cv2, "CascadeClassifier"):
                    raise RuntimeError(
                        "OpenCV CascadeClassifier is unavailable. "
                        "Install opencv-contrib-python: pip install opencv-contrib-python"
                    )
                classifier = cv2.CascadeClassifier(path)
                if not classifier.empty():
                    print(f"[Detector] Loaded Haar cascade: {path}")
                    return classifier

        raise FileNotFoundError(
            f"Haar cascade not found. Expected '{cascade_path}' in project folder "
            "or OpenCV built-in haarcascade_russian_plate_number.xml."
        )

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect number plate bounding boxes in a BGR frame.

        Args:
            frame: Input image from webcam (BGR format).

        Returns:
            List of (x, y, width, height) tuples for each detected plate.
            Returns empty list if no plate is found.
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Histogram equalization improves contrast for cascade detection
        gray = cv2.equalizeHist(gray)

        plates = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        if len(plates) == 0:
            return []

        # Return plates sorted by area (largest first — likely closest/most visible)
        plates_list = [(int(x), int(y), int(w), int(h)) for x, y, w, h in plates]
        plates_list.sort(key=lambda b: b[2] * b[3], reverse=True)
        return plates_list

    def extract_plate_region(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 5
    ) -> Optional[np.ndarray]:
        """
        Crop the plate region from the frame with optional padding.

        Args:
            frame: Full BGR frame.
            bbox: (x, y, width, height) bounding box.
            padding: Extra pixels around the crop.

        Returns:
            Cropped plate image or None if invalid.
        """
        if frame is None or frame.size == 0:
            return None

        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_frame, x + w + padding)
        y2 = min(h_frame, y + h + padding)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()
