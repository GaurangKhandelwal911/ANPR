"""
Main entry point for the ANPR (Automatic Number Plate Recognition) system.

Captures live webcam video, detects plates, performs OCR, checks the
vehicle database, enforces access control, and logs all events.

Keyboard Controls:
    S — Trigger manual scan
    L — Show security logs in console
    C — Clear on-screen results
    Q — Quit application

Future scope hooks (commented in code):
    - YOLOv8 plate detection
    - DeepSORT vehicle tracking
    - Arduino/ESP32 gate control
    - Face recognition for driver verification
    - Firebase/MySQL backend
    - Streamlit/Flask dashboard
    - SMS/Email alerts
    - IP/CCTV camera support
"""

import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from database import VehicleDatabase
from detector import PlateDetector
from logger import SecurityLogger
from ocr import PlateOCR
from utils import (
    create_unknown_vehicle_info,
    draw_center_message,
    draw_help_text,
    draw_info_panel,
    draw_plate_box,
)


class ANPRSystem:
    """
    Orchestrates the full ANPR pipeline: capture → detect → OCR →
    database lookup → access control → logging → GUI overlay.
    """

    WINDOW_NAME = "ANPR System — Automatic Number Plate Recognition"
    CAMERA_INDEX = 0

    def __init__(self):
        """Initialize all ANPR subsystems."""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.script_dir)

        self.detector: Optional[PlateDetector] = None
        self.ocr_engine: Optional[PlateOCR] = None
        self.database: Optional[VehicleDatabase] = None
        self.logger: Optional[SecurityLogger] = None
        self.cap: Optional[cv2.VideoCapture] = None

        # Current scan results displayed on screen
        self.current_info: Optional[Dict[str, str]] = None
        self.current_confidence: float = 0.0
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.access_granted: bool = False
        self.status_message: str = "Press S to Scan | Point camera at number plate"
        self.show_logs_flag: bool = False
        self.auto_scan: bool = True
        self.scan_cooldown: float = 2.0
        self.last_scan_time: float = 0.0

    def initialize(self) -> bool:
        """
        Initialize camera, detector, OCR, database, and logger.

        Returns:
            True if all components initialized successfully.
        """
        print("=" * 60)
        print("  ANPR System — Initializing...")
        print("=" * 60)

        # --- Database ---
        try:
            db_path = os.path.join(self.script_dir, "vehicle_database.csv")
            self.database = VehicleDatabase(csv_path=db_path)
            print(f"[Main] Database loaded: {len(self.database.plate_list)} vehicles")
        except (FileNotFoundError, ValueError) as exc:
            print(f"[Main] Database error: {exc}")
            return False

        # --- Logger ---
        try:
            log_path = os.path.join(self.script_dir, "security_log.csv")
            self.logger = SecurityLogger(log_path=log_path)
            print(f"[Main] Security logger ready: {log_path}")
        except Exception as exc:
            print(f"[Main] Logger error: {exc}")
            return False

        # --- Plate Detector (Haar Cascade) ---
        # Future: swap PlateDetector with YOLOv8Detector here
        try:
            cascade_path = os.path.join(self.script_dir, "haarcascade_plate.xml")
            self.detector = PlateDetector(cascade_path=cascade_path)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[Main] Detector error: {exc}")
            return False

        # --- OCR Engine ---
        try:
            self.ocr_engine = PlateOCR(gpu=False)
        except RuntimeError as exc:
            print(f"[Main] OCR error: {exc}")
            return False

        # --- Webcam ---
        # Future: support RTSP/IP camera URL instead of index
        # e.g. cv2.VideoCapture("rtsp://user:pass@ip:554/stream")
        try:
            self.cap = cv2.VideoCapture(self.CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Camera index {self.CAMERA_INDEX} is unavailable. "
                    "Check webcam connection and permissions."
                )
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("[Main] Webcam connected successfully.")
        except RuntimeError as exc:
            print(f"[Main] Camera error: {exc}")
            return False

        print("[Main] All systems ready. Starting live feed...")
        print("=" * 60)
        return True

    def process_scan(self, frame: np.ndarray) -> None:
        """
        Run full detection + OCR + database lookup on a single frame.

        Args:
            frame: Current webcam frame (BGR).
        """
        if self.detector is None or self.ocr_engine is None or self.database is None:
            return

        self.status_message = "Scanning..."
        plates = self.detector.detect(frame)

        if not plates:
            self.status_message = "No plate detected — adjust camera angle"
            self.current_info = None
            self.current_bbox = None
            return

        # Process the largest detected plate
        bbox = plates[0]
        plate_crop = self.detector.extract_plate_region(frame, bbox)

        if plate_crop is None or plate_crop.size == 0:
            self.status_message = "Plate region extraction failed"
            return

        # OCR recognition
        plate_text, confidence = self.ocr_engine.recognize(plate_crop)

        if not plate_text:
            self.status_message = "OCR failed or low confidence — retry scan"
            self.current_bbox = bbox
            self.current_confidence = confidence
            self.current_info = None
            self.access_granted = False
            return

        # Database lookup (exact + fuzzy)
        vehicle_info = self.database.search(plate_text)

        if vehicle_info is None:
            vehicle_info = create_unknown_vehicle_info(plate_text)
            self.status_message = f"Unknown Vehicle: {plate_text}"
        else:
            self.status_message = f"Identified: {vehicle_info['NumberPlate']}"

        access = vehicle_info.get("Access", "Denied").strip().capitalize()
        self.access_granted = access == "Granted"

        self.current_info = vehicle_info
        self.current_confidence = confidence
        self.current_bbox = bbox

        # Log the event (duplicate suppression handled by logger)
        if self.logger:
            logged = self.logger.log_event(plate_text, vehicle_info, confidence)
            if logged:
                print(
                    f"[Scan] {plate_text} | {vehicle_info['OwnerName']} | "
                    f"Access: {access} | Confidence: {confidence:.1%}"
                )

        # Future: trigger gate control via Arduino/ESP32 serial here
        # if self.access_granted:
        #     gate_controller.open_gate()
        # else:
        #     gate_controller.deny_access()

    def clear_results(self) -> None:
        """Clear all on-screen scan results."""
        self.current_info = None
        self.current_bbox = None
        self.current_confidence = 0.0
        self.access_granted = False
        self.status_message = "Screen cleared — Press S to Scan"

    def render_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw GUI overlays on the current frame.

        Args:
            frame: Raw webcam frame.

        Returns:
            Annotated frame ready for display.
        """
        display = frame.copy()

        # Draw plate bounding box if available
        if self.current_bbox is not None:
            label = ""
            if self.current_info:
                label = self.current_info.get("NumberPlate", "")
            draw_plate_box(display, self.current_bbox, self.access_granted, label)

        # Draw info panel when vehicle is identified or OCR succeeded
        if self.current_info is not None:
            draw_info_panel(
                display,
                self.current_info,
                self.access_granted,
                self.current_confidence,
            )
        elif self.current_confidence > 0 and self.current_bbox is not None:
            # Show partial info when OCR had low confidence
            draw_center_message(
                display,
                f"Low OCR Confidence: {self.current_confidence:.1%}",
                color=(0, 165, 255),
            )

        # Status bar at top-right
        cv2.putText(
            display,
            self.status_message[:60],
            (display.shape[1] - 620, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        draw_help_text(display)
        return display

    def run(self) -> None:
        """Main application loop."""
        if not self.initialize():
            print("\n[Main] Initialization failed. Exiting.")
            sys.exit(1)

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("[Main] Failed to read frame from camera.")
                    self.status_message = "Camera read error — check connection"
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    draw_center_message(blank, "Camera Error", (0, 0, 255))
                    cv2.imshow(self.WINDOW_NAME, blank)
                else:
                    # Auto-scan at intervals when enabled
                    now = time.time()
                    if self.auto_scan and (now - self.last_scan_time) >= self.scan_cooldown:
                        self.process_scan(frame)
                        self.last_scan_time = now

                    display = self.render_frame(frame)
                    cv2.imshow(self.WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q")):
                    print("[Main] Quit requested.")
                    break
                elif key in (ord("s"), ord("S")):
                    if ret and frame is not None:
                        print("[Main] Manual scan triggered.")
                        self.process_scan(frame)
                        self.last_scan_time = time.time()
                elif key in (ord("l"), ord("L")):
                    if self.logger:
                        self.logger.display_logs_in_console()
                elif key in (ord("c"), ord("C")):
                    self.clear_results()
                    print("[Main] Screen cleared.")

        except KeyboardInterrupt:
            print("\n[Main] Interrupted by user.")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release camera and destroy OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[Main] Resources released. Goodbye!")


def main():
    """Entry point — run the ANPR system."""
    system = ANPRSystem()
    system.run()


if __name__ == "__main__":
    main()
