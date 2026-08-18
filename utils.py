"""
Utility functions for the ANPR (Automatic Number Plate Recognition) system.

This module provides shared helpers for text cleaning, plate validation,
OCR error correction, and drawing GUI overlays on video frames.
"""

import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration constants (can be overridden from main.py)
# ---------------------------------------------------------------------------
OCR_CONFIDENCE_THRESHOLD = 0.45
FUZZY_MATCH_THRESHOLD = 85
LOG_DUPLICATE_INTERVAL_SECONDS = 30

# Common OCR misreads mapped to correct characters (applied on digit positions)
OCR_CORRECTIONS = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "G": "6",
    "Z": "2",
    "Q": "0",
    "D": "0",
}

# Indian number plate pattern (flexible): e.g. MH12DE1433, DL4CNA1234
INDIAN_PLATE_PATTERN = re.compile(
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"
)


def get_timestamp() -> Tuple[str, str]:
    """
    Return current date and time as formatted strings.

    Returns:
        Tuple of (date_string, time_string) in YYYY-MM-DD and HH:MM:SS format.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")


def clean_plate_text(raw_text: str) -> str:
    """
    Clean OCR output by removing special characters and converting to uppercase.

    Args:
        raw_text: Raw string returned by the OCR engine.

    Returns:
        Cleaned alphanumeric plate string in uppercase.
    """
    if not raw_text:
        return ""

    # Keep only letters and digits
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text)
    return cleaned.upper()


def apply_ocr_corrections(plate_text: str) -> str:
    """
    Correct common OCR mistakes on likely digit positions in a plate string.

    Indian plates typically follow: XX##XXX#### (state, district, series, number).
    The last 4 characters and district digits are usually numeric.

    Args:
        plate_text: Cleaned plate string.

    Returns:
        Plate string with common OCR substitutions applied.
    """
    if len(plate_text) < 4:
        return plate_text

    chars = list(plate_text)

    # Last 4 characters are almost always digits
    for i in range(max(0, len(chars) - 4), len(chars)):
        if chars[i] in OCR_CORRECTIONS:
            chars[i] = OCR_CORRECTIONS[chars[i]]

    # District code digits (positions 2-3 after state code) when long enough
    if len(chars) >= 4:
        for i in range(2, min(4, len(chars))):
            if chars[i] in OCR_CORRECTIONS:
                chars[i] = OCR_CORRECTIONS[chars[i]]

    return "".join(chars)


def is_valid_plate(plate_text: str) -> bool:
    """
    Validate whether a string resembles a valid Indian number plate.

    Args:
        plate_text: Cleaned plate string.

    Returns:
        True if the plate matches the expected pattern, False otherwise.
    """
    if not plate_text or len(plate_text) < 6:
        return False
    return bool(INDIAN_PLATE_PATTERN.match(plate_text))


def draw_info_panel(
    frame: np.ndarray,
    info: Dict[str, str],
    access_granted: bool,
    confidence: float,
    panel_x: int = 10,
    panel_y: int = 10,
) -> np.ndarray:
    """
    Draw a semi-transparent information panel on the video frame.

    Args:
        frame: BGR image from the webcam.
        info: Dictionary of vehicle details to display.
        access_granted: True for green (granted), False for red (denied).
        confidence: OCR confidence score (0.0 to 1.0).
        panel_x: Left x-coordinate of the panel.
        panel_y: Top y-coordinate of the panel.

    Returns:
        Frame with overlay drawn.
    """
    overlay = frame.copy()
    color = (0, 200, 0) if access_granted else (0, 0, 220)
    status_text = "ACCESS GRANTED" if access_granted else "ACCESS DENIED"

    lines = [
        f"Plate Number    : {info.get('NumberPlate', 'N/A')}",
        f"Owner Name      : {info.get('OwnerName', 'Unknown')}",
        f"Vehicle Model   : {info.get('VehicleModel', 'N/A')}",
        f"Vehicle Type    : {info.get('VehicleType', 'N/A')}",
        f"Registered State: {info.get('RegisteredState', 'N/A')}",
        f"Registered Date : {info.get('RegisteredDate', 'N/A')}",
        f"Access          : {info.get('Access', 'N/A')}",
        f"OCR Confidence  : {confidence:.1%}",
        f"Status          : {status_text}",
    ]

    line_height = 28
    panel_width = 520
    panel_height = line_height * len(lines) + 20

    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (30, 30, 30),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        color,
        2,
    )

    for idx, line in enumerate(lines):
        y = panel_y + 25 + idx * line_height
        text_color = color if "Status" in line else (255, 255, 255)
        cv2.putText(
            frame,
            line,
            (panel_x + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_plate_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    access_granted: bool,
    label: str = "",
) -> np.ndarray:
    """
    Draw a colored bounding box around a detected number plate.

    Args:
        frame: BGR image.
        bbox: (x, y, width, height) of the plate region.
        access_granted: Green if True, red if False.
        label: Optional text label above the box.

    Returns:
        Frame with bounding box drawn.
    """
    x, y, w, h = bbox
    color = (0, 200, 0) if access_granted else (0, 0, 220)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    if label:
        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame


def draw_center_message(
    frame: np.ndarray,
    message: str,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """
    Draw a centered status message on the frame (e.g. 'Scanning...').

    Args:
        frame: BGR image.
        message: Text to display.
        color: BGR color tuple.

    Returns:
        Frame with message drawn.
    """
    h, w = frame.shape[:2]
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    x = (w - text_size[0]) // 2
    y = (h + text_size[1]) // 2
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return frame


def draw_help_text(frame: np.ndarray) -> np.ndarray:
    """
    Draw keyboard shortcut help at the bottom of the frame.

    Args:
        frame: BGR image.

    Returns:
        Frame with help text drawn.
    """
    help_text = "S: Scan | L: Show Logs | C: Clear | Q: Quit"
    cv2.putText(
        frame,
        help_text,
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return frame


def create_unknown_vehicle_info(plate: str) -> Dict[str, str]:
    """
    Build a display dictionary for an unrecognized vehicle.

    Args:
        plate: Recognized plate number (may not be in database).

    Returns:
        Dictionary with unknown vehicle fields populated.
    """
    return {
        "NumberPlate": plate,
        "RegisteredDate": "N/A",
        "RegisteredState": "N/A",
        "OwnerName": "Unknown Vehicle",
        "VehicleType": "N/A",
        "VehicleModel": "N/A",
        "Access": "Denied",
    }
