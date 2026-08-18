"""
OCR (Optical Character Recognition) module for the ANPR system.

Preprocesses plate images and extracts text using EasyOCR.
Includes text cleaning, OCR error correction, and confidence filtering.
"""

from typing import Optional, Tuple

import cv2
import easyocr
import numpy as np

from utils import (
    OCR_CONFIDENCE_THRESHOLD,
    apply_ocr_corrections,
    clean_plate_text,
    is_valid_plate,
)


class PlateOCR:
    """
    Performs OCR on cropped number plate images using EasyOCR.

    Pipeline: grayscale → Gaussian blur → adaptive threshold → morphology → OCR
    """

    def __init__(
        self,
        languages: Optional[list] = None,
        confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
        gpu: bool = False,
    ):
        """
        Initialize the EasyOCR reader.

        Args:
            languages: OCR language list (default: English).
            confidence_threshold: Minimum confidence to accept a result.
            gpu: Use GPU acceleration if available.

        Raises:
            RuntimeError: If EasyOCR fails to initialize.
        """
        self.confidence_threshold = confidence_threshold
        self.languages = languages or ["en"]

        try:
            print("[OCR] Initializing EasyOCR (first run may download models)...")
            self.reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)
            print("[OCR] EasyOCR ready.")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize EasyOCR: {exc}") from exc

    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a plate crop for improved OCR accuracy.

        Steps:
            1. Convert to grayscale
            2. Apply Gaussian blur to reduce noise
            3. Adaptive thresholding for binarization
            4. Morphological closing to connect broken characters

        Args:
            plate_image: BGR or grayscale plate crop.

        Returns:
            Preprocessed binary image.
        """
        if plate_image is None or plate_image.size == 0:
            return np.array([])

        # Step 1: Grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()

        # Step 2: Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 3: Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        # Step 4: Morphology — close small gaps in characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return morphed

    def recognize(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR on a plate image and return cleaned text with confidence.

        Tries both original color crop and preprocessed binary image,
        returning the result with the highest confidence.

        Args:
            plate_image: Cropped plate region (BGR).

        Returns:
            Tuple of (plate_text, confidence). Returns ("", 0.0) on failure.
        """
        if plate_image is None or plate_image.size == 0:
            return "", 0.0

        candidates = []

        # Attempt OCR on original and preprocessed versions
        preprocessed = self.preprocess(plate_image)
        images_to_try = [plate_image]
        if preprocessed.size > 0:
            images_to_try.append(preprocessed)

        for img in images_to_try:
            try:
                results = self.reader.readtext(img, detail=1, paragraph=False)
            except Exception as exc:
                print(f"[OCR] Recognition error: {exc}")
                continue

            if not results:
                continue

            # Combine all detected text segments
            combined_text = "".join([res[1] for res in results])
            avg_confidence = sum([res[2] for res in results]) / len(results)

            cleaned = clean_plate_text(combined_text)
            corrected = apply_ocr_corrections(cleaned)

            if corrected:
                candidates.append((corrected, avg_confidence))

        if not candidates:
            return "", 0.0

        # Pick the candidate with highest confidence
        best_text, best_conf = max(candidates, key=lambda c: c[1])

        if best_conf < self.confidence_threshold:
            print(
                f"[OCR] Low confidence ({best_conf:.2%}) for '{best_text}' "
                f"— below threshold {self.confidence_threshold:.0%}"
            )
            return "", best_conf

        if not is_valid_plate(best_text):
            print(f"[OCR] Invalid plate format rejected: '{best_text}'")
            return "", best_conf

        return best_text, best_conf

    def recognize_with_display_image(self, plate_image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Recognize plate text and return the preprocessed image for debugging.

        Args:
            plate_image: Cropped plate BGR image.

        Returns:
            Tuple of (text, confidence, preprocessed_image).
        """
        preprocessed = self.preprocess(plate_image)
        text, confidence = self.recognize(plate_image)
        return text, confidence, preprocessed
