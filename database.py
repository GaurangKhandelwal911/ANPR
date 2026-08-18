"""
Vehicle database module for the ANPR system.

Loads vehicle records from a CSV file and provides exact and fuzzy
plate lookup using thefuzz library.
"""

import os
from typing import Dict, List, Optional

import pandas as pd
from thefuzz import fuzz, process

from utils import FUZZY_MATCH_THRESHOLD


class VehicleDatabase:
    """
    Manages the vehicle registration database stored in CSV format.

    Supports exact matching and fuzzy matching for OCR-tolerant lookups.
    """

    REQUIRED_COLUMNS = [
        "NumberPlate",
        "RegisteredDate",
        "RegisteredState",
        "OwnerName",
        "VehicleType",
        "VehicleModel",
        "Access",
    ]

    def __init__(self, csv_path: str = "vehicle_database.csv"):
        """
        Initialize the database and load records from CSV.

        Args:
            csv_path: Path to the vehicle_database.csv file.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV is empty, corrupted, or missing columns.
        """
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self.plate_list: List[str] = []
        self.load_database()

    def load_database(self) -> None:
        """
        Load and validate the vehicle database from CSV.

        Raises:
            FileNotFoundError: If CSV file is missing.
            ValueError: If CSV is invalid or empty.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Vehicle database not found: '{self.csv_path}'. "
                "Please ensure vehicle_database.csv exists in the project folder."
            )

        try:
            self.df = pd.read_csv(self.csv_path, dtype=str)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"Vehicle database '{self.csv_path}' is empty or corrupted."
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"Failed to read vehicle database '{self.csv_path}': {exc}"
            ) from exc

        if self.df is None or self.df.empty:
            raise ValueError("Vehicle database contains no records.")

        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(
                f"Vehicle database is missing required columns: {', '.join(missing)}"
            )

        # Normalize plate numbers for consistent lookup
        self.df["NumberPlate"] = (
            self.df["NumberPlate"].astype(str).str.strip().str.upper()
        )
        self.plate_list = self.df["NumberPlate"].tolist()

    def is_empty(self) -> bool:
        """Return True if the database has no records."""
        return self.df is None or self.df.empty

    def search_exact(self, plate: str) -> Optional[Dict[str, str]]:
        """
        Search for an exact plate match in the database.

        Args:
            plate: Normalized plate number string.

        Returns:
            Vehicle record as dict, or None if not found.
        """
        if self.is_empty():
            return None

        plate = plate.strip().upper()
        match = self.df[self.df["NumberPlate"] == plate]

        if match.empty:
            return None

        return self._row_to_dict(match.iloc[0])

    def search_fuzzy(self, plate: str, threshold: int = FUZZY_MATCH_THRESHOLD) -> Optional[Dict[str, str]]:
        """
        Fuzzy search for a plate using thefuzz ratio scoring.

        Example: 'MH12DEI433' can match 'MH12DE1433' at >= 85% similarity.

        Args:
            plate: OCR-recognized plate string.
            threshold: Minimum similarity score (0-100) to accept a match.

        Returns:
            Best matching vehicle record, or None if below threshold.
        """
        if self.is_empty() or not self.plate_list:
            return None

        plate = plate.strip().upper()
        result = process.extractOne(plate, self.plate_list, scorer=fuzz.ratio)

        if result is None:
            return None

        best_match, score = result[0], result[1]
        if score >= threshold:
            return self.search_exact(best_match)

        return None

    def search(self, plate: str, threshold: int = FUZZY_MATCH_THRESHOLD) -> Optional[Dict[str, str]]:
        """
        Search database using exact match first, then fuzzy match.

        Args:
            plate: Plate number to look up.
            threshold: Fuzzy match minimum score.

        Returns:
            Vehicle record dict or None.
        """
        exact = self.search_exact(plate)
        if exact:
            return exact
        return self.search_fuzzy(plate, threshold)

    def get_all_records(self) -> List[Dict[str, str]]:
        """
        Return all vehicle records as a list of dictionaries.

        Returns:
            List of vehicle record dicts.
        """
        if self.is_empty():
            return []
        return [self._row_to_dict(row) for _, row in self.df.iterrows()]

    @staticmethod
    def _row_to_dict(row: pd.Series) -> Dict[str, str]:
        """Convert a DataFrame row to a plain dictionary."""
        return {
            "NumberPlate": str(row.get("NumberPlate", "")),
            "RegisteredDate": str(row.get("RegisteredDate", "")),
            "RegisteredState": str(row.get("RegisteredState", "")),
            "OwnerName": str(row.get("OwnerName", "")),
            "VehicleType": str(row.get("VehicleType", "")),
            "VehicleModel": str(row.get("VehicleModel", "")),
            "Access": str(row.get("Access", "Denied")).strip().capitalize(),
        }
