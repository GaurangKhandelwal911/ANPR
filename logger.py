"""
Security logging module for the ANPR system.

Maintains a CSV log of all vehicle access events with duplicate
suppression within a configurable time window.
"""

import csv
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

from utils import LOG_DUPLICATE_INTERVAL_SECONDS, get_timestamp


class SecurityLogger:
    """
    Logs vehicle access events to security_log.csv.

    Prevents duplicate entries for the same plate within 30 seconds
    (configurable via LOG_DUPLICATE_INTERVAL_SECONDS in utils.py).
    """

    LOG_COLUMNS = [
        "Date",
        "Time",
        "NumberPlate",
        "RegisteredState",
        "OwnerName",
        "VehicleType",
        "VehicleModel",
        "AccessStatus",
        "Confidence",
    ]

    def __init__(self, log_path: str = "security_log.csv"):
        """
        Initialize the logger and ensure the log file exists.

        Args:
            log_path: Path to the security log CSV file.
        """
        self.log_path = log_path
        self.recent_logs: Dict[str, datetime] = {}
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Create the log file with headers if it does not exist."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.LOG_COLUMNS)
                writer.writeheader()

    def _is_duplicate(self, plate: str) -> bool:
        """
        Check if this plate was logged within the duplicate interval.

        Args:
            plate: Normalized plate number.

        Returns:
            True if logging should be skipped as a duplicate.
        """
        plate = plate.strip().upper()
        now = datetime.now()

        # Remove expired entries from tracking dict
        expired = [
            p
            for p, ts in self.recent_logs.items()
            if now - ts > timedelta(seconds=LOG_DUPLICATE_INTERVAL_SECONDS)
        ]
        for p in expired:
            del self.recent_logs[p]

        if plate in self.recent_logs:
            elapsed = (now - self.recent_logs[plate]).total_seconds()
            if elapsed < LOG_DUPLICATE_INTERVAL_SECONDS:
                return True

        return False

    def log_event(
        self,
        plate: str,
        vehicle_info: Dict[str, str],
        confidence: float,
    ) -> bool:
        """
        Append an access event to the security log.

        Args:
            plate: Recognized plate number.
            vehicle_info: Vehicle details from database or unknown defaults.
            confidence: OCR confidence score.

        Returns:
            True if logged successfully, False if skipped as duplicate.
        """
        plate = plate.strip().upper()

        if self._is_duplicate(plate):
            return False

        date_str, time_str = get_timestamp()
        access_status = vehicle_info.get("Access", "Denied")

        row = {
            "Date": date_str,
            "Time": time_str,
            "NumberPlate": plate,
            "RegisteredState": vehicle_info.get("RegisteredState", "N/A"),
            "OwnerName": vehicle_info.get("OwnerName", "Unknown"),
            "VehicleType": vehicle_info.get("VehicleType", "N/A"),
            "VehicleModel": vehicle_info.get("VehicleModel", "N/A"),
            "AccessStatus": access_status,
            "Confidence": f"{confidence:.2f}",
        }

        try:
            with open(self.log_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.LOG_COLUMNS)
                writer.writerow(row)
        except OSError as exc:
            print(f"[Logger] Failed to write log entry: {exc}")
            return False

        self.recent_logs[plate] = datetime.now()
        return True

    def read_logs(self, limit: Optional[int] = None) -> list:
        """
        Read log entries from the CSV file.

        Args:
            limit: Maximum number of most recent entries to return.

        Returns:
            List of log entry dictionaries.
        """
        if not os.path.exists(self.log_path):
            return []

        entries = []
        try:
            with open(self.log_path, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    entries.append(dict(row))
        except (OSError, csv.Error) as exc:
            print(f"[Logger] Failed to read logs: {exc}")
            return []

        if limit and len(entries) > limit:
            return entries[-limit:]

        return entries

    def display_logs_in_console(self, limit: int = 20) -> None:
        """
        Print recent log entries to the console.

        Args:
            limit: Number of recent entries to display.
        """
        entries = self.read_logs(limit=limit)
        if not entries:
            print("\n" + "=" * 70)
            print("  SECURITY LOG — No entries found.")
            print("=" * 70 + "\n")
            return

        print("\n" + "=" * 70)
        print(f"  SECURITY LOG — Last {len(entries)} entries")
        print("=" * 70)
        for entry in entries:
            print(
                f"  [{entry.get('Date', '')} {entry.get('Time', '')}] "
                f"Plate: {entry.get('NumberPlate', '')} | "
                f"Owner: {entry.get('OwnerName', '')} | "
                f"Access: {entry.get('AccessStatus', '')} | "
                f"Confidence: {entry.get('Confidence', '')}"
            )
        print("=" * 70 + "\n")
