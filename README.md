# Automatic Number Plate Recognition (ANPR) System

A Python-based Automatic Number Plate Recognition system for vehicle
identification and access control.

## Overview

This project detects vehicle number plates from a live webcam feed,
recognizes the plate characters using EasyOCR, validates the recognized
number, searches the vehicle database, determines access status and
records the event in a security log.

## Features

- Number plate detection using OpenCV Haar Cascade
- Plate region extraction
- Image preprocessing
- OCR using EasyOCR
- OCR error correction
- Indian number plate format validation
- Exact database matching
- Fuzzy matching using TheFuzz
- Access granted/denied decision
- Security event logging
- Duplicate log suppression
- Real-time OpenCV interface

## Technology Stack

- Python
- OpenCV
- EasyOCR
- NumPy
- Pandas
- TheFuzz
- Python-Levenshtein

## System Workflow

Webcam
→ Plate Detection
→ Plate Extraction
→ OCR Preprocessing
→ EasyOCR
→ Text Cleaning
→ OCR Correction
→ Plate Validation
→ Database Lookup
→ Access Decision
→ Security Logging

## Project Structure

```text
ANPR-System/
├── main.py
├── detector.py
├── ocr.py
├── database.py
├── logger.py
├── utils.py
├── haarcascade_plate.xml
├── vehicle_database.csv
├── security_log.csv
├── requirements.txt
