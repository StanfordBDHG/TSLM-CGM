#
# This source file is part of the TSLM-CGM project
#

import json
import os
from typing import Tuple, List
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# Azure config
load_dotenv()
ACCOUNT_NAME    = os.environ["AZURE_ACCOUNT_NAME"]
SAS_TOKEN       = os.environ["AZURE_SAS_TOKEN"]
CONTAINER_NAME  = os.environ["AZURE_CONTAINER_NAME"]
DATASET_PREFIX  = os.environ["AZURE_DATASET_PREFIX"]

ACCOUNT_URL = f"https://{ACCOUNT_NAME}.blob.core.windows.net?{SAS_TOKEN}"

CGM_PREFIX = (f"{DATASET_PREFIX}/wearable_blood_glucose/"
              "continuous_glucose_monitoring/dexcom_g6")


def get_container_client() -> ContainerClient:
    service = BlobServiceClient(account_url=ACCOUNT_URL)
    return service.get_container_client(CONTAINER_NAME)


def load_cgm_for_patient(patient_id: str) -> Tuple[List[str], List[float]]:
    """
    Download and parse the CGM JSON for a single patient.

    Returns:
    - timestamps : List[str]
        ISO-8601 datetime strings, one per reading, e.g.
        ["2023-07-27T23:51:04Z", "2023-07-27T23:56:04Z", ...]
    - glucose_values : List[float]
        Blood glucose in mg/dL, one per reading, e.g.
        [113.0, 117.0, ...]

    Raises ValueError if no valid readings are found in the file
    """
    client = get_container_client()
    blob_path = f"{CGM_PREFIX}/{patient_id}/{patient_id}_DEX.json"

    raw = client.get_blob_client(blob_path).download_blob().readall()
    data = json.loads(raw)

    timestamps = []
    glucose_values = []

    for reading in data["body"]["cgm"]:
        # Only use readings where event_type is EGV (Estimated Glucose Value)
        if reading.get("event_type") != "EGV":
            continue

        # Get timestamp
        ts = (reading["effective_time_frame"]
                     ["time_interval"]
                     ["start_date_time"])

        # Get glucose value
        bg = reading.get("blood_glucose", {})
        value = bg.get("value")
        if value is None:
            continue

        timestamps.append(ts)
        glucose_values.append(float(value))

    if not glucose_values:
        raise ValueError(f"No valid readings found for patient {patient_id}")

    return timestamps, glucose_values


def get_cgm_stats(glucose_values: List[float]) -> dict:
    """
    Compute basic statistics for a glucose series.
    To use when building the time series text description for model prompt.
    """
    n = len(glucose_values)
    mean = sum(glucose_values) / n
    variance = sum((x - mean) ** 2 for x in glucose_values) / n
    std = variance ** 0.5
    minimum = min(glucose_values)
    maximum = max(glucose_values)

    # Time in range: standard clinical thresholds
    # Normal: 70-180 mg/dL
    in_range  = sum(1 for x in glucose_values if 70 <= x <= 180) / n * 100
    low       = sum(1 for x in glucose_values if x < 70)  / n * 100
    high      = sum(1 for x in glucose_values if x > 180) / n * 100

    return {
        "n_readings":    n,
        "mean":          round(mean, 1),
        "std":           round(std, 1),
        "min":           round(minimum, 1),
        "max":           round(maximum, 1),
        "pct_in_range":  round(in_range, 1),
        "pct_low":       round(low, 1),
        "pct_high":      round(high, 1),
    }