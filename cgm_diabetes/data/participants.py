#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import io
import os
from typing import Dict
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv
import pandas as pd

# Map the long study_group strings to short labels
LABEL_MAP = {
    "healthy":                                              "healthy",
    "pre_diabetes_lifestyle_controlled":                    "prediabetes_lifestyle",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled":
                                                            "oral_non_insulin",
    "insulin_dependent":                                    "insulin_dependent",
}


def get_container_client() -> ContainerClient:
    load_dotenv()
    account_name   = os.environ["AZURE_ACCOUNT_NAME"]
    sas_token      = os.environ["AZURE_SAS_TOKEN"]
    container_name = os.environ["AZURE_CONTAINER_NAME"]
    account_url    = f"https://{account_name}.blob.core.windows.net?{sas_token}"
    service = BlobServiceClient(account_url=account_url)
    return service.get_container_client(container_name)


def load_participants() -> Dict[str, Dict]:
    """
    Download participants.tsv from Azure and return a dict with patient IDs as keys.

    Returns
    -------
    {
        "1001": {"label": "prediabetes_lifestyle", "split": "train"},
        "1002": {"label": "healthy",               "split": "train"},
        ...
    }
    Only includes patients who have wearable_blood_glucose data
    and whose study_group is one of the four known labels.
    """
    load_dotenv()
    dataset_prefix = os.environ["AZURE_DATASET_PREFIX"]
    client = get_container_client()
    blob_path = f"{dataset_prefix}/participants.tsv"

    # Download TSV into memory
    raw = client.get_blob_client(blob_path).download_blob().readall()
    df = pd.read_csv(io.BytesIO(raw), sep="\t", dtype={"person_id": str})

    original_len = len(df)

    # Keep only patients with CGM data and a recognised label
    df = df[df["wearable_blood_glucose"].astype(str).str.lower() == "true"]
    df = df[df["study_group"].isin(LABEL_MAP.keys())]

    filtered_len = len(df)

    # Map labels to short versions
    df["label"] = df["study_group"].map(LABEL_MAP)

    # Print skipped patients
    n_skipped = original_len - filtered_len
    print(f"[participants] Loaded {filtered_len} patients")
    print(f"[participants] Skipped {n_skipped} patients")
    print(f"  {df['recommended_split'].value_counts().to_dict()}")
    print(f"  {df['label'].value_counts().to_dict()}")

    # Convert to dict
    participants = {
        str(row.person_id): {
            "label": row.label,
            "split": row.recommended_split,
        }
        for row in df.itertuples()
    }

    return participants




def get_split(participants: Dict, split: str) -> Dict:
    """Return only patients belonging to a given split (train/val/test)."""
    return {pid: info for pid, info in participants.items()
            if info["split"] == split}


def get_label_distribution(participants: Dict) -> Dict[str, int]:
    """Count how many patients per label."""
    labels = [info["label"] for info in participants.values()]
    return pd.Series(labels).value_counts().to_dict()