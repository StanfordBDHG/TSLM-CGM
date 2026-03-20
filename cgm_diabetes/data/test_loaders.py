#
# Verify participants.py and cgm_loader.py are working.
# python -m cgm_diabetes.data.test_loaders
#

from cgm_diabetes.data.participants import load_participants, get_label_distribution
from cgm_diabetes.data.cgm_loader import load_cgm_for_patient, get_cgm_stats

# Test participants
print("=== Participants ===")
participants = load_participants()
print(f"Label distribution: {get_label_distribution(participants)}")
print(f"First 3 entries:")
for pid, info in list(participants.items())[:3]:
    print(f"  {pid}: {info}")

# Test CGM Loader
print("\n=== CGM Loader ===")
timestamps, glucose = load_cgm_for_patient("1001")
print(f"Patient 1001: {len(glucose)} readings")
print(f"First 5 timestamps: {timestamps[:5]}")
print(f"First 5 glucose values: {glucose[:5]}")
print(f"Stats: {get_cgm_stats(glucose)}")