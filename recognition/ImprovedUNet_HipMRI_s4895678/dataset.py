"""Data loading for the HipMRI prostate segmentation project.

Part 1 (this commit): leakage-free, patient-level stratified train/val/test split.
Volume loading follows the NIfTI approach in the COMP3710 task sheet, Appendix B.
"""
import csv
import os
import random
from collections import defaultdict

DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open"
IMAGE_DIR = f"{DATA_DIR}/semantic_MRs"
LABEL_DIR = f"{DATA_DIR}/semantic_labels_only"
SPLIT_CSV = "results/split.csv"


def list_cases():
    """Return sorted case names such as 'B040_Week3' (one per scan)."""
    return sorted(f.replace("_LFOV.nii.gz", "")
                  for f in os.listdir(IMAGE_DIR) if f.endswith("_LFOV.nii.gz"))


def image_path(case):
    return f"{IMAGE_DIR}/{case}_LFOV.nii.gz"


def label_path(case):
    # Images and labels pair by swapping LFOV -> SEMANTIC in the filename.
    return f"{LABEL_DIR}/{case}_SEMANTIC.nii.gz"


def patient_of(case):
    """Patient ID is the text before the first underscore."""
    return case.split("_")[0]


def make_split(val_frac=0.15, test_frac=0.15, seed=42):
    """Split by PATIENT so no patient appears in more than one split.

    Patients are grouped by how many scans they have (full course = 8,
    single = 1, otherwise partial) and each group is split separately, so
    train/val/test get a similar mix. Returns {case_name: 'train'|'val'|'test'}.
    """
    by_patient = defaultdict(list)
    for case in list_cases():
        by_patient[patient_of(case)].append(case)

    strata = defaultdict(list)
    for patient, cases in by_patient.items():
        n = len(cases)
        key = "full" if n == 8 else ("single" if n == 1 else "partial")
        strata[key].append(patient)

    rng = random.Random(seed)
    patient_split = {}
    for key in sorted(strata):
        patients = sorted(strata[key])
        rng.shuffle(patients)
        n_val = max(1, round(val_frac * len(patients)))
        n_test = max(1, round(test_frac * len(patients)))
        for i, p in enumerate(patients):
            if i < n_test:
                patient_split[p] = "test"
            elif i < n_test + n_val:
                patient_split[p] = "val"
            else:
                patient_split[p] = "train"

    return {c: patient_split[patient_of(c)] for c in list_cases()}


def save_split(split, path=SPLIT_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "patient", "split"])
        for case, s in split.items():
            w.writerow([case, patient_of(case), s])


def load_split(path=SPLIT_CSV):
    with open(path) as f:
        return {r["case"]: r["split"] for r in csv.DictReader(f)}


def check_no_leakage(split):
    """Assert that every patient sits in exactly one split."""
    seen = defaultdict(set)
    for case, s in split.items():
        seen[patient_of(case)].add(s)
    bad = {p: s for p, s in seen.items() if len(s) > 1}
    assert not bad, f"Patients in more than one split: {bad}"


if __name__ == "__main__":
    split = make_split()
    check_no_leakage(split)
    save_split(split)
    for s in ("train", "val", "test"):
        cases = [c for c, v in split.items() if v == s]
        print(f"{s}: {len(cases)} volumes from {len({patient_of(c) for c in cases})} patients")
