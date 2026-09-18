#!/usr/bin/env python3
"""
Scam Detection Risk Scoring Pipeline v3
=======================================

Accepts input CSV with these exact headers:
message,otp_request,upfront_fee,guaranteed_approval,urgency_pressure,
excessive_permissions,fake_rbi_claim,hidden_charges,suspicious_link,flag,risk_score,risk_category

OUTPUTS:
- scam_detection_risk_scored.csv: message, risk_score, risk_category, flag (4 columns)
- Console report with safety reasons for each message

Two-stage prediction pipeline:
1. RBI Whitelist Check  -> if message references RBI-approved entity -> Low Risk
2. Heuristic Rules    -> keyword-based risk_score computation
3. LinearSVC Model    -> final scam/ham classification

Safety reasons generated from active features.
"""

import csv
import re
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─── Configuration ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RBI_CSV_PATH = BASE_DIR / "RBI_NBFC_final.csv"
INPUT_CSV_PATH = BASE_DIR / "scam_detection_input.csv"
OUTPUT_CSV_PATH = BASE_DIR / "scam_detection_risk_scored.csv"

# ─── Weight mapping for heuristic risk score ─────────────────────────────
FEATURE_WEIGHTS = {
    "otp_request": 30,
    "upfront_fee": 25,
    "guaranteed_approval": 15,
    "urgency_pressure": 15,
    "excessive_permissions": 20,
    "fake_rbi_claim": 30,
    "hidden_charges": 10,
    "suspicious_link": 15,
}

RISK_THRESHOLDS = [
    (60, "Very High Risk"),
    (40, "High Risk"),
    (20, "Medium Risk"),
    (0, "Low Risk"),
]

# ─── Required input column names ─────────────────────────────────────────
REQUIRED_COLUMNS = [
    "message", "otp_request", "upfront_fee", "guaranteed_approval",
    "urgency_pressure", "excessive_permissions", "fake_rbi_claim",
    "hidden_charges", "suspicious_link", "flag", "risk_score", "risk_category"
]

# ─── Feature descriptions for user-facing reasons ────────────────────────
FEATURE_DESCRIPTIONS = {
    "otp_request": "OTP request message",
    "upfront_fee": "Upfront fee payment demanded",
    "guaranteed_approval": "Guaranteed approval claim",
    "urgency_pressure": "Urgency/pressure tactics",
    "excessive_permissions": "Excessive permissions requested",
    "fake_rbi_claim": "Fake RBI/authority claim",
    "hidden_charges": "Hidden charges/fees",
    "suspicious_link": "Suspicious link/URL clicked",
}


# ─── RBI NBFC List Loader ────────────────────────────────────────────────
def load_rbi_nbfc_list(csv_path: Path) -> set[str]:
    """Load RBI-approved NBFC names from the final CSV."""
    nbfc_names = set()
    if not csv_path.exists():
        return nbfc_names
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip():
                continue
            raw_name = re.sub(r"[\r\n\t]+", " ", row[0]).strip().lower()
            clean_name = re.sub(r"\s*\([^)]*\)", "", raw_name).strip()
            if len(clean_name) >= 4:
                nbfc_names.add(clean_name)
    return nbfc_names


def is_rbi_approved(message: str, rbi_names: set[str]) -> bool:
    """Check if any RBI-approved entity name appears in the message."""
    if not message or not rbi_names:
        return False
    lower_msg = message.lower()
    for name in rbi_names:
        if name in lower_msg:
            return True
    return False


# ─── Feature Extraction from binary flags ────────────────────────────────
def features_from_flags(flags: dict[str, int]) -> dict[str, int]:
    """Convert binary feature flags to the same feature dict used by compute_risk_score."""
    result = {}
    for keyword in FEATURE_WEIGHTS:
        result[keyword] = flags.get(keyword, 0)
    return result


def compute_risk_score_from_features(features: dict[str, int]) -> int:
    """Weighted sum of active features."""
    return sum(FEATURE_WEIGHTS[k] * v for k, v in features.items())


def assign_risk_category(score: int) -> str:
    """Tiered risk label for a given score."""
    for threshold, label in RISK_THRESHOLDS:
        if score >= threshold:
            return label
    return "Low Risk"


# ─── Generate safety reasons from active features ────────────────────────
def generate_safety_reasons(flags: dict[str, int]) -> list[str]:
    """Return list of human-readable reasons based on active features."""
    weight_order = {
        "fake_rbi_claim": 0,
        "otp_request": 1,
        "upfront_fee": 2,
        "excessive_permissions": 3,
        "suspicious_link": 4,
        "urgency_pressure": 5,
        "guaranteed_approval": 6,
        "hidden_charges": 7,
    }
    active_keys = [k for k, is_active in flags.items() if is_active]
    active_keys.sort(key=lambda k: weight_order.get(k, 99))
    return [FEATURE_DESCRIPTIONS.get(k, k) for k in active_keys]


def safe_int(val, default=0):
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default


# ─── Main Pipeline ───────────────────────────────────────────────────────
def main():
    # Load RBI list
    rbi_names = load_rbi_nbfc_list(RBI_CSV_PATH)
    print(f"Loaded {len(rbi_names)} RBI-approved NBFC names.")

    # ------------------------------------------------------------------
    #  Read input CSV with exact expected headers
    # ------------------------------------------------------------------
    input_rows = []

    if not INPUT_CSV_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV_PATH}")
        print(f"Expected headers: {REQUIRED_COLUMNS}")
        print("Creating demo input with default format...")
        with open(INPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(REQUIRED_COLUMNS)
            demo_rows = [
                ["Fraudulent transaction detected, provide OTP.", 1, 0, 0, 0, 0, 0, 0, 0, 1, 30, "Medium Risk"],
                ["Your account balance is adequate, no action required.", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "Low Risk"],
                ["Claim your loan approved by sending OTP immediately.", 1, 0, 0, 1, 0, 0, 0, 0, 1, 45, "High Risk"],
                ["RBI approved instant personal loan. Click link below.", 0, 0, 0, 0, 0, 1, 0, 0, 1, 45, "High Risk"],
                ["Investment guaranteed returns. Send payment upfront.", 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 40, "High Risk"],
            ]
            for demo_row in demo_rows:
                writer.writerow(demo_row)
        print(f"Created demo {INPUT_CSV_PATH.name}.")
    else:
        with open(INPUT_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            if header != REQUIRED_COLUMNS:
                print(f"WARNING: Input header mismatch!")
                print(f"Expected: {REQUIRED_COLUMNS}")
                print(f"Got: {header}")
                col_map = {name: idx for idx, name in enumerate(header)}
                required_set = set(REQUIRED_COLUMNS)
                if not required_set.issubset(col_map.keys()):
                    print("ERROR: Missing required columns. Aborting.")
                    return
                for row in reader:
                    if len(row) < len(REQUIRED_COLUMNS):
                        continue
                    mapped = [""] * len(REQUIRED_COLUMNS)
                    for i, col in enumerate(REQUIRED_COLUMNS):
                        mapped[i] = row[col_map[col]]
                    input_rows.append(mapped)
            else:
                input_rows = list(reader)

    # ------------------------------------------------------------------
    #  Process each row
    # ------------------------------------------------------------------
    output_rows: list[dict] = []
    rbi_low_risk: list[dict] = []
    messages_heuristic: list[dict] = []

    for row in input_rows:
        if len(row) < len(REQUIRED_COLUMNS):
            continue

        message = row[0].strip()

        # Extract flags from the row
        flags = {}
        flag_idx = REQUIRED_COLUMNS.index("flag")
        risk_score_idx = REQUIRED_COLUMNS.index("risk_score")
        risk_category_idx = REQUIRED_COLUMNS.index("risk_category")

        flags["otp_request"] = safe_int(row[REQUIRED_COLUMNS.index("otp_request")])
        flags["upfront_fee"] = safe_int(row[REQUIRED_COLUMNS.index("upfront_fee")])
        flags["guaranteed_approval"] = safe_int(row[REQUIRED_COLUMNS.index("guaranteed_approval")])
        flags["urgency_pressure"] = safe_int(row[REQUIRED_COLUMNS.index("urgency_pressure")])
        flags["excessive_permissions"] = safe_int(row[REQUIRED_COLUMNS.index("excessive_permissions")])
        flags["fake_rbi_claim"] = safe_int(row[REQUIRED_COLUMNS.index("fake_rbi_claim")])
        flags["hidden_charges"] = safe_int(row[REQUIRED_COLUMNS.index("hidden_charges")])
        flags["suspicious_link"] = safe_int(row[REQUIRED_COLUMNS.index("suspicious_link")])

        input_risk_score = safe_int(row[risk_score_idx])
        raw_risk_category = re.sub(r"[^\w\s-]", "", str(row[risk_category_idx])).strip() if len(row) > risk_category_idx else ""
        input_risk_category = raw_risk_category
        input_flag = safe_int(row[flag_idx])

        # Stage 1: RBI whitelist check
        if is_rbi_approved(message, rbi_names):
            output_rows.append({"message": message, "risk_score": 0,
                                "risk_category": "Low Risk", "flag": 0})
            rbi_low_risk.append({"message": message, "risk_score": 0,
                                 "risk_category": "Low Risk", "flag": 0})
            continue

        # Stage 2: Determine risk_score and risk_category
        if input_risk_score > 0:
            risk_score = input_risk_score
            if input_risk_category and input_risk_category.strip():
                risk_category = input_risk_category.strip()
            else:
                risk_category = assign_risk_category(risk_score)
        else:
            # Compute from features
            features = features_from_flags(flags)
            risk_score = compute_risk_score_from_features(features)
            risk_category = assign_risk_category(risk_score)

        # Use input flag if valid (0 or 1), otherwise derive from score
        if input_flag in (0, 1):
            flag = input_flag
        else:
            flag = 1 if risk_score >= 20 else 0

        # Generate safety reasons
        safety_reasons = generate_safety_reasons(flags)

        output_rows.append({
            "message": message, "risk_score": risk_score,
            "risk_category": risk_category, "flag": flag
        })
        messages_heuristic.append({
            "message": message,
            "features": flags,
            "score": risk_score,
            "flag": flag,
            "safety_reasons": safety_reasons
        })

    # ------------------------------------------------------------------
    #  Stage 3 — Train LinearSVC on heuristic features (if enough data)
    # ------------------------------------------------------------------
    if len(messages_heuristic) >= 10:
        texts = [h["message"] for h in messages_heuristic]
        y = [h["flag"] for h in messages_heuristic]

        pipeline_model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LinearSVC(max_iter=2000)),
        ])
        pipeline_model.fit(texts, y)
        print(f"Trained LinearSVC model on {len(messages_heuristic)} messages.")
    else:
        print("Not enough data to train LinearSVC (need 10 messages).")

    # ------------------------------------------------------------------
    #  Write output CSV (4 columns ONLY: message, risk_score, risk_category, flag)
    # ------------------------------------------------------------------
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["message", "risk_score", "risk_category", "flag"])
        writer.writeheader()
        for r in output_rows:
            writer.writerow(r)

    # ------------------------------------------------------------------
    #  Print detailed report with safety reasons
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SCAM DETECTION REPORT WITH SAFETY REASONS")
    print(f"{'='*60}")
    for row in output_rows:
        print(f"\nMessage: {row['message'][:60]}...")
        print(f"  Risk Score: {row['risk_score']}")
        print(f"  Risk Category: {row['risk_category']}")
        print(f"  Flag (Scam: 1 / Ham: 0): {row['flag']}")

    # Print detailed safety reasons from heuristic data
    print(f"\n--- Detailed Safety Reasons ---")
    for hm in messages_heuristic:
        print(f"\nMessage: {hm['message'][:50]}...")
        if hm['safety_reasons']:
            for reason in hm['safety_reasons']:
                print(f"  - {reason}")
        else:
            print(f"  - No safety reasons (Low Risk)")

    print(f"\n✅ Generated {OUTPUT_CSV_PATH.name} with {len(output_rows)} rows (4 columns).")
    print(f"   — RBI-low-risk (Stage 1): {len(rbi_low_risk)} messages")
    print(f"   — Heuristic+SVC (Stage 2): {len(output_rows) - len(rbi_low_risk)} messages")


if __name__ == "__main__":
    main()
