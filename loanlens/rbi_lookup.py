"""
RBI NBFC Lender Verification Module

Handles lookup of RBI-registered NBFCs with robust name matching
that accounts for:
- uppercase/lowercase differences
- internal newlines and extra spaces
- punctuation variations
- Pvt vs Private, Ltd vs Limited, Co vs Company, & vs and
- Parenthetical notes: (Formerly: ...), (Name as per MCA...), (CIC), etc.
- Common corporate suffixes (searching "Bajaj Finance" matches "Bajaj Finance Limited")
- Meaningful token overlap and similarity scoring
"""

import re
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple


# ─── Configuration ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RBI_CSV_PATH = BASE_DIR / "RBI_NBFC_final.csv"

_RBI_ENTITIES: List[Dict[str, Any]] = []
_RBI_NAME_INDEX: Dict[str, List[Dict[str, Any]]] = {}
_RBI_BASE_INDEX: Dict[str, List[Dict[str, Any]]] = {}

_GENERIC_TERMS: Set[str] = {
    "bank", "finance", "loan", "services", "service", "private", "limited",
    "pvt", "ltd", "company", "corporation", "corp", "group", "holdings",
    "holding", "associates", "india", "indian", "financial", "lending",
    "credit", "capital", "investments", "investment", "invest", "fincap",
    "finvest", "co", "and", "the", "of", "in", "for", "management", "scheme",
    "enterprise", "enterprises", "leasing", "securities", "commercial", "housing",
    "nbfc", "cic", "p2p", "factor", "icc"
}

_loaded = False


def normalize_name(name: str) -> str:
    """Normalize a lender or RBI entity name for matching.

    Handles:
    - Lowercase conversion
    - Newlines / tabs collapsed to single space
    - Punctuation removal
    - Common abbreviations: pvt -> private, ltd -> limited, co -> company, & -> and
    """
    if not name:
        return ""

    s = name.lower().strip()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[.,;:!\?\"'`()\-/\\]", " ", s)
    s = re.sub(r"\bpvt\b", "private", s)
    s = re.sub(r"\bltd\b", "limited", s)
    s = re.sub(r"\bco\b", "company", s)
    s = re.sub(r"\b&\b", "and", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_corp_suffixes(normalized: str) -> str:
    """Strip common corporate suffixes to get core business name.
    
    e.g. 'bajaj finance limited' -> 'bajaj finance'
         '121 finance private limited' -> '121 finance'
    """
    if not normalized:
        return ""
    # Strip from end first
    s = re.sub(
        r"\b(private limited|pvt ltd|private ltd|pvt limited|limited|ltd|private|company|co|llp|llc|inc|corp|corporation)\b$",
        "",
        normalized.strip()
    ).strip()
    # Strip any trailing 'and' or punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_primary_and_aliases(raw_name: str) -> Tuple[str, List[str]]:
    """Extract the primary clean company name and any aliases/former names."""
    raw_clean = re.sub(r"[\r\n\t]+", " ", raw_name).strip()
    # Remove parenthetical comments for primary name
    primary = re.sub(r"\s*\([^)]*\)", "", raw_clean).strip()
    
    # Extract former/alternate names inside parentheses
    parentheses = re.findall(r"\(([^)]+)\)", raw_clean)
    aliases: List[str] = []
    for p in parentheses:
        # Clean prefix like "Formerly: ", "Name as per MCA - "
        cleaned_p = re.sub(
            r"^(formerly|name\s+as\s+per\s+mca|formerly\s+known\s+as|known\s+as)\s*[:\-]?\s*",
            "",
            p,
            flags=re.IGNORECASE
        ).strip()
        if cleaned_p:
            for sub in re.split(r"\band\b|;|,", cleaned_p, flags=re.IGNORECASE):
                sub_clean = sub.strip()
                if len(sub_clean) >= 3:
                    aliases.append(sub_clean)
                    
    return primary or raw_clean, aliases


def _load_rbi_data():
    """Load RBI NBFC data from CSV into memory with multi-key indexing."""
    global _RBI_ENTITIES, _RBI_NAME_INDEX, _RBI_BASE_INDEX, _loaded

    if _loaded:
        return

    _RBI_ENTITIES = []
    _RBI_NAME_INDEX = {}
    _RBI_BASE_INDEX = {}

    if not RBI_CSV_PATH.exists():
        print(f"WARNING: RBI dataset not found at {RBI_CSV_PATH}")
        _loaded = True
        return

    with open(RBI_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 1:
                continue

            raw_name = row[0].strip()
            if not raw_name:
                continue

            regional_office = row[1].strip() if len(row) > 1 else ""
            deposit_accepting = row[2].strip() if len(row) > 2 else ""
            classification = row[3].strip() if len(row) > 3 else ""
            cin = row[4].strip() if len(row) > 4 else ""
            layer = row[5].strip() if len(row) > 5 else ""
            address = row[6].strip() if len(row) > 6 else ""
            contact = row[7].strip() if len(row) > 7 else ""

            primary_name, aliases = _extract_primary_and_aliases(raw_name)

            entity = {
                "name": primary_name,
                "full_name": raw_name,
                "aliases": aliases,
                "cin": cin,
                "regional_office": regional_office,
                "classification": classification,
                "layer": layer,
                "deposit_status": deposit_accepting,
                "address": address,
                "contact": contact,
            }

            _RBI_ENTITIES.append(entity)

            # Index full raw name, primary clean name, and aliases
            names_to_index = [raw_name, primary_name] + aliases
            for n in names_to_index:
                norm = normalize_name(n)
                if norm:
                    if norm not in _RBI_NAME_INDEX:
                        _RBI_NAME_INDEX[norm] = []
                    _RBI_NAME_INDEX[norm].append(entity)

                    # Also index base (corporate suffix stripped)
                    base = strip_corp_suffixes(norm)
                    if base and len(base) >= 3:
                        if base not in _RBI_BASE_INDEX:
                            _RBI_BASE_INDEX[base] = []
                        _RBI_BASE_INDEX[base].append(entity)

    _loaded = True


def search_lender(lender_name: str) -> Dict[str, Any]:
    """Search for an RBI-registered lender.

    Returns a dict with status and relevant info:
    - REGISTERED: strong exact or base-name match found
    - PARTIAL_MATCH: similar entity with high confidence
    - NOT_FOUND: no match in database
    """
    _load_rbi_data()

    if not lender_name or not lender_name.strip():
        return {
            "status": "NOT_FOUND",
            "name": "",
            "cin": "",
            "regional_office": "",
            "classification": "",
            "layer": "",
            "deposit_status": "",
        }

    input_raw = lender_name.strip()
    input_norm = normalize_name(input_raw)
    input_base = strip_corp_suffixes(input_norm)

    if not input_norm:
        return {
            "status": "NOT_FOUND",
            "name": "",
            "cin": "",
            "regional_office": "",
            "classification": "",
            "layer": "",
            "deposit_status": "",
        }

    # 1. Exact match on normalized full name / primary name / alias
    if input_norm in _RBI_NAME_INDEX:
        entity = _RBI_NAME_INDEX[input_norm][0]
        return {
            "status": "REGISTERED",
            "name": entity["name"],
            "cin": entity["cin"],
            "regional_office": entity["regional_office"],
            "classification": entity["classification"],
            "layer": entity["layer"],
            "deposit_status": entity.get("deposit_status", ""),
            "address": entity.get("address", ""),
        }

    # 2. Base match (e.g. "Bajaj Finance" matches "Bajaj Finance Limited")
    if input_base and len(input_base) >= 4 and input_base in _RBI_BASE_INDEX:
        entity = _RBI_BASE_INDEX[input_base][0]
        return {
            "status": "REGISTERED",
            "name": entity["name"],
            "cin": entity["cin"],
            "regional_office": entity["regional_office"],
            "classification": entity["classification"],
            "layer": entity["layer"],
            "deposit_status": entity.get("deposit_status", ""),
            "address": entity.get("address", ""),
        }

    # 3. Fuzzy & Partial Matching
    input_words = set(input_norm.split())
    input_meaningful = input_words - _GENERIC_TERMS
    input_base_words = set(input_base.split())

    best_match: Optional[Dict[str, Any]] = None
    best_score = 0
    best_confidence = 0

    for entity in _RBI_ENTITIES:
        entity_norm = normalize_name(entity["name"])
        entity_base = strip_corp_suffixes(entity_norm)
        entity_words = set(entity_norm.split())
        entity_meaningful = entity_words - _GENERIC_TERMS

        # Check if one base is a complete prefix / substring of the other
        if input_base and entity_base:
            if input_base == entity_base:
                return {
                    "status": "REGISTERED",
                    "name": entity["name"],
                    "cin": entity["cin"],
                    "regional_office": entity["regional_office"],
                    "classification": entity["classification"],
                    "layer": entity["layer"],
                    "deposit_status": entity.get("deposit_status", ""),
                    "address": entity.get("address", ""),
                }

        # Meaningful word overlap
        if input_meaningful and entity_meaningful:
            overlap = input_meaningful & entity_meaningful
            if overlap:
                # Calculate Jaccard-like confidence on meaningful words
                overlap_ratio = len(overlap) / max(len(input_meaningful), len(entity_meaningful))
                score = len(overlap) * 20 + int(overlap_ratio * 50)

                # Bonus if all input meaningful words are contained
                if input_meaningful.issubset(entity_meaningful):
                    score += 40
                elif entity_meaningful.issubset(input_meaningful):
                    score += 30

                # Check if non-generic unique prefix matches
                if len(input_base) >= 6 and (input_base in entity_base or entity_base in input_base):
                    score += 30

                confidence = min(95, round((len(overlap) / max(len(input_meaningful), 1)) * 100))

                if score > best_score:
                    best_score = score
                    best_match = entity
                    best_confidence = confidence

    # Return partial match if confidence is strong enough
    if best_match and best_score >= 50 and best_confidence >= 50:
        return {
            "status": "PARTIAL_MATCH",
            "name": best_match["name"],
            "cin": best_match["cin"],
            "regional_office": best_match["regional_office"],
            "classification": best_match["classification"],
            "layer": best_match["layer"],
            "deposit_status": best_match.get("deposit_status", ""),
            "address": best_match.get("address", ""),
            "match_confidence": best_confidence,
        }

    return {
        "status": "NOT_FOUND",
        "name": "",
        "cin": "",
        "regional_office": "",
        "classification": "",
        "layer": "",
        "deposit_status": "",
    }


def rbi_verification(lender_name: str) -> Dict[str, Any]:
    """Public API for RBI lender verification."""
    return search_lender(lender_name)


# Preload on module import
_load_rbi_data()
