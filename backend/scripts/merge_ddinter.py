"""Merge DDInter CSV files into a single JSON database for the interaction checker.

Usage:
    python scripts/merge_ddinter.py

Reads all ddinter_*.csv files from data/ and writes data/drug_interactions.json.
"""

import csv
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT = DATA_DIR / "drug_interactions.json"


def normalize_drug_name(name: str) -> str:
    """Normalize drug name for consistent matching."""
    name = name.strip()
    # Remove common suffixes that vary between entries
    for suffix in [" sodium", " potassium", " hydrochloride", " sulfate",
                    " acetate", " mesylate", " fumarate", " tartrate",
                    " maleate", " phosphate", " citrate", " succinate",
                    " besylate", " tosylate", " HCl", " HCI", " Cl"]:
        if name.lower().endswith(suffix.lower()):
            name = name[: -len(suffix)].strip()
    return name


def merge_csvs() -> dict:
    """Read all DDInter CSVs and merge into a dict keyed by (drug_a_lower, drug_b_lower)."""
    interactions: dict[tuple[str, str], dict] = {}

    for csv_file in sorted(DATA_DIR.glob("ddinter_*.csv")):
        print(f"Reading {csv_file.name}...")
        with open(csv_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                if not row or not row.get("Drug_A"):
                    continue
                drug_a = (row.get("Drug_A") or "").strip()
                drug_b = (row.get("Drug_B") or "").strip()
                level = (row.get("Level") or "").strip()

                if not drug_a or not drug_b or not level:
                    continue

                # Normalize to lowercase sorted pair for dedup
                key = tuple(sorted([drug_a.lower(), drug_b.lower()]))

                if key not in interactions:
                    interactions[key] = {
                        "drug_a": drug_a,
                        "drug_b": drug_b,
                        "level": level,
                        "normalized_a": normalize_drug_name(drug_a),
                        "normalized_b": normalize_drug_name(drug_b),
                        "sources": [],
                    }

                interactions[key]["sources"].append(csv_file.stem)

                # Keep the higher severity if we see duplicates
                severity_order = {"Minor": 0, "Moderate": 1, "Major": 2}
                if severity_order.get(level, 0) > severity_order.get(
                    interactions[key]["level"], 0
                ):
                    interactions[key]["level"] = level

                count += 1
            print(f"  {count} rows")

    return interactions


def build_index(interactions: dict) -> dict:
    """Build a drug-name -> list of interactions index for fast lookup."""
    drug_index: dict[str, list[str]] = {}

    for key, entry in interactions.items():
        for drug_field in ["drug_a", "drug_b"]:
            drug_lower = entry[drug_field].lower()
            if drug_lower not in drug_index:
                drug_index[drug_lower] = []
            drug_index[drug_lower].append(key[0] if drug_field == "drug_a" else key[1])

    return drug_index


def main():
    interactions = merge_csvs()

    # Build output
    entries = []
    for key, entry in interactions.items():
        entries.append({
            "drug_a": entry["drug_a"],
            "drug_b": entry["drug_b"],
            "level": entry["level"],
            "normalized_a": entry["normalized_a"],
            "normalized_b": entry["normalized_b"],
        })

    # Sort by severity (Major first), then alphabetically
    severity_order = {"Major": 0, "Moderate": 1, "Minor": 2}
    entries.sort(key=lambda e: (severity_order.get(e["level"], 3), e["drug_a"].lower()))

    output = {
        "version": "1.0",
        "source": "DDInter (ddinter.scbdd.com)",
        "license": "CC BY 4.0",
        "total_interactions": len(entries),
        "severity_counts": {
            "Major": sum(1 for e in entries if e["level"] == "Major"),
            "Moderate": sum(1 for e in entries if e["level"] == "Moderate"),
            "Minor": sum(1 for e in entries if e["level"] == "Minor"),
        },
        "interactions": entries,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(entries)} interactions to {OUTPUT}")
    print(f"Severity: {output['severity_counts']}")


if __name__ == "__main__":
    main()
