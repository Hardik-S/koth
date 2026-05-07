"""Guard the KotH feature extractor against black-material key regressions.

The notebooks keep black piece counts in a lowercase-keyed dictionary:
``{"p": ..., "n": ..., "b": ..., "r": ..., "q": ...}``. A prior version looked
those values up with uppercase piece symbols, which silently zeroed every black
material count and total. This verifier is intentionally source-based so it can
run without notebook execution or the full DS3000 Python environment.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SOURCES = [
    PROJECT_ROOT / "DS3000_Phase_2 (1).ipynb",
    PROJECT_ROOT / "DS3000_Phase_2_executed.ipynb",
    PROJECT_ROOT / "koth_colab_notebook.md",
]

BAD_LOOKUP = "black_material.get(p.upper(), 0)"
EXPECTED_COUNT_LOOKUP = "features.extend([black_material.get(p, 0) for p in 'pnbrq'])"
EXPECTED_TOTAL_LOOKUP = (
    "black_total = sum(black_material.get(p, 0) * "
    "piece_values.get(p.upper(), 0) for p in 'pnbrq')"
)


def main() -> int:
    missing_sources = [path for path in FEATURE_SOURCES if not path.exists()]
    if missing_sources:
        for path in missing_sources:
            print(f"Missing expected feature source: {path.relative_to(PROJECT_ROOT)}")
        return 1

    failed = False
    for path in FEATURE_SOURCES:
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(PROJECT_ROOT)

        if BAD_LOOKUP in source:
            print(f"{rel}: still contains uppercase black_material lookup")
            failed = True

        if EXPECTED_COUNT_LOOKUP not in source:
            print(f"{rel}: missing lowercase black piece count lookup")
            failed = True

        if EXPECTED_TOTAL_LOOKUP not in source:
            print(f"{rel}: missing lowercase black material total lookup")
            failed = True

    if failed:
        return 1

    print("Black material feature lookups use lowercase dictionary keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
