"""Project-wide constants that more than one milestone depends on (see CLAUDE.md)."""

# Temporal splits, inclusive months (CLAUDE.md "Splits", set 2026-10-01).
TRAIN = ("2021-01", "2024-09")
VALIDATE = ("2024-10", "2025-09")
TEST = ("2025-10", "2026-09")
# The panel runs from TRAIN[0] to the last complete 311 month: clearlane.ingest.sr311.panel_months().

H3_RES = 9
