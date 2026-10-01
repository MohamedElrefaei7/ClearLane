"""Project-wide constants that more than one milestone depends on (see CLAUDE.md)."""

# Temporal splits, inclusive months (CLAUDE.md "Splits", set 2026-10-01).
TRAIN = ("2021-01", "2024-09")
VALIDATE = ("2024-10", "2025-09")
TEST = ("2025-10", "2026-09")
PANEL_MONTHS = (TRAIN[0], TEST[1])

H3_RES = 9
