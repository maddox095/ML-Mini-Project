"""Export the current selected forest for the static browser demonstration.

The browser uses this JSON representation only for inference.  The joblib
pipeline remains the authoritative archived model.
"""

from __future__ import annotations

def main() -> None:
    from export_forest_model import main as export_forest
    export_forest()


if __name__ == "__main__":
    main()
