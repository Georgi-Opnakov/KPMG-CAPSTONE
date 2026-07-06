from __future__ import annotations

from calendar_service import calendar_metadata, initialise_calendar_db


def main() -> None:
    db_path = initialise_calendar_db(force=False)
    print(f"Calendar cache ready: {db_path}")
    print(calendar_metadata().to_string(index=False))


if __name__ == "__main__":
    main()
