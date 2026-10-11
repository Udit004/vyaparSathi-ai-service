from __future__ import annotations

from datetime import date


# Known major Indian retail/festive shopping date windows (inclusive start and end).
# Covers pre-festival shopping surge windows for retail / kirana stores.
_FESTIVAL_WINDOWS: tuple[tuple[date, date], ...] = (
    # 2024
    (date(2024, 1, 13), date(2024, 1, 16)),  # Makar Sankranti / Pongal / Lohri
    (date(2024, 1, 24), date(2024, 1, 27)),  # Republic Day shopping window
    (date(2024, 3, 7), date(2024, 3, 9)),    # Maha Shivratri
    (date(2024, 3, 22), date(2024, 3, 26)),  # Holi shopping & celebration
    (date(2024, 4, 8), date(2024, 4, 12)),   # Eid al-Fitr & Chaitra Navratri
    (date(2024, 8, 13), date(2024, 8, 16)),  # Independence Day sales
    (date(2024, 8, 17), date(2024, 8, 20)),  # Raksha Bandhan
    (date(2024, 8, 24), date(2024, 8, 27)),  # Janmashtami
    (date(2024, 9, 6), date(2024, 9, 9)),    # Ganesh Chaturthi
    (date(2024, 10, 2), date(2024, 10, 14)), # Navratri, Durga Puja, Dussehra
    (date(2024, 10, 27), date(2024, 11, 8)), # Dhanteras, Diwali, Bhai Dooj, Chhath Puja
    (date(2024, 12, 23), date(2025, 1, 2)),  # Christmas & New Year

    # 2025
    (date(2025, 1, 13), date(2025, 1, 16)),  # Makar Sankranti / Pongal
    (date(2025, 1, 24), date(2025, 1, 27)),  # Republic Day
    (date(2025, 2, 24), date(2025, 2, 27)),  # Maha Shivratri
    (date(2025, 3, 12), date(2025, 3, 16)),  # Holi
    (date(2025, 3, 29), date(2025, 4, 2)),   # Eid al-Fitr
    (date(2025, 8, 7), date(2025, 8, 10)),   # Raksha Bandhan
    (date(2025, 8, 13), date(2025, 8, 17)),  # Independence Day & Janmashtami
    (date(2025, 8, 25), date(2025, 8, 29)),  # Ganesh Chaturthi
    (date(2025, 9, 21), date(2025, 10, 3)),  # Navratri, Durga Puja, Dussehra
    (date(2025, 10, 17), date(2025, 10, 30)),# Dhanteras, Diwali, Bhai Dooj, Chhath Puja
    (date(2025, 12, 23), date(2026, 1, 2)),  # Christmas & New Year

    # 2026
    (date(2026, 1, 13), date(2026, 1, 16)),  # Makar Sankranti / Pongal
    (date(2026, 1, 24), date(2026, 1, 27)),  # Republic Day
    (date(2026, 2, 14), date(2026, 2, 17)),  # Maha Shivratri
    (date(2026, 3, 2), date(2026, 3, 5)),    # Holi
    (date(2026, 3, 18), date(2026, 3, 22)),  # Eid al-Fitr
    (date(2026, 8, 13), date(2026, 8, 16)),  # Independence Day
    (date(2026, 8, 26), date(2026, 8, 30)),  # Raksha Bandhan
    (date(2026, 9, 2), date(2026, 9, 5)),    # Janmashtami
    (date(2026, 9, 12), date(2026, 9, 16)),  # Ganesh Chaturthi
    (date(2026, 10, 10), date(2026, 10, 22)),# Navratri, Durga Puja, Dussehra
    (date(2026, 11, 5), date(2026, 11, 17)), # Dhanteras, Diwali, Bhai Dooj, Chhath Puja
    (date(2026, 12, 23), date(2027, 1, 2)),  # Christmas & New Year

    # 2027
    (date(2027, 1, 13), date(2027, 1, 16)),  # Makar Sankranti
    (date(2027, 1, 24), date(2027, 1, 27)),  # Republic Day
    (date(2027, 3, 4), date(2027, 3, 8)),    # Maha Shivratri & Eid
    (date(2027, 3, 20), date(2027, 3, 24)),  # Holi
    (date(2027, 8, 13), date(2027, 8, 18)),  # Independence Day & Raksha Bandhan
    (date(2027, 9, 2), date(2027, 9, 6)),    # Ganesh Chaturthi
    (date(2027, 9, 29), date(2027, 10, 11)), # Navratri & Dussehra
    (date(2027, 10, 26), date(2027, 11, 8)), # Diwali festive window
    (date(2027, 12, 23), date(2028, 1, 2)),  # Christmas & New Year

    # 2028
    (date(2028, 1, 13), date(2028, 1, 16)),  # Makar Sankranti
    (date(2028, 1, 24), date(2028, 1, 27)),  # Republic Day
    (date(2028, 3, 9), date(2028, 3, 13)),   # Holi
    (date(2028, 8, 3), date(2028, 8, 7)),    # Raksha Bandhan
    (date(2028, 8, 13), date(2028, 8, 16)),  # Independence Day
    (date(2028, 10, 17), date(2028, 10, 29)),# Navratri & Dussehra
    (date(2028, 11, 13), date(2028, 11, 26)),# Diwali festive window
    (date(2028, 12, 23), date(2029, 1, 2)),  # Christmas & New Year
)


def is_indian_festival_window(target_date: date) -> int:
    """Returns 1 if target_date falls inside a major Indian festive shopping window, 0 otherwise."""
    for start_window, end_window in _FESTIVAL_WINDOWS:
        if start_window <= target_date <= end_window:
            return 1

    # Generalized fallback: In India, late October through mid November is virtually always festive
    if target_date.month == 10 and target_date.day >= 20:
        return 1
    if target_date.month == 11 and target_date.day <= 15:
        return 1

    return 0


def is_payday_window(target_date: date) -> int:
    """
    Returns 1 if target_date falls during retail salary/pension payday cycles
    (1st to 5th or 28th to 31st of any month), 0 otherwise.
    """
    return 1 if (target_date.day <= 5 or target_date.day >= 28) else 0


def is_weekend(target_date: date) -> int:
    """Returns 1 if Saturday or Sunday, 0 otherwise."""
    return 1 if target_date.weekday() >= 5 else 0


def get_calendar_features(target_date: date) -> dict[str, int]:
    """Returns all retail calendar indicators for the given date."""
    return {
        "day_of_week": target_date.weekday(),
        "day_of_month": target_date.day,
        "month": target_date.month,
        "is_weekend": is_weekend(target_date),
        "is_payday_window": is_payday_window(target_date),
        "is_festival_window": is_indian_festival_window(target_date),
    }
