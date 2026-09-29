"""Display formatting for hardware values."""

from __future__ import annotations

BYTES_PER_GB = 1024**3
BYTES_PER_MB = 1024**2
BYTES_PER_KB = 1024


def format_bytes(value: int | float | None, binary: bool = True) -> str:
    """Render a byte count as a short human string such as ``6.4 GB``."""
    if value is None:
        return "n/a"
    number = float(value)
    negative = number < 0
    number = abs(number)
    divisor = 1024.0 if binary else 1000.0
    for unit, size in (
        ("TB", divisor**4),
        ("GB", divisor**3),
        ("MB", divisor**2),
        ("KB", divisor),
    ):
        if number >= size:
            text = f"{number / size:.1f} {unit}"
            return f"-{text}" if negative else text
    text = f"{number:.0f} B"
    return f"-{text}" if negative else text


def format_rate(bytes_per_second: float | None) -> str:
    """Render a throughput value such as ``4.2 MB/s``."""
    if bytes_per_second is None:
        return "0 B/s"
    if bytes_per_second < 1024:
        return f"{bytes_per_second:.0f} B/s"
    return f"{format_bytes(bytes_per_second)}/s"


def format_temperature(celsius: float | None) -> str:
    """Render a temperature, or ``n/a`` when no sensor exists."""
    if celsius is None:
        return "n/a"
    return f"{celsius:.0f}°C"


def format_percent(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}%"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(max(0, seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
