"""Colour palettes and stylesheet for the desktop interface.

Two themes plus a ``system`` option that follows the Qt platform palette. The
palette is a single source of truth so the Qt widgets and the OpenCV overlay
renderer stay visually consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

from visionai.ai import taxonomy


@dataclass(frozen=True)
class Palette:
    """Colours shared by the Qt stylesheet and the frame overlay."""

    name: str
    background: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    success: str
    warning: str
    danger: str
    overlay_text: tuple[int, int, int] = (255, 255, 255)
    overlay_backdrop: tuple[int, int, int] = (24, 26, 32)
    is_dark: bool = True

    def qcolor(self, key: str) -> str:
        return getattr(self, key)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "background": self.background,
            "surface": self.surface,
            "text": self.text,
            "accent": self.accent,
            "is_dark": self.is_dark,
        }


DARK = Palette(
    name="dark",
    background="#12141a",
    surface="#1b1e26",
    surface_alt="#232733",
    border="#2f3542",
    text="#e6e9ef",
    text_muted="#98a1b3",
    accent="#4c8dff",
    success="#3ddc84",
    warning="#ffc23d",
    danger="#ff5f56",
    overlay_text=(236, 240, 246),
    overlay_backdrop=(20, 22, 28),
    is_dark=True,
)

LIGHT = Palette(
    name="light",
    background="#f4f6fa",
    surface="#ffffff",
    surface_alt="#eef1f7",
    border="#d3d9e4",
    text="#1c2130",
    text_muted="#5b6478",
    accent="#2f6fe0",
    success="#1f9d55",
    warning="#b87700",
    danger="#c92a2a",
    overlay_text=(28, 33, 48),
    overlay_backdrop=(250, 250, 252),
    is_dark=False,
)

THEMES: dict[str, Palette] = {"dark": DARK, "light": LIGHT}


def get_palette(name: str) -> Palette:
    return THEMES.get(str(name).lower(), DARK)


def stylesheet(palette: Palette, font_family: str = "") -> str:
    """Build the application-wide Qt stylesheet for a palette."""
    family = font_family or "Inter, 'Noto Sans', 'DejaVu Sans', sans-serif"
    return f"""
QWidget {{
    background-color: {palette.background};
    color: {palette.text};
    font-family: {family};
    font-size: 13px;
}}
QMainWindow::separator {{ background: {palette.border}; width: 1px; height: 1px; }}
QFrame#Card {{
    background-color: {palette.surface};
    border: 1px solid {palette.border};
    border-radius: 10px;
}}
QLabel#CardTitle {{
    color: {palette.text_muted};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QLabel#Metric {{ font-size: 22px; font-weight: 600; }}
QLabel#MetricMuted {{ color: {palette.text_muted}; font-size: 12px; }}
QLabel#Banner {{
    background-color: {palette.surface_alt};
    border-left: 3px solid {palette.accent};
    border-radius: 6px;
    padding: 8px 10px;
    color: {palette.text_muted};
}}
QLabel#StatusRunning {{ color: {palette.success}; font-weight: 600; }}
QLabel#StatusPaused {{ color: {palette.warning}; font-weight: 600; }}
QLabel#StatusError {{ color: {palette.danger}; font-weight: 600; }}

QTabWidget::pane {{
    border: 1px solid {palette.border};
    border-radius: 10px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {palette.text_muted};
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{
    background: {palette.surface};
    color: {palette.text};
    border: 1px solid {palette.border};
    border-bottom-color: {palette.surface};
}}
QTabBar::tab:hover {{ color: {palette.text}; }}

QPushButton {{
    background-color: {palette.surface_alt};
    border: 1px solid {palette.border};
    border-radius: 7px;
    padding: 7px 14px;
    color: {palette.text};
}}
QPushButton:hover {{ border-color: {palette.accent}; }}
QPushButton:pressed {{ background-color: {palette.border}; }}
QPushButton:disabled {{ color: {palette.text_muted}; border-color: {palette.border}; }}
QPushButton#Primary {{
    background-color: {palette.accent};
    border-color: {palette.accent};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#Danger {{ color: {palette.danger}; }}
QPushButton:checked {{
    background-color: {palette.accent};
    border-color: {palette.accent};
    color: #ffffff;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {palette.surface};
    border: 1px solid {palette.border};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {palette.accent};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {palette.accent};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {palette.surface};
    border: 1px solid {palette.border};
    selection-background-color: {palette.accent};
    color: {palette.text};
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {palette.border};
    border-radius: 4px;
    background: {palette.surface};
}}
QCheckBox::indicator:checked {{ background: {palette.accent}; border-color: {palette.accent}; }}

QProgressBar {{
    background-color: {palette.surface_alt};
    border: 1px solid {palette.border};
    border-radius: 5px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{ background-color: {palette.accent}; border-radius: 4px; }}

QTableWidget, QTreeWidget, QListWidget {{
    background-color: {palette.surface};
    border: 1px solid {palette.border};
    border-radius: 8px;
    gridline-color: {palette.border};
}}
QHeaderView::section {{
    background-color: {palette.surface_alt};
    color: {palette.text_muted};
    padding: 6px;
    border: none;
    border-bottom: 1px solid {palette.border};
    font-weight: 600;
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {palette.border}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {palette.border}; border-radius: 5px; min-width: 30px; }}

QGroupBox {{
    border: 1px solid {palette.border};
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {palette.text_muted};
    font-weight: 600;
}}
QToolTip {{
    background-color: {palette.surface_alt};
    color: {palette.text};
    border: 1px solid {palette.border};
    padding: 4px;
}}
QSplitter::handle {{ background: {palette.border}; }}
"""


def expression_color(label: str) -> tuple[int, int, int]:
    """BGR colour for an expression label, shared with the overlay renderer."""
    return taxonomy.bgr_for(label)
