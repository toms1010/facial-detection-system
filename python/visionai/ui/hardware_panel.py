"""Hardware panel: CPU, RAM, GPU, temperature, disk and network telemetry."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from visionai.hardware.formatting import (
    format_bytes,
    format_duration,
    format_rate,
    format_temperature,
)
from visionai.hardware.hardware_bridge import HardwareBridge, HardwareSnapshot
from visionai.ui.theme import Palette
from visionai.ui.widgets import Banner, Card, KeyValueGrid, UsageBar

LOG = logging.getLogger(__name__)


class HardwarePanel(QWidget):
    """Live hardware telemetry, sampled by the bridge on its own thread."""

    def __init__(
        self,
        bridge: HardwareBridge,
        palette: Palette,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._palette = palette
        self._last: HardwareSnapshot | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.notes_banner = Banner("Hardware monitoring has not produced a sample yet.")
        layout.addWidget(self.notes_banner)

        tiles = QGridLayout()
        tiles.setHorizontalSpacing(10)
        tiles.setVerticalSpacing(10)
        self.bars: dict[str, UsageBar] = {}
        specifications = (
            ("cpu", "CPU"),
            ("cpu_temp", "CPU temperature"),
            ("ram", "RAM"),
            ("gpu", "GPU"),
            ("gpu_temp", "GPU temperature"),
            ("disk", "Disk"),
        )
        for index, (key, caption) in enumerate(specifications):
            bar = UsageBar(caption, palette)
            self.bars[key] = bar
            tiles.addWidget(bar, index // 2, index % 2)
        layout.addLayout(tiles)

        network_card = Card("Network")
        self.rx_bar = UsageBar("Download", palette, maximum=1_000_000.0)
        self.tx_bar = UsageBar("Upload", palette, maximum=1_000_000.0)
        network_card.add(self.rx_bar)
        network_card.add(self.tx_bar)
        self.network_table = QTableWidget(0, 4)
        self.network_table.setHorizontalHeaderLabels(
            ["Interface", "Down", "Up", "State"]
        )
        self.network_table.verticalHeader().setVisible(False)
        self.network_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.network_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.network_table.setMaximumHeight(160)
        network_card.add(self.network_table)
        layout.addWidget(network_card)

        detail_card = Card("System and sensors")
        self.system_grid = KeyValueGrid()
        detail_card.add(self.system_grid)
        self.sensor_table = QTableWidget(0, 2)
        self.sensor_table.setHorizontalHeaderLabels(["Sensor", "Temperature"])
        self.sensor_table.verticalHeader().setVisible(False)
        self.sensor_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sensor_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.sensor_table.setMaximumHeight(150)
        detail_card.add(self.sensor_table)
        layout.addWidget(detail_card, stretch=1)

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        for bar in self.bars.values():
            bar.set_palette_colors(palette)
        self.rx_bar.set_palette_colors(palette)
        self.tx_bar.set_palette_colors(palette)

    def update_snapshot(self, snapshot: HardwareSnapshot) -> None:
        """Push one snapshot into every widget."""
        self._last = snapshot
        self.notes_banner.setText(
            "  |  ".join(snapshot.notes) if snapshot.notes else "All sensors reporting normally."
        )

        self.bars["cpu"].set_value(
            snapshot.cpu_percent,
            f"{snapshot.cpu_percent:.0f}%  ({snapshot.cpu_cores} cores, {snapshot.cpu_frequency_mhz:.0f} MHz)",
        )
        self.bars["cpu_temp"].set_value(
            snapshot.cpu_temp_c,
            f"{format_temperature(snapshot.cpu_temp_c)}  {snapshot.temperature_source}".strip(),
        )
        self.bars["ram"].set_value(
            snapshot.ram_percent,
            f"{format_bytes(snapshot.ram_used_bytes)} / {format_bytes(snapshot.ram_total_bytes)}",
        )
        if snapshot.gpu_percent is None:
            self.bars["gpu"].set_unknown("no GPU telemetry")
        else:
            self.bars["gpu"].set_value(
                snapshot.gpu_percent,
                f"{snapshot.gpu_percent:.0f}%  {snapshot.gpu_name}".strip(),
            )
        self.bars["gpu_temp"].set_value(
            snapshot.gpu_temp_c,
            format_temperature(snapshot.gpu_temp_c),
        )
        self.bars["disk"].set_value(
            snapshot.disk_percent,
            f"{snapshot.disk_percent:.0f}%  {format_bytes(snapshot.disk_total_bytes)} at {snapshot.disk_path}",
        )

        self.rx_bar.set_value(
            min(snapshot.net_rx_bytes_per_second, 1_000_000.0),
            f"{format_rate(snapshot.net_rx_bytes_per_second)}  {snapshot.net_interface}",
        )
        self.tx_bar.set_value(
            min(snapshot.net_tx_bytes_per_second, 1_000_000.0),
            f"{format_rate(snapshot.net_tx_bytes_per_second)}  {snapshot.net_interface}",
        )
        self._populate_network(snapshot)
        self._populate_system(snapshot)
        self._populate_sensors(snapshot)

    def _populate_network(self, snapshot: HardwareSnapshot) -> None:
        interfaces = snapshot.net_interfaces
        self.network_table.setRowCount(len(interfaces))
        for row, iface in enumerate(interfaces):
            values = (
                str(iface.get("name", "")),
                format_rate(float(iface.get("rx_bytes_per_second", 0.0))),
                format_rate(float(iface.get("tx_bytes_per_second", 0.0))),
                "up" if iface.get("is_up") else "down",
            )
            for column, text in enumerate(values):
                item = self.network_table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.network_table.setItem(row, column, item)
                item.setText(text)
                if column:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )

    def _populate_system(self, snapshot: HardwareSnapshot) -> None:
        self.system_grid.set("CPU", snapshot.cpu_model or "unknown")
        self.system_grid.set("Load average", "  ".join(f"{v:.2f}" for v in snapshot.load_average))
        self.system_grid.set("Swap", f"{format_bytes(snapshot.swap_total_bytes)} total")
        self.system_grid.set(
            "GPU", f"{snapshot.gpu_vendor} {snapshot.gpu_name}".strip() or "not detected"
        )
        if snapshot.gpu_memory_total_bytes:
            self.system_grid.set(
                "GPU memory",
                f"{format_bytes(snapshot.gpu_memory_used_bytes)} / "
                f"{format_bytes(snapshot.gpu_memory_total_bytes)}",
            )
        self.system_grid.set("Disk free", format_bytes(snapshot.disk_free_bytes))
        self.system_grid.set("Host", snapshot.distribution or "unknown distribution")
        self.system_grid.set("Kernel", snapshot.kernel or "unknown")
        self.system_grid.set("Uptime", format_duration(snapshot.uptime_seconds))
        self.system_grid.set("Backend", f"{snapshot.backend} v{snapshot.version}")
        self.system_grid.set("Sample time", f"{snapshot.sample_ms:.1f} ms")

    def _populate_sensors(self, snapshot: HardwareSnapshot) -> None:
        sensors = snapshot.temperature_sensors
        self.sensor_table.setRowCount(len(sensors))
        for row, sensor in enumerate(sensors):
            for column, text in enumerate(
                (str(sensor.get("label", "")), format_temperature(sensor.get("celsius")))
            ):
                item = self.sensor_table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.sensor_table.setItem(row, column, item)
                item.setText(text)
                if column:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )

    def summary_text(self) -> str:
        """One-line summary used by the tests."""
        return "  ".join(f"{k}={bar.value_text()}" for k, bar in self.bars.items())
