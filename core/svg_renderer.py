"""
SVG pinout diagram renderer for chip datasheets.

Renders PinInfo lists as SVG vector diagrams with:
- DIP (Dual In-line Package) and QFP (Quad Flat Package) layouts
- Pin-type color coding with GND secondary validation
- Interactive tooltips and click events (CustomEvent)
- Auto-generated color legend

Usage:
    renderer = PinoutSVGRenderer()
    svg_string = renderer.render(pins, "DIP-8", "NE5532")
"""

import json
import re
from html import escape
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from io import StringIO

from core import PinInfo, PinType, get_settings

_GND_PATTERN = re.compile(r'GND', re.IGNORECASE)

PIN_COLORS: Dict[PinType, Dict[str, str]] = {
    PinType.POWER: {
        "fill": "#FF6B6B",
        "stroke": "#C0392B",
        "text": "#FFFFFF",
    },
    PinType.GROUND: {
        "fill": "#2C3E50",
        "stroke": "#1A252F",
        "text": "#FFFFFF",
    },
    PinType.IO: {
        "fill": "#3498DB",
        "stroke": "#2980B9",
        "text": "#FFFFFF",
    },
    PinType.BIDIRECTIONAL: {
        "fill": "#1ABC9C",
        "stroke": "#16A085",
        "text": "#FFFFFF",
    },
    PinType.INPUT: {
        "fill": "#2ECC71",
        "stroke": "#27AE60",
        "text": "#FFFFFF",
    },
    PinType.OUTPUT: {
        "fill": "#E74C3C",
        "stroke": "#C0392B",
        "text": "#FFFFFF",
    },
    PinType.ANALOG: {
        "fill": "#9B59B6",
        "stroke": "#8E44AD",
        "text": "#FFFFFF",
    },
    PinType.NC: {
        "fill": "#BDC3C7",
        "stroke": "#95A5A6",
        "text": "#7F8C8D",
    },
    PinType.SPECIAL: {
        "fill": "#F39C12",
        "stroke": "#D68910",
        "text": "#FFFFFF",
    },
}

DEFAULT_COLOR: Dict[str, str] = {
    "fill": "#95A5C7",
    "stroke": "#7F8C8D",
    "text": "#FFFFFF",
}

GND_COLOR: Dict[str, str] = {
    "fill": "#374151",
    "stroke": "#1f2937",
    "text": "#ffffff",
}

LEGEND_ITEMS: Tuple[Tuple[PinType, str], ...] = (
    (PinType.POWER, "电源 (VCC/VDD)"),
    (PinType.GROUND, "接地 (GND/VSS)"),
    (PinType.IO, "通用I/O"),
    (PinType.BIDIRECTIONAL, "GPIO双向"),
    (PinType.INPUT, "输入引脚"),
    (PinType.OUTPUT, "输出引脚"),
    (PinType.ANALOG, "模拟引脚"),
    (PinType.SPECIAL, "特殊功能"),
    (PinType.NC, "未连接"),
)

_GND_NAMES: frozenset = frozenset({"GND", "VSS", "AGND", "DGND", "PGND"})

_QFP_PACKAGES: frozenset = frozenset({"QFP", "LQFP", "TQFP", "QFN", "BGA", "CSP"})
_DIP_PACKAGES: frozenset = frozenset({"DIP", "SOIC", "TSSOP", "SSOP", "SOP"})


def _is_ground_pin(name: str, pin_type: PinType) -> bool:
    """Check whether a pin is a ground pin by name or type.

    Performs both exact name matching (against _GND_NAMES) and regex
    pattern matching (against _GND_PATTERN) to catch variants like
    AGND, DGND, PGND that may be classified under a different PinType.
    """
    name_upper = name.upper().strip()
    if name_upper in _GND_NAMES:
        return True
    if pin_type == PinType.GROUND:
        return True
    return bool(_GND_PATTERN.search(name))


@dataclass(frozen=True)
class SVGConfig:
    """Immutable rendering configuration for SVG pinout diagrams.

    Attributes:
        pin_width: Pin rectangle width in pixels.
        pin_height: Pin rectangle height in pixels.
        pin_spacing: Vertical spacing between pins.
        chip_padding: Padding around the chip body (mainly for QFP).
        pin_number_font_size: Font size for pin numbers.
        pin_name_font_size: Font size for pin names.
        title_font_size: Font size for the chip title.
        show_legend: Whether to render the color legend.
    """

    pin_width: int = 240
    pin_height: int = 62
    pin_spacing: int = 10
    chip_padding: int = 300
    pin_number_font_size: int = 17
    pin_name_font_size: int = 20
    title_font_size: int = 30
    show_legend: bool = True

    @classmethod
    def from_settings(cls, settings=None) -> "SVGConfig":
        """Create an SVGConfig from the global Settings singleton."""
        if settings is None:
            settings = get_settings()
        return cls(
            pin_width=settings.SVG_PIN_WIDTH,
            pin_height=settings.SVG_PIN_HEIGHT,
            pin_spacing=settings.SVG_PIN_SPACING,
            chip_padding=settings.SVG_CHIP_PADDING,
            pin_number_font_size=settings.SVG_PIN_NUMBER_FONT_SIZE,
            pin_name_font_size=settings.SVG_PIN_NAME_FONT_SIZE,
            title_font_size=settings.SVG_TITLE_FONT_SIZE,
            show_legend=settings.SVG_SHOW_LEGEND,
        )


class PinoutSVGRenderer:
    """Renders PinInfo lists as SVG pinout diagrams.

    Supports DIP (dual-column) and QFP (quad-side) package layouts with
    pin-type color coding, interactive tooltips, click events, and an
    auto-generated color legend.

    Thread-safe: SVGConfig is frozen and _cached_styles is written once.
    """

    __slots__ = ('config', '_cached_styles')

    def __init__(self, config: Optional[SVGConfig] = None, settings=None):
        """Initialize the renderer.

        Args:
            config: Optional SVGConfig; defaults to one built from *settings*.
            settings: Optional Settings; defaults to the global singleton.
        """
        self.config = config or SVGConfig.from_settings(settings or get_settings())
        self._cached_styles = None

    def render(self, pins: List[PinInfo], package: str = "DIP",
               chip_name: str = "") -> str:
        """Render a pinout SVG diagram.

        Selects DIP or QFP layout based on the package string:
            1. Check for QFP-family keywords (QFP, LQFP, TQFP, QFN, BGA, CSP).
            2. Check for DIP-family keywords (DIP, SOIC, TSSOP, SSOP, SOP).
            3. Fallback: >40 pins → QFP, otherwise DIP.

        Args:
            pins: List of PinInfo objects.
            package: Package type string, e.g. 'DIP-8', 'LQFP-48'.
            chip_name: Chip name displayed at the top.

        Returns:
            Complete SVG string.
        """
        if not pins:
            return self._render_empty()

        package = (package or "DIP").upper()

        qfp_like = any(k in package for k in _QFP_PACKAGES)
        dip_like = any(k in package for k in _DIP_PACKAGES)

        if qfp_like or (not dip_like and len(pins) > 40):
            return self._render_qfp(pins, chip_name)
        return self._render_dip(pins, chip_name)

    def _render_dip(self, pins: List[PinInfo], chip_name: str) -> str:
        """Render a DIP (Dual In-line Package) layout.

        Left-side pins are drawn top-to-bottom (Pin 1 → N/2).
        Right-side pins are drawn bottom-to-top (Pin N/2+1 → N)
        following the standard DIP counter-clockwise numbering.

        Args:
            pins: PinInfo list sorted by pin number.
            chip_name: Chip name for the title.

        Returns:
            SVG string with DIP layout.
        """
        C = self.config
        pin_count = len(pins)
        per_side = (pin_count + 1) // 2
        step = C.pin_height + C.pin_spacing
        chip_w = 160
        chip_h = per_side * step + 20
        pad_x = 30

        left_x0 = pad_x
        left_x1 = left_x0 + C.pin_width
        chip_x = left_x1 + 4
        chip_x2 = chip_x + chip_w
        right_x0 = chip_x2 + 4
        right_x1 = right_x0 + C.pin_width

        title_y = 36
        chip_y = 58
        legend_y = chip_y + chip_h + 44

        legend_col_w = 200
        legend_row_h = 34
        legend_cols = 4
        legend_rows = (len(LEGEND_ITEMS) + legend_cols - 1) // legend_cols
        legend_h = legend_rows * legend_row_h + 8

        total_w = right_x1 + pad_x
        total_h = legend_y + legend_h + 20

        buf = StringIO()
        buf.write(self._svg_header(total_w, total_h))
        buf.write(self._svg_styles())

        if chip_name:
            buf.write(f'<text x="{total_w/2:.1f}" y="{title_y}" class="chip-title">{chip_name}</text>')

        buf.write(self._chip_body(chip_x, chip_y, chip_w, chip_h))
        buf.write(f'<circle cx="{chip_x + chip_w/2:.1f}" cy="{chip_y + 14}" r="11" fill="#d1d5db" stroke="#9ca3af" stroke-width="1.5"/>')

        for i, pin in enumerate(pins[:per_side]):
            y = chip_y + 12 + i * step
            buf.write(self._pin_rect(pin, left_x0, y, C.pin_width, C.pin_height, align="left"))

        right_pins = pins[per_side:]
        for i in range(len(right_pins) - 1, -1, -1):
            pin = right_pins[i]
            y = chip_y + 12 + (len(right_pins) - 1 - i) * step
            buf.write(self._pin_rect(pin, right_x0, y, C.pin_width, C.pin_height, align="right"))

        if C.show_legend:
            lx = (total_w - legend_cols * legend_col_w) / 2
            buf.write(self._legend(lx, legend_y, legend_col_w, legend_row_h))

        buf.write('</svg>')
        return buf.getvalue()

    def _render_qfp(self, pins: List[PinInfo], chip_name: str) -> str:
        """Render a QFP (Quad Flat Package) layout.

        Pins are distributed across four sides in counter-clockwise order:
            Bottom → Right → Top (reversed) → Left.

        Args:
            pins: PinInfo list sorted by pin number.
            chip_name: Chip name for the title.

        Returns:
            SVG string with QFP layout.
        """
        C = self.config
        pin_count = len(pins)
        per_side = (pin_count + 3) // 4

        side_len = max(200, per_side * 50)
        step = (side_len - 36) / max(per_side - 1, 1)
        font_scale = min(1.0, 50.0 / max(step, 1))
        ph = max(36, min(C.pin_height, int(step * 0.85)))

        chip_x = C.chip_padding + C.pin_width + 4
        chip_y = 58

        legend_col_w = 200
        legend_row_h = 34
        legend_cols = 4
        legend_rows = (len(LEGEND_ITEMS) + legend_cols - 1) // legend_cols
        legend_h = legend_rows * legend_row_h + 8

        total_w = side_len + 2 * C.chip_padding + 2 * C.pin_width
        total_h = chip_y + side_len + C.pin_width + 50 + legend_h

        buf = StringIO()

        buf.write(self._svg_header(total_w, total_h))
        buf.write(self._svg_styles())

        if chip_name:
            buf.write(f'<text x="{total_w/2:.1f}" y="36" class="chip-title">{chip_name}</text>')

        buf.write(self._chip_body(chip_x, chip_y, side_len, side_len))
        buf.write(f'<circle cx="{chip_x + 20}" cy="{chip_y + 20}" r="8" fill="#1e293b"/>')

        bottom_pins = pins[:per_side]
        right_pins = pins[per_side:2*per_side]
        top_pins = pins[2*per_side:3*per_side]
        left_pins = pins[3*per_side:]

        pw = C.pin_width

        for i, pin in enumerate(bottom_pins):
            x = chip_x + 18 + i * step
            buf.write(self._pin_vertical(pin, x, chip_y + side_len + 4, ph, pw, upward=False, font_scale=font_scale))

        for i, pin in enumerate(right_pins):
            y = chip_y + side_len - 18 - i * step
            buf.write(self._pin_rect(pin, chip_x + side_len + 4, y - ph/2, pw, ph, align="right", font_scale=font_scale))

        for i in range(len(top_pins) - 1, -1, -1):
            pin = top_pins[i]
            x = chip_x + 18 + (len(top_pins) - 1 - i) * step
            buf.write(self._pin_vertical(pin, x, chip_y - pw - 4, ph, pw, upward=True, font_scale=font_scale))

        for i, pin in enumerate(left_pins):
            y = chip_y + 18 + i * step
            buf.write(self._pin_rect(pin, C.chip_padding, y, pw, ph, align="left", font_scale=font_scale))

        if C.show_legend:
            legend_y = chip_y + side_len + pw + 40
            lx = (total_w - legend_cols * legend_col_w) / 2
            buf.write(self._legend(lx, legend_y, legend_col_w, legend_row_h))

        buf.write('</svg>')
        return buf.getvalue()

    def _svg_header(self, w: float, h: float) -> str:
        """Generate the SVG root element with viewBox and responsive sizing."""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {w:.1f} {h:.1f}" '
            f'width="100%" height="100%" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'style="overflow:visible;display:block;" '
            f'class="pinout-svg">'
        )

    def _svg_styles(self) -> str:
        """Generate the CSS <style> block (cached after first call)."""
        if self._cached_styles is None:
            C = self.config
            self._cached_styles = f'''<defs><style>
.chip-title{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;font-size:{C.title_font_size}px;font-weight:700;text-anchor:middle;fill:#1e293b;}}
.chip-body{{fill:#f8fafc;stroke:#334155;stroke-width:2.5;filter:drop-shadow(0 3px 6px rgba(0,0,0,.10));}}
.pin-g{{cursor:pointer;}}
.pin-g:hover .pin-r{{filter:brightness(1.12);stroke-width:2.5;}}
.pin-r{{stroke-width:1.8;transition:all .15s;}}
.pnum{{font-family:"SF Mono","Monaco","Fira Code",monospace;font-size:{C.pin_number_font_size}px;font-weight:600;text-anchor:middle;dominant-baseline:central;pointer-events:none;}}
.pname{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;font-size:{C.pin_name_font_size}px;font-weight:700;text-anchor:middle;dominant-baseline:central;pointer-events:none;}}
.leg-label{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;font-size:18px;fill:#1e293b;font-weight:500;dominant-baseline:central;}}
</style></defs>'''
        return self._cached_styles

    def _chip_body(self, x: float, y: float, w: float, h: float) -> str:
        """Generate the chip body rectangle SVG element."""
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" class="chip-body" rx="5" ry="5"/>'

    def _get_pin_colors(self, pin: PinInfo) -> Tuple[str, str, str]:
        """Return (fill, stroke, text) colors for a pin.

        GND pins detected by name take priority over PinType to ensure
        visual consistency regardless of upstream classification.
        """
        if _is_ground_pin(pin.name, pin.pin_type):
            return GND_COLOR["fill"], GND_COLOR["stroke"], GND_COLOR["text"]

        color = PIN_COLORS.get(pin.pin_type, DEFAULT_COLOR)
        return color.get("fill", "#94a3b8"), color.get("stroke", "#475569"), "#0f172a"

    def _pin_base(self, pin: PinInfo, x: float, y: float, w: float, h: float, num_y: float, name_y: float, font_scale: float = 1.0) -> str:
        """Generate a single pin's SVG <g> element with rect, number, and name."""
        fill, stroke, tc = self._get_pin_colors(pin)
        tooltip = self._tooltip(pin)
        click = self._click_handler(pin)
        cx = x + w / 2

        cfg = self.config
        fs_num = cfg.pin_number_font_size * font_scale if font_scale != 1.0 else None
        fs_name = cfg.pin_name_font_size * font_scale if font_scale != 1.0 else None
        num_style = f' style="font-size:{fs_num:.1f}px"' if fs_num else ''
        name_style = f' style="font-size:{fs_name:.1f}px"' if fs_name else ''

        return (f'<g class="pin-g" data-pin="{pin.number}">\n'
                f'  <title>{tooltip}</title>\n'
                f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" class="pin-r" rx="5" ry="5" onclick="{click}"/>\n'
                f'  <text x="{cx:.1f}" y="{num_y:.1f}" class="pnum" fill="{tc}"{num_style}>{pin.number}</text>\n'
                f'  <text x="{cx:.1f}" y="{name_y:.1f}" class="pname" fill="{tc}"{name_style}>{pin.name}</text>\n'
                f'</g>')

    def _pin_rect(self, pin: PinInfo, x: float, y: float, w: float, h: float, align: str = "left", font_scale: float = 1.0) -> str:
        """Generate a horizontal pin element (DIP left/right, QFP left/right)."""
        num_y = y + h * 0.32
        name_y = y + h * 0.70
        return self._pin_base(pin, x, y, w, h, num_y, name_y, font_scale)

    def _pin_vertical(self, pin: PinInfo, x: float, y: float, pw: float, ph: float, upward: bool = False, font_scale: float = 1.0) -> str:
        """Generate a vertical pin element (QFP top/bottom)."""
        num_y = y + ph * 0.28
        name_y = y + ph * 0.68
        return self._pin_base(pin, x, y, pw, ph, num_y, name_y, font_scale)

    def _legend(self, x: float, y: float, col_w: float, row_h: float) -> str:
        """Generate the color legend SVG group.

        GND legend items use GND_COLOR (not PIN_COLORS[GROUND]) to match
        the actual rendering of GND-detected pins.
        """
        cols = 4
        buf = StringIO()
        buf.write(f'<g transform="translate({x:.1f},{y:.1f})">')

        for i, (pin_type, label) in enumerate(LEGEND_ITEMS):
            col = i % cols
            row = i // cols
            lx = col * col_w
            ly = row * row_h
            color = PIN_COLORS.get(pin_type, DEFAULT_COLOR)
            fill = color.get("fill", "#94a3b8")
            stroke = color.get("stroke", "#475569")

            if pin_type == PinType.GROUND:
                fill = GND_COLOR["fill"]
                stroke = GND_COLOR["stroke"]

            buf.write(f'<rect x="{lx}" y="{ly + 6}" width="22" height="20" fill="{fill}" stroke="{stroke}" rx="4"/>')
            buf.write(f'<text x="{lx + 30}" y="{ly + row_h/2 + 2}" class="leg-label">{label}</text>')

        buf.write('</g>')
        return buf.getvalue()

    def _click_handler(self, pin: PinInfo) -> str:
        """Generate the onclick JavaScript for a pin click event.

        Dispatches a CustomEvent('pinClick') on window with pin details,
        allowing frontend code to listen without tight coupling.
        """
        d = {"number": pin.number, "name": pin.name, "type": pin.pin_type, "functions": pin.functions or []}
        json_str = escape(json.dumps(d, ensure_ascii=False), quote=True)
        return f"window.dispatchEvent(new CustomEvent('pinClick',{{detail:{json_str}}}))"

    def _tooltip(self, pin: PinInfo) -> str:
        """Generate tooltip text for a pin (shown on hover via SVG <title>).

        Shows pin number, name, type, and up to 3 alternate functions.
        """
        parts = [f"Pin {pin.number}: {pin.name}", f"Type: {pin.pin_type}"]
        if pin.functions:
            funcs = pin.functions[:3]
            parts.append(f"Functions: {', '.join(funcs)}")
        return " | ".join(parts)

    def _render_empty(self) -> str:
        """Render a placeholder SVG when no pin data is available."""
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200">'
                '<rect x="50" y="50" width="300" height="100" fill="#f5f5f5" stroke="#ddd" rx="5"/>'
                '<text x="200" y="100" text-anchor="middle" fill="#999" font-size="14">暂无引脚数据</text>'
                '</svg>')
