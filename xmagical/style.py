"""Common visual settings and color tools for this benchmark."""

import colorsys
from typing import Tuple

RGBTuple = Tuple[float, float, float]


def rgb(r: float, g: float, b: float) -> RGBTuple:
    return (r / 255.0, g / 255.0, b / 255.0)


def darken_rgb(rgb: RGBTuple) -> RGBTuple:
    """Produce a darker version of a base color."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    hls_new = (h, max(0, l * 0.9), s)
    return colorsys.hls_to_rgb(*hls_new)


def lighten_rgb(rgb: RGBTuple, times: float = 1.0) -> RGBTuple:
    """Produce a lighter version of a given base color."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    mult = 1.4 ** times
    hls_new = (h, 1 - (1 - l) / mult, s)
    return colorsys.hls_to_rgb(*hls_new)


GOAL_LINE_THICKNESS = 0.01
SHAPE_LINE_THICKNESS = 0.015
ROBOT_LINE_THICKNESS = 0.01
COLORS_RGB = {
    # I'm using Berkeley-branded versions of RGBY from
    # https://brand.berkeley.edu/colors/ (lightened).
    "blue": lighten_rgb(rgb(0x3B, 0x7E, 0xA1), 1.7),  # founder's rock
    "yellow": lighten_rgb(rgb(0xFD, 0xB5, 0x15), 1.7),  # california gold
    "red": lighten_rgb(rgb(0xEE, 0x1F, 0x60), 1.7),  # rose garden
    "green": lighten_rgb(rgb(0x85, 0x94, 0x38), 1.7),  # soybean
    "grey": rgb(162, 163, 175),  # cool grey (not sure which one)
    "brown": rgb(224, 171, 118),  # buff
}
# "zoom out" factor when rendering arena; values above 1 will show parts of the
# arena border in allocentric view.
ARENA_ZOOM_OUT = 1.02
