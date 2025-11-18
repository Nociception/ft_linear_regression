"""
Adaptative colors palette for dalton modes.
"""

from typing import Dict, Tuple


BASE_PALETTE: Dict[str, Tuple[float, float, float]] = {
    "name": "base",
    "cmap": "viridis",
    "blue": (0.0, 0.447, 0.741),
    "orange": (0.85, 0.325, 0.098),
    "red": (0.9, 0.1, 0.1),
    "green": (0.466, 0.674, 0.188),
    "purple": (0.494, 0.184, 0.556),
    "cyan": (0.301, 0.745, 0.933),
    "yellow": (0.929, 0.694, 0.125),
    "gray": (0.5, 0.5, 0.5),
    "black": (0.0, 0.0, 0.0),
}

DALTON_PALETTE: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "protanopia": {
        "name": "protanopia",
        "cmap": "cividis",
        "blue": (0.0, 0.447, 0.741),
        "orange": (0.85, 0.325, 0.098),
        "red": (0.55, 0.24, 0.1),
        "green": (0.466, 0.674, 0.188),
        "purple": (0.45, 0.18, 0.55),
        "cyan": (0.3, 0.74, 0.93),
        "yellow": (0.93, 0.69, 0.12),
        "gray": (0.5, 0.5, 0.5),
        "black": (0.0, 0.0, 0.0),
    },
    "deuteranopia": {
        "name": "deuteranopia",
        "cmap": "cividis",
        "blue": (0.0, 0.447, 0.741),
        "orange": (0.85, 0.33, 0.1),
        "red": (0.9, 0.15, 0.1),
        "green": (0.5, 0.65, 0.2),
        "purple": (0.49, 0.18, 0.56),
        "cyan": (0.3, 0.74, 0.93),
        "yellow": (0.93, 0.69, 0.12),
        "gray": (0.5, 0.5, 0.5),
        "black": (0.0, 0.0, 0.0),
    },
    "tritanopia": {
        "name": "tritanopia",
        "cmap": "cubehelix",
        "blue": (0.0, 0.45, 0.72),
        "orange": (0.85, 0.32, 0.1),
        "red": (0.9, 0.1, 0.1),
        "green": (0.46, 0.67, 0.19),
        "purple": (0.49, 0.18, 0.56),
        "cyan": (0.3, 0.73, 0.9),
        "yellow": (0.93, 0.69, 0.12),
        "gray": (0.5, 0.5, 0.5),
        "black": (0.0, 0.0, 0.0),
    },
}


def get_palette(
    dalton_type: str | None = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Returns the palette adapted to the type of color blindness, or the base palette."""
    if dalton_type is None:
        return BASE_PALETTE
    if dalton_type not in DALTON_PALETTE:
        raise ValueError(
            f"Unknown dalton_type: {dalton_type}. Choose from {list(DALTON_PALETTE.keys())}"
        )
    return DALTON_PALETTE[dalton_type]
