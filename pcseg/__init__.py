"""pcseg — Point Cloud Semantic Segmentation on S3DIS.

The project package. Right now it holds just the shared constants
(the 13 S3DIS classes); modules for data, models, training and
evaluation get added in their chunks (see BUILD_PROCESS.md).
"""

__version__ = "0.0.1"

# 13 S3DIS semantic classes (the standard ordering, meaning that this is the exact order that the dataset defines).
S3DIS_CLASSES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
)

NUM_CLASSES = len(S3DIS_CLASSES)

# A fixed RGB color (0–255) per class, in the same order as S3DIS_CLASSES.
# This is the common S3DIS visualization palette. Kept here (not in a script)
# because qualitative prediction-vs-ground-truth renders in Chunk 6 reuse the
# exact same legend, so colors stay consistent across the whole project.
S3DIS_COLORS = (
    (0, 255, 0),      # ceiling
    (0, 0, 255),      # floor
    (0, 255, 255),    # wall
    (255, 255, 0),    # beam
    (255, 0, 255),    # column
    (100, 100, 255),  # window
    (200, 200, 100),  # door
    (170, 120, 200),  # table
    (255, 0, 0),      # chair
    (200, 100, 100),  # sofa
    (10, 200, 100),   # bookcase
    (200, 200, 200),  # board
    (50, 50, 50),     # clutter
)

assert len(S3DIS_COLORS) == NUM_CLASSES, "one color per class"
