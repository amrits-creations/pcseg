__version__ = "0.0.1"

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
