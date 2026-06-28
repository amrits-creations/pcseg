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
