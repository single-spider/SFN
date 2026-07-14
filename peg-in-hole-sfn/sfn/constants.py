"""Project-wide constants and shape split definitions."""

IMAGE_CHANNELS = 3
CROP_WIDTH = 250
CROP_HEIGHT = 200
POSITION_GRID_SIZE = 21
POSITION_CENTER = 10
POSITION_RESOLUTION_MM = 1.0
ORIENTATION_ANGLES_DEG = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
MASK_BACKGROUND = 0
MASK_PEG = 1
MASK_SEAM = 2
# Shape-disjoint research split.  With the current 16 synthetic shape assets,
# use 75% for training and hold out 25% for model-selection/final reporting.
# This is intentionally larger than the earlier 4/2/10 split; the learned
# position head overfit badly when trained on only four shape families.
TRAIN_SEEN_SHAPES = [
    "square-triangle",
    "square-square",
    "square-pentagon",
    "square-hexagon",
    "square-concave1",
    "square-convex1",
    "square-convex2",
    "square-convex3",
    "square-convex4",
    "square-fillet1",
    "square-fillet2",
    "square-fillet3",
]
VALIDATION_UNSEEN_SHAPES = ["square-diamond", "square-trapezoid"]
TEST_UNSEEN_SHAPES = ["square-concave2", "square-fillet4"]
DEFAULT_SHAPE_SPLITS = {
    "train_seen": TRAIN_SEEN_SHAPES,
    "validation_unseen": VALIDATION_UNSEEN_SHAPES,
    "test_unseen": TEST_UNSEEN_SHAPES,
}
ALL_EXPECTED_SHAPES = sorted({s for values in DEFAULT_SHAPE_SPLITS.values() for s in values})
