import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset', 'FIFA World Cup 2022')
EVENT_DATA_DIR = os.path.join(DATASET_ROOT, 'Event Data')
TRACKING_DATA_DIR = os.path.join(DATASET_ROOT, 'Tracking Data')

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Extraction Config
WINDOW_SIZE_FRAMES = 150  # ~5 seconds at ~30Hz - buildup period before shot
FRAME_RATE = 29.97
