from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "archive(3).zip"
OUTPUT_PATH = BASE_DIR / "partidas_com_features_v2.csv"

WINDOWS = [3, 5, 10, 15]
EMA_SPANS = [5, 10]

GOALS_CAP = 5.0
SHOTS_PER_GOAL_CAP = 20.0
SHOTS_ON_TARGET_PER_GOAL_CAP = 10.0
YELLOW_CARDS_CAP = 5.0
RED_CARDS_CAP = 2.0
STREAK_CAP = 10.0

FINISHED_STATUSES = {"FT", "AET", "PEN"}