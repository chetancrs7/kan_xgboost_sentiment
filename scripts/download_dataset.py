import kagglehub
import shutil
from pathlib import Path

# Anchor outputs to the repo root (parent of scripts/) so the dataset lands in
# the project's data/ dir no matter which directory the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

download_path = kagglehub.dataset_download("kazanova/sentiment140")

print(f"Dataset downloaded to: {download_path}")

target_dir = PROJECT_ROOT / "data" / "raw"
target_dir.mkdir(parents=True, exist_ok=True)

source_file = Path(download_path) / "training.1600000.processed.noemoticon.csv"
destination_file = target_dir / "sentiment140_dataset.csv"
shutil.copy(source_file, destination_file)

print(f"Dataset saved to: {destination_file}")