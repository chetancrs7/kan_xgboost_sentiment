import kagglehub
import shutil
from pathlib import Path

download_path = kagglehub.dataset_download("kazanova/sentiment140")

print(f"Dataset downloaded to: {download_path}")

target_dir = Path("data/raw")
target_dir.mkdir(parents=True, exist_ok=True)

source_file = Path(download_path) / "training.1600000.processed.noemoticon.csv"
destination_file = target_dir / "sentiment140_dataset.csv"
shutil.copy(source_file, destination_file)

print(f"Dataset saved to: {destination_file}")