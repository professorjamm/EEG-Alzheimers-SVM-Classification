from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATHS = (
    PROJECT_ROOT / "Tasks" / "data.csv",
    PROJECT_ROOT / "data.csv",
)
REQUIRED_COLUMNS = {"Subject", "Group", "Delta", "Theta", "Alpha", "Beta"}


def _resolve_data_path(data_path=None):
  """Resolve the CSV location for channel-level EEG band-power data."""
  if data_path:
    candidate = Path(data_path).expanduser()
    if not candidate.is_absolute():
      candidate = PROJECT_ROOT / candidate
    if candidate.exists():
      return candidate
    raise FileNotFoundError(f"Could not find data.csv at: {candidate}")

  for candidate in DEFAULT_DATA_PATHS:
    if candidate.exists():
      return candidate

  expected = "\n".join(str(path) for path in DEFAULT_DATA_PATHS)
  raise FileNotFoundError(
      "Could not locate data.csv. Checked:\n" + expected
  )


def preprocess_data(data_path=None):
  """Load channel-level data.csv and aggregate it to one row per subject."""
  csv_path = _resolve_data_path(data_path)
  channel_level_df = pd.read_csv(csv_path)

  missing_columns = REQUIRED_COLUMNS.difference(channel_level_df.columns)
  if missing_columns:
    missing = ", ".join(sorted(missing_columns))
    raise ValueError(f"data.csv is missing required columns: {missing}")

  subject_level_df = (
      channel_level_df.groupby(["Subject", "Group"], as_index=False)[
          ["Delta", "Theta", "Alpha", "Beta"]
      ]
      .mean()
      .sort_values("Subject")
      .reset_index(drop=True)
  )

  return subject_level_df
  
