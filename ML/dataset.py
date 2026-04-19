from pathlib import Path

import pandas as pd
from pandas import pivot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATHS = (
    PROJECT_ROOT / "Tasks" / "data.csv",
    PROJECT_ROOT / "data.csv",
)
REQUIRED_COLUMNS = {"Subject", "Group", "Channel", "Delta", "Theta", "Alpha", "Beta"}


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


def preprocess_data(data_path=None, mode="subject"):
  """Load channel-level data.csv and aggregate it to one row per subject."""
  csv_path = _resolve_data_path(data_path)
  #channel_level_df = pd.read_csv(csv_path)
  df = pd.read_csv(csv_path)

  missing_columns = REQUIRED_COLUMNS.difference(df.columns)
  if missing_columns:
    missing = ", ".join(sorted(missing_columns))
    raise ValueError(f"data.csv is missing required columns: {missing}")

  #for 4 features
  if mode == "subject":
   subject_df = (
      df.groupby(["Subject", "Group"], as_index=False)[
          ["Delta", "Theta", "Alpha", "Beta"]
      ]
      .mean()
      .sort_values("Subject")
      .reset_index(drop=True)
   )
   return subject_df
  #for 76 features
  elif mode == "channel":
   if "Channel" not in df.columns:
    raise ValueError("Channel column is required for channel-level features")
   pivot_df = df.pivot_table(
      index=["Subject", "Group"],
      columns="Channel",
      values=["Delta", "Theta", "Alpha", "Beta"]
  )
   # Flatten multi-index columns → Delta_Fp1, Theta_Fp1, etc.
   pivot_df.columns = [
          f"{band}_{channel}" for band, channel in pivot_df.columns
    ]
   pivot_df = pivot_df.reset_index().sort_values("Subject")
   return pivot_df
  else:
    raise ValueError("mode must be 'subject' or 'channel'")