import pandas as pd
import os
import mne

# Participants info 
participants_dir = os.path.join(os.path.dirname(__file__), "..", "Dataset", "ds004504")
participants_df = pd.read_csv(os.path.join(participants_dir, "participants.tsv"), sep="\t")

p_folder_names = []

for row in participants_df.itertuples(index=False):
  p_id = row.participant_id
  p_group = row.Group

  if p_group == "A" or p_group == "C":
    p_folder_names.append(p_id)

# Load data from A(Alz) and C(Healthy)
data_dir = os.path.join(os.path.dirname(__file__),"..", "Dataset", "ds004504", "derivatives")
bands = {"delta":(1,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30)}

all_data = []

for sub in os.listdir(data_dir):
  if sub not in p_folder_names:
    continue

  sub_path = os.path.join(data_dir, sub, "eeg")
  eeg_file = [f for f in os.listdir(sub_path) if f.endswith(".set")][0]
  raw = mne.io.read_raw_eeglab(os.path.join(sub_path, eeg_file), preload=True)
  raw.pick("eeg")
  raw.filter(1,40)

  spectrum = raw.compute_psd(
    method="welch",
    fmin=1,
    fmax=40,
    n_fft=2048
  )

  psd = spectrum.get_data()
  freqs = spectrum.freqs
    
  # Compute band power
  band_power = {}
  for band, (fmin,fmax) in bands.items():
    idx = (freqs >= fmin) & (freqs <= fmax)
    band_power[band] = psd[:, idx].mean(axis=1)

  for i in range(len(raw.ch_names)):
    ch = raw.ch_names[i]
    d = band_power["delta"][i]
    t = band_power["theta"][i]
    a = band_power["alpha"][i]
    b = band_power["beta"][i]
    all_data.append([sub, ch, d, t, a, b])

df = pd.DataFrame(all_data, columns=["Subject","Channel","Delta","Theta","Alpha","Beta"])

# Merge with participants info to get group labels
df_final = df.merge(
  participants_df[["participant_id","Group"]],
  left_on="Subject",
  right_on="participant_id"
)

df_final = df_final.drop(columns=["participant_id"])

# Group by channel and group, then calculate mean power for each band
summary = df_final.groupby(['Channel','Group'])[['Alpha','Theta','Delta', 'Beta']].mean().reset_index()
print(summary)