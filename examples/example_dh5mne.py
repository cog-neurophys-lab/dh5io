# -*- coding: utf-8 -*-
"""Example: load a DH5 file with MNE, create epochs, and compute trial-averaged responses.

Analysis choices
----------------
- Only trials with StimNo 40–45 and Outcome=0 (Hit) are included.
- Epochs are aligned to EV02 trigger 5 (not to trial onset).
- Epoch window: -0.5 s to +1.5 s around trigger 5.
- Baseline correction over the pre-trigger interval (-0.5 s to 0 s).

Steps
-----
1. Read CONT1 as a lazy MNE Raw object (no data copied into RAM yet).
2. Build epochs aligned to trigger 5 that fall inside selected trials.
3. Compute trial-averaged (evoked) response per StimNo condition.
4. Plot the grand-average and per-condition evoked responses.
"""

import pathlib

import matplotlib.pyplot as plt
import mne
import numpy as np

from dh5io.dh5mne import read_raw_dh5

EXAMPLE_FILE = pathlib.Path(__file__).parent / "example.dh5"

STIM_NOS = range(40, 46)  # StimNo 40, 41, 42, 43, 44, 45
OUTCOME = 0  # Hit
TRIGGER = 5  # EV02 event code to align on
TMIN = -0.5  # seconds before trigger
TMAX = 1.5  # seconds after trigger
BASELINE = (-0.5, 0.0)  # pre-trigger baseline

# ---------------------------------------------------------------------------
# 1. Load the file as a lazy Raw object
# ---------------------------------------------------------------------------
raw = read_raw_dh5(EXAMPLE_FILE, cont_ids=[1], preload=False)
print(raw)
print(f"Sampling frequency : {raw.info['sfreq']} Hz")
print(f"Number of channels : {raw.info['nchan']}")
print(f"Total duration     : {raw.times[-1]:.1f} s")

# ---------------------------------------------------------------------------
# 2. Build epochs aligned to trigger 5, restricted to selected trials
#
#    Strategy:
#    a) Extract all trigger-5 event times from the EV02 annotations.
#    b) For each trigger, check whether it falls inside a trial annotation
#       with the desired StimNo and Outcome.
#    c) Keep only those triggers; build mne.Epochs from the filtered set.
# ---------------------------------------------------------------------------

# Collect trial windows for the desired conditions: {stim_no: [(onset, offset), ...]}
trial_windows: dict[int, list[tuple[float, float]]] = {sn: [] for sn in STIM_NOS}
for ann in raw.annotations:
    desc = str(ann["description"])
    if not desc.startswith("trial/"):
        continue
    parts = {kv.split("=")[0]: kv.split("=")[1] for kv in desc.split("/")[1:]}
    sn = int(parts["StimNo"])
    oc = int(parts["Outcome"])
    if sn in STIM_NOS and oc == OUTCOME:
        onset = float(ann["onset"])
        trial_windows[sn].append((onset, onset + float(ann["duration"])))

n_selected_trials = sum(len(v) for v in trial_windows.values())
print(
    f"\nSelected trials (StimNo {min(STIM_NOS)}–{max(STIM_NOS)}, Outcome={OUTCOME}): "
    f"{n_selected_trials}"
)
for sn in STIM_NOS:
    print(f"  StimNo={sn}: {len(trial_windows[sn])} trials")

# Get all trigger-5 event times from the "event/5" annotations
trigger_times = np.array(
    [
        float(ann["onset"])
        for ann in raw.annotations
        if str(ann["description"]) == f"event/{TRIGGER}"
    ]
)
print(f"\nTotal trigger-{TRIGGER} events in file: {len(trigger_times)}")

# For each trigger, find which StimNo trial (if any) contains it
trigger_stim: list[tuple[float, int]] = []  # (trigger_time, stim_no)
for t in trigger_times:
    for sn, windows in trial_windows.items():
        for t_start, t_end in windows:
            if t_start <= t <= t_end:
                trigger_stim.append((t, sn))
                break
        else:
            continue
        break

print(f"Trigger-{TRIGGER} events inside selected trials: {len(trigger_stim)}")

# Build MNE events array from the filtered triggers: [sample, 0, stim_no]
sfreq = raw.info["sfreq"]
events = np.array(
    [[int(round(t * sfreq)), 0, sn] for t, sn in trigger_stim],
    dtype=np.int64,
)
event_id = {f"StimNo={sn}": sn for sn in STIM_NOS if sn in events[:, 2]}

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=TMIN,
    tmax=TMAX,
    baseline=BASELINE,
    preload=True,
    verbose=False,
)
print(
    f"\nEpochs: {len(epochs)} kept × {epochs.info['nchan']} ch × {len(epochs.times)} samples"
)

# ---------------------------------------------------------------------------
# 3. Compute evoked response per StimNo condition
# ---------------------------------------------------------------------------
evokeds: dict[int, mne.Evoked] = {}
for sn in STIM_NOS:
    key = f"StimNo={sn}"
    if key not in event_id:
        continue
    subset = epochs[key]
    if len(subset) == 0:
        continue
    evoked = subset.average()
    evoked.comment = f"StimNo={sn} (n={len(subset)})"
    evokeds[sn] = evoked
    print(f"  StimNo={sn}: {len(subset):3d} epochs → evoked")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
grand_avg = mne.grand_average(list(evokeds.values()))
grand_avg.comment = "Grand average"

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Top: grand average
axes[0].plot(grand_avg.times, grand_avg.data[0] * 1e6, color="black", linewidth=1.5)
axes[0].axvline(0, color="grey", linestyle="--", linewidth=0.8, label="trigger 5")
axes[0].axhline(0, color="grey", linestyle="-", linewidth=0.5)
axes[0].set_ylabel("Amplitude (µV)")
axes[0].set_title(
    f"Grand-average evoked response — StimNo {min(STIM_NOS)}–{max(STIM_NOS)}, "
    f"Outcome={OUTCOME}, aligned on trigger {TRIGGER}"
)
axes[0].legend(fontsize=8)

# Bottom: per-condition
cmap = plt.cm.tab10
for idx, (sn, evoked) in enumerate(sorted(evokeds.items())):
    axes[1].plot(
        evoked.times,
        evoked.data[0] * 1e6,
        label=evoked.comment,
        color=cmap(idx % 10),
        linewidth=1.0,
    )
axes[1].axvline(0, color="grey", linestyle="--", linewidth=0.8)
axes[1].axhline(0, color="grey", linestyle="-", linewidth=0.5)
axes[1].set_xlabel("Time relative to trigger 5 (s)")
axes[1].set_ylabel("Amplitude (µV)")
axes[1].set_title("Per-condition evoked responses")
axes[1].legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(pathlib.Path(__file__).parent / "example_dh5mne.png", dpi=150)
plt.show()
