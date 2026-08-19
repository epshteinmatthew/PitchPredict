#%%
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import torch
import json
from torch.utils.data import Dataset, DataLoader

PAD_CALL = 16  # pitch_call ids are 0-15
PAD_TYPE = 18  # pitch_type ids are 0-17
N_PITCH_TYPES = 18

STRIKE_CALLS = {0, 2, 8, 9, 11, 15}
FOUL_CALLS = {4, 5, 6, 12}
BALL_CALLS = {1, 3, 7, 10}

EVAL_PITCHER = 693433  # Bryan Woo — per-pitcher val slice
EVAL_BATTER = 643289
DATA_ROOT = Path("/home/matthew/Documents/PitchPredict/data")


def count_from_calls(calls):
    balls, strikes = 0, 0
    for c in calls:
        if c in BALL_CALLS:
            balls = min(3, balls + 1)
        elif c in FOUL_CALLS:
            if strikes < 2:
                strikes += 1
        elif c in STRIKE_CALLS:
            strikes = min(2, strikes + 1)
    return balls, strikes


def parse_pitch_file(path):
    with open(path) as file:
        item = json.load(file)
    calls = item["pitch_calls_so_far"]
    types = item["pitch_types_so_far"]
    balls, strikes = count_from_calls(calls)
    label = int(item["pitch_type"])
    if label < 0 or label >= N_PITCH_TYPES:
        label = 13  # UN
    numeric = [
        item["batter_avg"], item["batter_obp"], item["batter_slg"],
        item["offense_score"], item["defense_score"],
        item["inning"], item["at_bat_number"], item["pitch_number"],
        balls, strikes,
    ]
    context = [
        item["outs"], item["on_1b"], item["on_2b"], item["on_3b"],
        item["inning_half"], item["p_throws"], item["stand"],
    ]
    last_call = calls[-1] if calls else PAD_CALL
    last_type = types[-1] if types else PAD_TYPE
    return numeric, context, last_call, last_type, item["pitcher_id"], item["batter_id"], label


class PitchesDataset(Dataset):
    def __init__(self, root, pitcher_id=None, cache_path=None):
        root = Path(root)
        cache_path = Path(cache_path) if cache_path else root.parent / "cache" / f"{root.name}.pt"
        if cache_path.exists():
            blob = torch.load(cache_path, weights_only=True)
            self._set_from_blob(blob)
            print(f"{root.name}: {len(self)} pitches (cache)")
        else:
            self._load_json(root)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._blob(), cache_path)
            print(f"{root.name}: {len(self)} pitches (cached to {cache_path})")
        if pitcher_id is not None:
            self._keep(self.pitcher_mlbam == pitcher_id)

    def _load_json(self, root):
        paths = list(root.glob("*.json"))
        numeric, context, last_call, last_type, pitcher_mlbam, batter_mlbam, labels = (
            [], [], [], [], [], [], []
        )
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, row in enumerate(pool.map(parse_pitch_file, paths, chunksize=256)):
                if i and i % 50000 == 0:
                    print(f"  loaded {i}/{len(paths)}")
                n, c, lc, lt, pmlbam, bmlbam, label = row
                numeric.append(n)
                context.append(c)
                last_call.append(lc)
                last_type.append(lt)
                pitcher_mlbam.append(pmlbam)
                batter_mlbam.append(bmlbam)
                labels.append(label)
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.context = torch.tensor(context, dtype=torch.long)
        self.last_call = torch.tensor(last_call, dtype=torch.long)
        self.last_type = torch.tensor(last_type, dtype=torch.long)
        self.pitcher_mlbam = torch.tensor(pitcher_mlbam, dtype=torch.long)
        self.pitcher_idx = torch.zeros(len(labels), dtype=torch.long)
        self.batter_mlbam = torch.tensor(batter_mlbam, dtype=torch.long)
        self.batter_idx = torch.zeros(len(labels), dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def _blob(self):
        return {
            "numeric": self.numeric,
            "context": self.context,
            "last_call": self.last_call,
            "last_type": self.last_type,
            "pitcher_mlbam": self.pitcher_mlbam,
            "batter_mlbam": self.batter_mlbam,
            "labels": self.labels,
        }

    def _set_from_blob(self, blob):
        self.numeric = blob["numeric"]
        self.context = blob["context"]
        self.last_call = blob["last_call"]
        self.last_type = blob["last_type"]
        self.pitcher_mlbam = blob["pitcher_mlbam"]
        self.batter_mlbam = blob["batter_mlbam"]
        self.labels = blob["labels"]
        self.pitcher_idx = torch.zeros(len(self.labels), dtype=torch.long)
        self.batter_idx = torch.zeros(len(self.labels), dtype=torch.long)

    def _keep(self, mask):
        self.numeric = self.numeric[mask]
        self.context = self.context[mask]
        self.last_call = self.last_call[mask]
        self.last_type = self.last_type[mask]
        self.pitcher_mlbam = self.pitcher_mlbam[mask]
        self.pitcher_idx = self.pitcher_idx[mask]
        self.batter_mlbam = self.batter_mlbam[mask]
        self.batter_idx = self.batter_idx[mask]
        self.labels = self.labels[mask]

    def numeric_matrix(self):
        return self.numeric

    def standardize(self, mean, std):
        self.numeric = (self.numeric - mean) / std.clamp_min(1e-6)

    def assign_pitcher_idx(self, pitcher_to_idx, unk):
        hi = max(int(self.pitcher_mlbam.max()), max(pitcher_to_idx))
        table = torch.full((hi + 1,), unk, dtype=torch.long)
        for mlbam, i in pitcher_to_idx.items():
            table[mlbam] = i
        self.pitcher_idx = table[self.pitcher_mlbam]

    def assign_batter_idx(self, batter_to_idx, unk):
        hi = max(int(self.batter_mlbam.max()), max(batter_to_idx))
        table = torch.full((hi + 1,), unk, dtype=torch.long)
        for mlbam, i in batter_to_idx.items():
            table[mlbam] = i
        self.batter_idx = table[self.batter_mlbam]


    def subset_pitcher(self, mlbam):
        mask = self.pitcher_mlbam == mlbam
        sub = object.__new__(PitchesDataset)
        sub.numeric = self.numeric[mask]
        sub.context = self.context[mask]
        sub.last_call = self.last_call[mask]
        sub.last_type = self.last_type[mask]
        sub.pitcher_mlbam = self.pitcher_mlbam[mask]
        sub.pitcher_idx = self.pitcher_idx[mask]
        sub.batter_idx = self.batter_idx[mask]
        sub.labels = self.labels[mask]
        return sub

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return {
            "numeric": self.numeric[idx],
            "context": self.context[idx],
            "last_call": self.last_call[idx],
            "last_type": self.last_type[idx],
            "pitcher_idx": self.pitcher_idx[idx],
            "batter_idx": self.batter_idx[idx]
        }, self.labels[idx]
