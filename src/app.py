#%%
import json
from collections import Counter

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


import torch
from torch import nn

from interface import train_loop, test_loop, predict_situation
from model import PitchModel
from training import PitchesDataset, DATA_ROOT, N_PITCH_TYPES, EVAL_PITCHER, PAD_CALL, PAD_TYPE, count_from_calls, \
    EVAL_BATTER

#%%
print("Loading all pitchers (JSON once, then cache)...")
train = PitchesDataset(DATA_ROOT / "train")
val = PitchesDataset(DATA_ROOT / "val")

pitcher_ids = sorted(set(train.pitcher_mlbam.tolist()))
pitcher_to_idx = {mlbam: i for i, mlbam in enumerate(pitcher_ids)}
unk_pitcher = len(pitcher_to_idx)
n_pitchers = unk_pitcher + 1
n_classes = N_PITCH_TYPES

train.assign_pitcher_idx(pitcher_to_idx, unk_pitcher)
val.assign_pitcher_idx(pitcher_to_idx, unk_pitcher)

batter_ids = sorted(set(train.batter_mlbam.tolist()))
batter_to_idx = {mlbam: i for i, mlbam in enumerate(batter_ids)}
unk_batter = len(batter_to_idx)
n_batters = unk_batter + 1

train.assign_batter_idx(batter_to_idx, unk_batter)
val.assign_batter_idx(batter_to_idx, unk_batter)

label_counts = Counter(train.labels.tolist())
majority_id, majority_n = label_counts.most_common(1)[0]
print(f"Pitchers: {len(pitcher_ids)}  (+1 unk)   train pitches: {len(train)}")
print(f"League majority class {majority_id}: {100 * majority_n / len(train):.1f}%")
league_top2 = sum(n for _, n in label_counts.most_common(2))
print(f"League majority top-2: {100 * league_top2 / len(train):.1f}%")

woo_train = train.subset_pitcher(EVAL_PITCHER)
if len(woo_train):
    woo_counts = Counter(woo_train.labels.tolist())
    w_id, w_n = woo_counts.most_common(1)[0]
    print(f"Woo train pitches: {len(woo_train)}  majority {w_id}: {100 * w_n / len(woo_train):.1f}%")
    woo_top2 = sum(n for _, n in woo_counts.most_common(2))
    print(f"Woo majority top-2: {100 * woo_top2 / len(woo_train):.1f}%")

mean = train.numeric.mean(dim=0)
std = train.numeric.std(dim=0)
train.standardize(mean, std)
val.standardize(mean, std)

woo_val = val.subset_pitcher(EVAL_PITCHER)
print(f"Woo val pitches: {len(woo_val)}")


model = PitchModel(n_classes, n_pitchers, n_batters)
model.load_state_dict(torch.load("pitch_predictor.pth", weights_only=True))


def train():
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train, model, loss_fn, optimizer)
        test_loop(val, model, loss_fn, split="Val (all)")
        test_loop(woo_val, model, loss_fn, split="Val (Woo)")
        print()
    print("Done!")


#%%
vocabs = json.loads((DATA_ROOT / "meta" / "vocabs.json").read_text())
TYPE_TO_ID = vocabs["pitch_type"]
CALL_TO_ID = vocabs["pitch_call"]
ID_TO_TYPE = {int(v): k for k, v in TYPE_TO_ID.items()}


class AtBat(BaseModel):
    game_date:int
    at_bat_number: int
    pitch_number:int
    pitcher_id: int
    batter_id: int
    batter_avg: float
    batter_obp: float
    batter_slg: float
    pitch_calls_so_far: list[int]
    pitch_types_so_far: list[int]
    outs: int
    on_1b: int
    on_2b: int
    on_3b: int
    offense_score: int
    defense_score: int
    inning: int
    inning_half: int
    p_throws:int
    stand: int

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict/")
def read_item(at_bat: AtBat):
    return predict_situation(at_bat.model_dump(), vocabs, TYPE_TO_ID, CALL_TO_ID, pitcher_to_idx, unk_pitcher, batter_to_idx, unk_batter, mean, std, ID_TO_TYPE, model)

