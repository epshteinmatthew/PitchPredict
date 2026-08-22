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
with open("stuff.json", "r") as f:
    stuff = json.load(f)

smean = torch.load('smean.pt', weights_only=True)
sstd = torch.load('sstd.pt', weights_only=True)


model = PitchModel(stuff['n_classes'], stuff['n_pitchers'], stuff['n_batters'])
model.load_state_dict(torch.load("pitch_predictor.pth", weights_only=True))

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
    return predict_situation(at_bat.model_dump(), stuff['vocabs'], stuff['type_to_id'], stuff['call_to_id'], stuff['pitcher_to_idx'], stuff['unk_pitcher'], stuff['batter_to_idx'], stuff['unk_batter'], smean, sstd, stuff['id_to_type'], model)

