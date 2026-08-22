#!/usr/bin/env python3
"""
Playground dataset builder. Reads cached Statcast parquets (no re-download)
and writes compact tensors under data/playground/{train,val,test}.pt.

Same 7/1/2 hash split as get_dataset.py. Adds pre-pitch Statcast context:
times-through-order, fielding alignment, official balls/strikes, pitcher
pitch count, previous-pitch kinematics, and padded at-bat sequences.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from get_dataset import (
    MIN_PA_FOR_CURRENT_SLASH,
    SeasonBatterStats,
    Vocab,
    assign_split,
    occupied,
    safe_int,
    safe_str,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLAY_DIR = DATA_DIR / "playground"
CACHE_DIR = Path.home() / ".pybaseball" / "cache"

SEQ_LEN = 8
PAD_CALL = 16
PAD_TYPE = 18

INNING_HALF = {"Top": 0, "Bot": 1}
THROWS_STAND = {"R": 0, "L": 1}
IF_ALIGN = {"": 0, "Standard": 1, "Infield shade": 2, "Strategic": 3}
OF_ALIGN = {"": 0, "Standard": 1, "Strategic": 2}

KEEP_COLS = [
    "game_pk",
    "game_date",
    "game_type",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "pitch_type",
    "description",
    "events",
    "balls",
    "strikes",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "bat_score",
    "fld_score",
    "inning",
    "inning_topbot",
    "p_throws",
    "stand",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "release_speed",
    "plate_x",
    "plate_z",
    "pfx_x",
    "pfx_z",
    "zone",
    "home_win_exp",
]


def load_statcast_cache() -> pd.DataFrame:
    files = sorted(CACHE_DIR.glob("get_statcast*.parquet"))
    if not files:
        raise SystemExit(f"No Statcast parquets in {CACHE_DIR}")
    frames = []
    for i, path in enumerate(files):
        df = pd.read_parquet(path)
        present = [c for c in KEEP_COLS if c in df.columns]
        frames.append(df[present])
        if i and i % 50 == 0:
            print(f"  loaded {i}/{len(files)} cache files")
    df = pd.concat(frames, ignore_index=True)
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"].copy()
    df = df.dropna(subset=["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"])
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    return df


def encode_align(table: dict[str, int], value) -> int:
    s = safe_str(value) or ""
    return table.get(s, 0)


def clip_int(value, lo, hi, default=0) -> int:
    n = safe_int(value)
    if n is None:
        return default
    return max(lo, min(hi, n))


def finite(value, default=0.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def add_shifted_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["game_pk", "at_bat_number"], sort=False)
    for col, fill in (
        ("release_speed", 0.0),
        ("plate_x", 0.0),
        ("plate_z", 0.0),
        ("pfx_x", 0.0),
        ("pfx_z", 0.0),
        ("zone", 0.0),
    ):
        df[f"prev_{col}"] = g[col].shift(1)
        df[f"prev_{col}"] = df[f"prev_{col}"].fillna(fill)
    df["pitcher_game_pitches"] = df.groupby(["game_pk", "pitcher"], sort=False).cumcount()
    return df


def add_sequences(df: pd.DataFrame, vocab: Vocab):
    type_ids = []
    call_ids = []
    for pt, desc in zip(df["pitch_type"].tolist(), df["description"].tolist()):
        type_ids.append(vocab.encode_pitch_type(safe_str(pt) or "UN"))
        call_ids.append(vocab.encode_pitch_call(safe_str(desc) or "unknown"))
    df["pitch_type_id"] = type_ids
    df["pitch_call_id"] = call_ids

    keys = list(zip(df["game_pk"].tolist(), df["at_bat_number"].tolist()))
    seq_types = []
    seq_calls = []
    seq_len = []
    cur = None
    t_hist: list[int] = []
    c_hist: list[int] = []
    for key, t, c in zip(keys, type_ids, call_ids):
        if key != cur:
            cur = key
            t_hist = []
            c_hist = []
        sl = min(len(t_hist), SEQ_LEN)
        pad_t = SEQ_LEN - sl
        pad_c = SEQ_LEN - sl
        seq_types.append([PAD_TYPE] * pad_t + t_hist[-SEQ_LEN:])
        seq_calls.append([PAD_CALL] * pad_c + c_hist[-SEQ_LEN:])
        seq_len.append(sl)
        t_hist.append(t)
        c_hist.append(c)
    df["seq_types"] = seq_types
    df["seq_calls"] = seq_calls
    df["seq_len"] = seq_len
    return df


def add_batter_slash(df: pd.DataFrame, batter_stats: SeasonBatterStats):
    avgs, obps, slgs = [], [], []
    dates = pd.to_datetime(df["game_date"]).dt.year.tolist()
    batters = df["batter"].tolist()
    events = df["events"].tolist()
    abs_n = df["at_bat_number"].tolist()
    pns = df["pitch_number"].tolist()
    last_ab = None
    pending = None
    for year, batter, event, ab, pn in zip(dates, batters, events, abs_n, pns):
        batter = int(batter)
        batter_stats.ensure_year(int(year))
        avg, obp, slg = batter_stats.rates_before(batter)
        avgs.append(avg)
        obps.append(obp)
        slgs.append(slg)
        ab_key = (int(ab), batter)
        if last_ab is not None and ab_key != last_ab and pending is not None:
            batter_stats.apply_event(pending[0], pending[1])
            pending = None
        last_ab = ab_key
        if event and not (isinstance(event, float) and math.isnan(event)):
            pending = (batter, str(event))
    if pending is not None:
        batter_stats.apply_event(pending[0], pending[1])
    df["batter_avg"] = avgs
    df["batter_obp"] = obps
    df["batter_slg"] = slgs
    return df


def rows_to_blob(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    numeric = []
    context = []
    for row in df.itertuples(index=False):
        offense = clip_int(row.bat_score, 0, 30)
        defense = clip_int(row.fld_score, 0, 30)
        numeric.append(
            [
                finite(row.batter_avg),
                finite(row.batter_obp),
                finite(row.batter_slg),
                float(offense),
                float(defense),
                float(offense - defense),
                float(clip_int(row.inning, 1, 20, 1)),
                float(clip_int(row.at_bat_number, 1, 120, 1)),
                float(clip_int(row.pitch_number, 1, 20, 1)),
                float(clip_int(row.pitcher_game_pitches, 0, 150)),
                float(clip_int(row.n_thruorder_pitcher, 1, 5, 1)),
                float(clip_int(row.n_priorpa_thisgame_player_at_bat, 0, 10)),
                finite(row.home_win_exp, 0.5),
                finite(row.prev_release_speed),
                finite(row.prev_plate_x),
                finite(row.prev_plate_z),
                finite(row.prev_pfx_x),
                finite(row.prev_pfx_z),
                finite(row.prev_zone),
            ]
        )
        context.append(
            [
                clip_int(row.outs_when_up, 0, 2),
                occupied(row.on_1b),
                occupied(row.on_2b),
                occupied(row.on_3b),
                INNING_HALF.get(safe_str(row.inning_topbot), 0),
                THROWS_STAND.get(safe_str(row.p_throws), 0),
                THROWS_STAND.get(safe_str(row.stand), 0),
                clip_int(row.balls, 0, 3),
                clip_int(row.strikes, 0, 2),
                encode_align(IF_ALIGN, row.if_fielding_alignment),
                encode_align(OF_ALIGN, row.of_fielding_alignment),
                clip_int(row.n_thruorder_pitcher, 1, 5, 1),
            ]
        )
    labels = [int(x) if 0 <= int(x) < 18 else 13 for x in df["pitch_type_id"].tolist()]
    return {
        "numeric": torch.tensor(numeric, dtype=torch.float32),
        "context": torch.tensor(context, dtype=torch.long),
        "seq_types": torch.tensor(df["seq_types"].tolist(), dtype=torch.long),
        "seq_calls": torch.tensor(df["seq_calls"].tolist(), dtype=torch.long),
        "seq_len": torch.tensor(df["seq_len"].tolist(), dtype=torch.long),
        "pitcher_mlbam": torch.tensor(df["pitcher"].astype("int64").tolist(), dtype=torch.long),
        "batter_mlbam": torch.tensor(df["batter"].astype("int64").tolist(), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> None:
    PLAY_DIR.mkdir(parents=True, exist_ok=True)
    meta_dir = DATA_DIR / "meta"
    vocab = Vocab(meta_dir / "vocabs.json")
    batter_stats = SeasonBatterStats(meta_dir=meta_dir)

    print("Loading Statcast cache ...")
    df = load_statcast_cache()
    print(f"  {len(df)} regular-season pitches")

    print("Kinematics / pitch counts ...")
    df = add_shifted_kinematics(df)

    print("Sequences + vocabs ...")
    df = add_sequences(df, vocab)
    vocab.save()

    print("Batter slash lines ...")
    df = add_batter_slash(df, batter_stats)

    splits: dict[str, list[int]] = defaultdict(list)
    pitchers = df["pitcher"].tolist()
    games = df["game_pk"].tolist()
    abs_n = df["at_bat_number"].tolist()
    pns = df["pitch_number"].tolist()
    for i, (p, g, ab, pn) in enumerate(zip(pitchers, games, abs_n, pns)):
        splits[assign_split(int(p), int(g), int(ab), int(pn))].append(i)

    for split, idx in splits.items():
        sub = df.iloc[idx].reset_index(drop=True)
        blob = rows_to_blob(sub)
        out = PLAY_DIR / f"{split}.pt"
        torch.save(blob, out)
        print(f"  {split}: {len(sub)} -> {out}")

    print("Done.")


if __name__ == "__main__":
    main()
