#!/usr/bin/env python3
"""
Download MLB Statcast pitch-level data and write one JSON file per pitch under:

  data/train/
  data/val/
  data/test/

Split is 7/10 train, 1/10 val, 2/10 test, assigned per-pitch with a deterministic
hash so each pitcher appears in all splits (not pitcher-stratified into one fold).

ML-oriented schema: all fields numeric. Labels are pitch_type and
outcome_type (Statcast description), each encoded via persisted vocabs.
outcome_type uses the same map as pitch_calls_so_far. Within each at-bat,
prior pitches are two parallel int arrays (pitch_calls_so_far,
pitch_types_so_far) — not ball/strike counts. Batter identity is
season-to-date batter_avg / batter_obp / batter_slg (no future leakage).
If a batter has fewer than 100 plate appearances in the current season,
use last season's slash line when available.

Use --start/--end for small test runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from pybaseball import batting_stats_bref, cache, statcast

MIN_PA_FOR_CURRENT_SLASH = 100

SEASON_RANGES = {
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
    2026: (date(2026, 3, 26), date(2026, 9, 27)),
}

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

INNING_HALF = {"Top": 0, "Bot": 1}
THROWS_STAND = {"R": 0, "L": 1}

# Total bases by hit type.
HIT_TB = {"single": 1, "double": 2, "triple": 3, "home_run": 4}

# Events that are not at-bats (still may count for OBP).
NON_AB_EVENTS = {
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "sac_fly",
    "sac_bunt",
    "sac_fly_double_play",
    "sac_bunt_double_play",
    "catcher_interf",
}

# Events that are not plate appearances for rate stats (ignore if seen).
IGNORE_EVENTS = {
    "caught_stealing_2b",
    "caught_stealing_3b",
    "caught_stealing_home",
    "pickoff_1b",
    "pickoff_2b",
    "pickoff_3b",
    "pickoff_caught_stealing_2b",
    "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "runner_double_play",
    "other_out",
}

KEEP_COLS = [
    "game_pk",
    "game_date",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
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
    "pitch_type",
    "description",
    "events",
    "game_type",
]


class Vocab:
    """String -> int maps for pitch type and pitch call / outcome (persisted)."""

    def __init__(self, path: Path):
        self.path = path
        self.pitch_type: dict[str, int] = {}
        self.pitch_call: dict[str, int] = {}
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.pitch_type = {
                str(k): int(v) for k, v in loaded.get("pitch_type", {}).items()
            }
            self.pitch_call = {
                str(k): int(v) for k, v in loaded.get("pitch_call", {}).items()
            }

    def _encode(self, table: dict[str, int], value: str | None, default: str) -> int:
        if value is None:
            value = default
        value = str(value)
        if value not in table:
            table[value] = len(table)
        return table[value]

    def encode_pitch_type(self, value: str | None) -> int:
        return self._encode(self.pitch_type, value, "UN")

    def encode_pitch_call(self, value: str | None) -> int:
        return self._encode(self.pitch_call, value, "unknown")

    def encode_outcome_type(self, value: str | None) -> int:
        return self.encode_pitch_call(value)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "pitch_type": self.pitch_type,
                    "pitch_call": self.pitch_call,
                    "outcome_type": self.pitch_call,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


@dataclass
class BatterLine:
    pa: int = 0
    ab: int = 0
    h: int = 0
    tb: int = 0
    bb: int = 0
    hbp: int = 0
    sf: int = 0

    def rates(self) -> tuple[float, float, float]:
        avg = (self.h / self.ab) if self.ab else 0.0
        obp_den = self.ab + self.bb + self.hbp + self.sf
        obp = ((self.h + self.bb + self.hbp) / obp_den) if obp_den else 0.0
        slg = (self.tb / self.ab) if self.ab else 0.0
        return round(avg, 3), round(obp, 3), round(slg, 3)

    def update(self, event: str) -> None:
        if event in IGNORE_EVENTS:
            return
        if event in HIT_TB:
            self.pa += 1
            self.ab += 1
            self.h += 1
            self.tb += HIT_TB[event]
            return
        if event in ("walk", "intent_walk"):
            self.pa += 1
            self.bb += 1
            return
        if event == "hit_by_pitch":
            self.pa += 1
            self.hbp += 1
            return
        if event in ("sac_fly", "sac_fly_double_play"):
            self.pa += 1
            self.sf += 1
            return
        if event in ("sac_bunt", "sac_bunt_double_play", "catcher_interf"):
            self.pa += 1
            return
        # Default PA ending in an at-bat (K, field_out, GIDP, etc.)
        if event and event not in NON_AB_EVENTS:
            self.pa += 1
            self.ab += 1


def _finite_rate(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def load_prior_season_slash(season: int, cache_dir: Path) -> dict[int, tuple[float, float, float]]:
    """
    Full-season AVG/OBP/SLG for `season`, keyed by MLBAM id.
    Cached under data/meta so we do not re-scrape Baseball Reference every run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"prior_slash_{season}.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return {
            int(k): (float(v[0]), float(v[1]), float(v[2])) for k, v in raw.items()
        }

    print(f"  loading {season} season slash lines (Baseball Reference) ...")
    try:
        df = batting_stats_bref(season)
    except Exception as exc:  # noqa: BLE001
        print(f"  warning: prior-season slash load failed for {season}: {exc}")
        return {}

    prior: dict[int, tuple[float, float, float]] = {}
    for _, row in df.iterrows():
        batter_id = safe_int(row.get("mlbID"))
        if batter_id is None:
            continue
        pa = safe_int(row.get("PA")) or 0
        if pa <= 0:
            continue
        avg = _finite_rate(row.get("BA"))
        obp = _finite_rate(row.get("OBP"))
        slg = _finite_rate(row.get("SLG"))
        if avg is None or obp is None or slg is None:
            continue
        prior[batter_id] = (avg, obp, slg)

    cache_path.write_text(
        json.dumps({str(k): list(v) for k, v in prior.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"  cached {len(prior)} batter slash lines -> {cache_path}")
    return prior


@dataclass
class SeasonBatterStats:
    """Season-to-date batter lines keyed by batter_id, reset each year."""

    year: int | None = None
    lines: dict[int, BatterLine] = field(default_factory=dict)
    prior: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    meta_dir: Path | None = None

    def ensure_year(self, year: int) -> None:
        if self.year != year:
            self.year = year
            self.lines.clear()
            if self.meta_dir is not None:
                self.prior = load_prior_season_slash(year - 1, self.meta_dir)
            else:
                self.prior = {}

    def rates_before(self, batter_id: int) -> tuple[float, float, float]:
        line = self.lines.get(batter_id, BatterLine())
        if line.pa < MIN_PA_FOR_CURRENT_SLASH and batter_id in self.prior:
            return self.prior[batter_id]
        return line.rates()

    def apply_event(self, batter_id: int, event: str | None) -> None:
        if not event:
            return
        line = self.lines.setdefault(batter_id, BatterLine())
        line.update(event)


def parse_args() -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(
        description="Build train/val/test pitch JSON dataset from MLB Statcast."
    )
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--chunk-days", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing pitch JSON files",
    )
    args = parser.parse_args()

    if args.start is None and args.end is None:
        args.start = SEASON_RANGES[2025][0]
        args.end = min(today, SEASON_RANGES[2026][1])
    else:
        args.start = (
            datetime.strptime(args.start, "%Y-%m-%d").date()
            if args.start
            else SEASON_RANGES[2025][0]
        )
        args.end = (
            datetime.strptime(args.end, "%Y-%m-%d").date()
            if args.end
            else min(today, SEASON_RANGES[2026][1])
        )

    if args.start > args.end:
        raise SystemExit("--start must be on or before --end")
    return args


def daterange_chunks(start: date, end: date, chunk_days: int):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def safe_int(value):
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_str(value):
    if pd.isna(value):
        return None
    return str(value)


def game_date_int(value) -> int | None:
    if pd.isna(value):
        return None
    if hasattr(value, "strftime"):
        return int(value.strftime("%Y%m%d"))
    s = str(value)[:10].replace("-", "")
    try:
        return int(s)
    except ValueError:
        return None


def occupied(value) -> int:
    return 0 if pd.isna(value) else 1


def encode_binary(table: dict[str, int], value) -> int:
    s = safe_str(value)
    if s is None:
        return -1
    return table.get(s, -1)


def assign_split(pitcher_id: int, game_pk: int, at_bat: int, pitch_number: int) -> str:
    """
    7/10 train, 1/10 val, 2/10 test.
    Hash includes pitcher_id so each pitcher's pitches are spread across splits.
    """
    key = f"{pitcher_id}:{game_pk}:{at_bat}:{pitch_number}".encode()
    bucket = int(hashlib.md5(key).hexdigest(), 16) % 10
    if bucket < 7:
        return "train"
    if bucket == 7:
        return "val"
    return "test"


def fetch_chunk(start: date, end: date) -> pd.DataFrame:
    print(f"Fetching Statcast {start} -> {end} ...")
    df = statcast(
        start_dt=start.isoformat(),
        end_dt=end.isoformat(),
        verbose=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"].copy()

    present = [c for c in KEEP_COLS if c in df.columns]
    df = df[present].copy()

    for col in ("game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"])
    df = df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)
    return df


def add_prior_pitch_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pitch, attach prior pitches in the same at-bat as parallel
    string lists (call + type). Empty on the first pitch of an AB.
    """
    if df.empty:
        df = df.copy()
        df["pitch_calls_so_far"] = []
        df["pitch_types_so_far"] = []
        return df

    call_histories: list[list[str]] = []
    type_histories: list[list[str]] = []
    for (_, _), group in df.groupby(["game_pk", "at_bat_number"], sort=False):
        calls_so_far: list[str] = []
        types_so_far: list[str] = []
        for _, row in group.iterrows():
            call_histories.append(list(calls_so_far))
            type_histories.append(list(types_so_far))
            calls_so_far.append(safe_str(row.get("description")) or "unknown")
            types_so_far.append(safe_str(row.get("pitch_type")) or "UN")

    df = df.copy()
    df["pitch_calls_so_far"] = call_histories
    df["pitch_types_so_far"] = type_histories
    return df


def row_to_record(
    row: pd.Series,
    vocab: Vocab,
    batter_avg: float,
    batter_obp: float,
    batter_slg: float,
) -> dict:
    pitcher_id = safe_int(row.get("pitcher"))
    batter_id = safe_int(row.get("batter"))
    pitch_type = safe_str(row.get("pitch_type")) or "UN"
    description = safe_str(row.get("description")) or "unknown"

    prior_calls = row.get("pitch_calls_so_far") or []
    prior_types = row.get("pitch_types_so_far") or []
    call_codes = [vocab.encode_pitch_call(c) for c in prior_calls]
    type_codes = [vocab.encode_pitch_type(pt) for pt in prior_types]

    return {
        "game_date": game_date_int(row.get("game_date")),
        "at_bat_number": safe_int(row.get("at_bat_number")),
        "pitch_number": safe_int(row.get("pitch_number")),
        "pitcher_id": pitcher_id,
        "batter_id" : batter_id,
        "batter_avg": batter_avg,
        "batter_obp": batter_obp,
        "batter_slg": batter_slg,
        "pitch_calls_so_far": call_codes,
        "pitch_types_so_far": type_codes,
        "outs": safe_int(row.get("outs_when_up")),
        "on_1b": occupied(row.get("on_1b")),
        "on_2b": occupied(row.get("on_2b")),
        "on_3b": occupied(row.get("on_3b")),
        "offense_score": safe_int(row.get("bat_score")),
        "defense_score": safe_int(row.get("fld_score")),
        "inning": safe_int(row.get("inning")),
        "inning_half": encode_binary(INNING_HALF, row.get("inning_topbot")),
        "p_throws": encode_binary(THROWS_STAND, row.get("p_throws")),
        "stand": encode_binary(THROWS_STAND, row.get("stand")),
        "pitch_type": vocab.encode_pitch_type(pitch_type),
        "outcome_type": vocab.encode_outcome_type(description),
    }


def write_pitch_files(
    df: pd.DataFrame,
    data_dir: Path,
    vocab: Vocab,
    batter_stats: SeasonBatterStats,
    overwrite: bool,
) -> int:
    written = 0
    for split in ("train", "val", "test"):
        (data_dir / split).mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        pitcher_id = safe_int(row.get("pitcher"))
        batter_id = safe_int(row.get("batter"))
        game_pk = safe_int(row.get("game_pk"))
        ab = safe_int(row.get("at_bat_number"))
        pn = safe_int(row.get("pitch_number"))
        gdate = game_date_int(row.get("game_date"))
        if (
            pitcher_id is None
            or batter_id is None
            or game_pk is None
            or ab is None
            or pn is None
            or gdate is None
        ):
            continue

        year = gdate // 10000
        batter_stats.ensure_year(year)
        avg, obp, slg = batter_stats.rates_before(batter_id)

        split = assign_split(pitcher_id, game_pk, ab, pn)
        # Filename keeps game_pk for uniqueness; not stored in JSON.
        out_path = data_dir / split / f"{game_pk}_{ab}_{pn}.json"
        if out_path.exists() and not overwrite:
            # Still advance batter stats so later pitches stay correct.
            batter_stats.apply_event(batter_id, safe_str(row.get("events")))
            continue

        record = row_to_record(row, vocab, avg, obp, slg)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, separators=(",", ":"))
        written += 1

        batter_stats.apply_event(batter_id, safe_str(row.get("events")))

    return written


def process_chunk(
    df: pd.DataFrame,
    data_dir: Path,
    vocab: Vocab,
    batter_stats: SeasonBatterStats,
    overwrite: bool,
    pitcher_ids: set[int],
) -> int:
    if df.empty:
        print("  no rows")
        return 0

    if "pitcher" in df.columns:
        pitcher_ids.update(
            int(x) for x in df["pitcher"].dropna().unique() if pd.notna(x)
        )

    df = add_prior_pitch_history(df)
    written = write_pitch_files(df, data_dir, vocab, batter_stats, overwrite)
    vocab.save()
    print(f"  wrote {written} new files ({len(df)} pitches in chunk)")
    return written


def main() -> None:
    args = parse_args()
    cache.enable()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (data_dir / split).mkdir(parents=True, exist_ok=True)

    meta_dir = data_dir / "meta"
    vocab = Vocab(meta_dir / "vocabs.json")
    batter_stats = SeasonBatterStats(meta_dir=meta_dir)
    pitcher_ids: set[int] = set()

    print(f"Output: {data_dir}/{{train,val,test}}")
    print(f"Range:  {args.start} -> {args.end}")
    print(f"Chunks: {args.chunk_days} day(s)")
    print("Split:  7/10 train, 1/10 val, 2/10 test (per-pitch, per-pitcher balanced)")
    print(
        f"Batter slash: prior season if PA < {MIN_PA_FOR_CURRENT_SLASH}, else season-to-date"
    )

    total_written = 0
    for start, end in daterange_chunks(args.start, args.end, args.chunk_days):
        try:
            df = fetch_chunk(start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  error fetching {start}->{end}: {exc}")
            time.sleep(args.sleep * 2)
            continue

        total_written += process_chunk(
            df, data_dir, vocab, batter_stats, args.overwrite, pitcher_ids
        )
        time.sleep(args.sleep)

    vocab.save()
    print(f"Done. Wrote {total_written} pitch JSON files.")
    print(f"Unique pitchers: {len(pitcher_ids)}")
    print(f"Pitch-type vocab size: {len(vocab.pitch_type)}")
    print(f"Outcome-type / pitch-call vocab size: {len(vocab.pitch_call)}")


if __name__ == "__main__":
    main()
