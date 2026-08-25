import json

import torch

from training import DATA_ROOT, count_from_calls, EVAL_PITCHER, EVAL_BATTER, PAD_CALL, PAD_TYPE

batch_size = 256




def iter_batches(ds, batch_size, shuffle=False):
    n = len(ds)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        sl = order[start:start + batch_size]
        yield {
            "numeric": ds.numeric[sl],
            "context": ds.context[sl],
            "last_call": ds.last_call[sl],
            "last_type": ds.last_type[sl],
            "pitcher_idx": ds.pitcher_idx[sl],
            "batter_idx": ds.batter_idx[sl]
        }, ds.labels[sl]


def topk_hits(pred, y, k=2):
    top = pred.topk(k, dim=1).indices
    hit1 = (top[:, 0] == y).sum().item()
    hit2 = (top == y.unsqueeze(1)).any(dim=1).sum().item()
    return hit1, hit2


def train_loop(dataset, model, loss_fn, optimizer):
    size = len(dataset)
    num_batches = max(1, (size + batch_size - 1) // batch_size)
    model.train()
    train_loss, hit1, hit2 = 0, 0, 0
    for batch, (X, y) in enumerate(iter_batches(dataset, batch_size, shuffle=True)):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        b1, b2 = topk_hits(pred, y, k=3)
        hit1 += b1
        hit2 += b2

        if batch % 500 == 0:
            current = batch * batch_size + len(y)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    print(
        f"Train: top-1 {(100*hit1/size):>0.1f}%, top-3 {(100*hit2/size):>0.1f}%, "
        f"Avg loss: {train_loss:>8f}"
    )


def test_loop(dataset, model, loss_fn, split="Val"):
    size = len(dataset)
    if size == 0:
        print(f"{split}: empty")
        return
    num_batches = max(1, (size + batch_size - 1) // batch_size)
    model.eval()
    test_loss, hit1, hit2 = 0, 0, 0

    with torch.no_grad():
        for X, y in iter_batches(dataset, batch_size, shuffle=False):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            b1, b2 = topk_hits(pred, y, k=3)
            hit1 += b1
            hit2 += b2

    test_loss /= num_batches
    print(
        f"{split}: top-1 {(100*hit1/size):>0.1f}%, top-3 {(100*hit2/size):>0.1f}%, "
        f"Avg loss: {test_loss:>8f}"
    )



def predict_situation(situation, vocabs, TYPE_TO_ID, CALL_TO_ID, pitcher_to_idx, unk_pitcher, batter_to_idx, unk_batter , mean, std, ID_TO_TYPE, model):
        """situation uses human field names; numeric fields are standardized with train mean/std."""
        calls = situation.get("pitch_calls_so_far")
        types = situation.get("pitch_types_so_far")
        balls, strikes = count_from_calls(calls)
        mlbam = situation.get("pitcher_id", EVAL_PITCHER)
        bmlbam = situation.get("batter_id", EVAL_BATTER)
        #honestly im not sure how well this works but we KNOW 0 won't be out of range. need to think of a better solution
        pitcher_idx = pitcher_to_idx.get(str(mlbam), 0)
        batter_idx = batter_to_idx.get(str(bmlbam), 0)

        numeric = torch.tensor([
            situation["batter_avg"],
            situation["batter_obp"],
            situation["batter_slg"],
            situation["offense_score"],
            situation["defense_score"],
            situation["inning"],
            situation["at_bat_number"],
            situation["pitch_number"]
        ], dtype=torch.float32)
        numeric = (numeric - mean) / std.clamp_min(1e-6)

        batch = {
            "numeric": numeric.unsqueeze(0),
            "context": torch.tensor([[
                situation["outs"],
                situation["on_1b"],
                situation["on_2b"],
                situation["on_3b"],
                situation["inning_half"],  # 0=Top, 1=Bot
                situation["p_throws"],  # 0=R, 1=L
                situation["stand"],  # 0=R, 1=L
                balls,
                strikes,
            ]], dtype=torch.long),
            "last_call": torch.tensor([calls[-1] if calls else PAD_CALL], dtype=torch.long),
            "last_type": torch.tensor([types[-1] if types else PAD_TYPE], dtype=torch.long),
            "last_call2": torch.tensor([calls[-2] if len(calls) > 1 else PAD_CALL], dtype=torch.long),
            "last_type2": torch.tensor([types[-2] if len(types) > 1 else PAD_TYPE], dtype=torch.long),
            "last_call3": torch.tensor([calls[-3] if len(calls) > 2 else PAD_CALL], dtype=torch.long),
            "last_type3": torch.tensor([types[-3] if len(types) > 2 else PAD_TYPE], dtype=torch.long),
            "pitcher_idx": torch.tensor([pitcher_idx], dtype=torch.long),
            "batter_idx": torch.tensor([batter_idx], dtype=torch.long),
        }

        model.eval()
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)[0]

        ranked = sorted(
            ((ID_TO_TYPE.get(i, str(i)), float(probs[i])) for i in range(len(probs))),
            key=lambda x: -x[1],
        )
        last_type_name = ID_TO_TYPE.get(types[-1], "none") if types else "none"
        id_to_call = {int(v): k for k, v in CALL_TO_ID.items()}
        last_call_name = id_to_call.get(calls[-1], "none") if calls else "none"
        print(f"Count: {balls}-{strikes}  |  last: {last_type_name} / {last_call_name}")
        print(f"Best guess: {ranked[0][0]}  ({100 * ranked[0][1]:.1f}%)")
        print("Likelihoods:")
        for name, p in ranked:
            if p < 0.005:
                continue
            print(f"  {name:4s}  {100 * p:5.1f}%")
        return ranked