#%%
import os
import torch
from torch import nn

from training import PAD_CALL, PAD_TYPE

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class PitchModel(nn.Module):
    def __init__(self, n_classes, n_pitchers, n_batters):
        super().__init__()

        self.pitcher_embed = nn.Embedding(num_embeddings=n_pitchers, embedding_dim=16)
        self.batter_embed = nn.Embedding(num_embeddings=n_batters, embedding_dim=16)
        self.p_throws_embed = nn.Embedding(num_embeddings=2, embedding_dim=2)
        self.half_embed = nn.Embedding(num_embeddings=2, embedding_dim=2)
        self.outs_embed = nn.Embedding(num_embeddings=4, embedding_dim=2)
        self.on_1b_embed = nn.Embedding(num_embeddings=2, embedding_dim=2)
        self.on_2b_embed = nn.Embedding(num_embeddings=2, embedding_dim=2)
        self.on_3b_embed = nn.Embedding(num_embeddings=2, embedding_dim=2)
        self.stand_embed = nn.Embedding(num_embeddings=2, embedding_dim=2)


        self.balls_embed = nn.Embedding(num_embeddings=4, embedding_dim=8)
        self.strikes_embed = nn.Embedding(num_embeddings=3, embedding_dim=8)


        self.last_call_embed = nn.Embedding(num_embeddings=17, embedding_dim=4, padding_idx=PAD_CALL)
        self.last_type_embed = nn.Embedding(num_embeddings=19, embedding_dim=4, padding_idx=PAD_TYPE)

        self.last_call2_embed = nn.Embedding(num_embeddings=17, embedding_dim=4, padding_idx=PAD_CALL)
        self.last_type2_embed = nn.Embedding(num_embeddings=19, embedding_dim=4, padding_idx=PAD_TYPE)

        self.last_call3_embed = nn.Embedding(num_embeddings=17, embedding_dim=4, padding_idx=PAD_CALL)
        self.last_type3_embed = nn.Embedding(num_embeddings=19, embedding_dim=4, padding_idx=PAD_TYPE)

        # 10 numeric + pitcher 16 + 7 context*2 + last_call 4 + last_type 4
        total_dims = 8 + 16 +16 + 2 * 7 + 16 + 16

        # LSTM processes merged step features: (4 call dims + 4 type dims = 8 inputs per step)
        self.seq_lstm = nn.LSTM(
            input_size=8, hidden_size=16, num_layers=1, batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.reduce_dims = nn.Linear(total_dims, 64)
        self.linear_relu_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.output_dims = nn.Linear(32, n_classes)

    def forward(self, x):
        emb_pitcher = self.dropout(self.pitcher_embed(x["pitcher_idx"]))
        emb_batter = self.dropout(self.batter_embed(x["batter_idx"]))
        emb_outs = self.outs_embed(x["context"][:, 0])
        emb_on_1b = self.on_1b_embed(x["context"][:, 1])
        emb_on_2b = self.on_2b_embed(x["context"][:, 2])
        emb_on_3b = self.on_3b_embed(x["context"][:, 3])
        emb_half = self.half_embed(x["context"][:, 4])
        emb_throws = self.p_throws_embed(x["context"][:, 5])
        emb_stand = self.stand_embed(x["context"][:, 6])

        emb_balls = self.balls_embed(x["context"][:, 7]) * 3
        emb_strikes = self.strikes_embed(x["context"][:, 8]) * 3

        emb_last_call = self.last_call_embed(x["last_call"])
        emb_last_type = self.last_type_embed(x["last_type"])
        emb_last_call2 = self.last_call2_embed(x["last_call2"])
        emb_last_type2 = self.last_type2_embed(x["last_type2"])
        emb_last_call3 = self.last_call3_embed(x["last_call3"])
        emb_last_type3 = self.last_type3_embed(x["last_type3"])

        seq_calls = torch.stack([emb_last_call3, emb_last_call2, emb_last_call], dim=1)
        seq_types = torch.stack([emb_last_type3, emb_last_type2, emb_last_type], dim=1)
        seq_input = torch.cat([seq_calls, seq_types], dim=2)
        _, (hn, _) = self.seq_lstm(seq_input)
        emb_sequence_summary = hn[-1]

        x = torch.cat([
            x["numeric"],
            emb_pitcher,
            emb_batter,
            emb_throws,
            emb_outs,
            emb_on_1b,
            emb_on_2b,
            emb_on_3b,
            emb_half,
            emb_stand,
            emb_balls,
            emb_strikes,
            emb_sequence_summary
        ], dim=1)
        x = self.reduce_dims(x)
        x = self.linear_relu_stack(x)
        return self.output_dims(x)