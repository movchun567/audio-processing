import os

import json
import glob
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings("ignore")


#  Config

class CFG:
    BASE_DIR   = "/kaggle/input/competitions/birdclef-2026"
    TEST_DIR   = f"{BASE_DIR}/test_soundscapes"
    SAMPLE_SUB = f"{BASE_DIR}/sample_submission.csv"

    DATASET_DIR        = "/kaggle/input/datasets/studentedvard/zvuk2-ast-model"
    TRAINED_MODEL_PATH = f"{DATASET_DIR}/bird_ast_model.pth"
    META_PATH          = f"{DATASET_DIR}/target_columns.json"

    TARGET_SR        = 32_000
    SEGMENT_SEC      = 5.0
    N_MELS           = 128
    N_FFT            = 1024
    HOP_LENGTH       = 320
    FMIN             = 20.0
    FMAX             = 16_000.0
    SPEC_TIME_FRAMES = 512

    PATCH_FREQ = 16
    PATCH_TIME = 16
    D_MODEL    = 384
    N_HEADS    = 6
    N_LAYERS   = 12
    MLP_RATIO  = 4.0
    DROPOUT    = 0.0

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()


#  Model architecture

class PatchEmbed(nn.Module):
    def __init__(self, freq_bins, time_frames, patch_freq, patch_time, d_model):
        super().__init__()
        self.n_freq  = freq_bins  // patch_freq
        self.n_time  = time_frames // patch_time
        self.n_patch = self.n_freq * self.n_time
        self.proj    = nn.Conv2d(
            1, d_model,
            kernel_size=(patch_freq, patch_time),
            stride     =(patch_freq, patch_time),
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        return x.transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim    = int(d_model * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class AudioSpectrogramTransformer(nn.Module):
    def __init__(self, num_classes, freq_bins, time_frames,
                 patch_freq, patch_time, d_model, n_heads, n_layers,
                 mlp_ratio, dropout):
        super().__init__()
        self.patch_embed = PatchEmbed(freq_bins, time_frames, patch_freq, patch_time, d_model)
        n_patch          = self.patch_embed.n_patch
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patch + 1, d_model))
        self.pos_drop    = nn.Dropout(dropout)
        self.blocks      = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        B  = x.size(0)
        x  = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_drop(x + self.pos_embed)
        x   = self.blocks(x)
        x   = self.norm(x)
        return torch.sigmoid(self.head(x[:, 0]))


#  Load metadata & checkpoint

if not Path(cfg.META_PATH).exists():
    raise FileNotFoundError(f"Metadata not found: {cfg.META_PATH}")

with open(cfg.META_PATH) as f:
    INF_TARGET_COLUMNS = json.load(f)

if not Path(cfg.TRAINED_MODEL_PATH).exists():
    raise FileNotFoundError(f"Checkpoint not found: {cfg.TRAINED_MODEL_PATH}")

checkpoint = torch.load(cfg.TRAINED_MODEL_PATH, map_location=cfg.DEVICE)
NUM_CLASSES = checkpoint["num_classes"]

snap = checkpoint.get("cfg_snapshot", {})
model = AudioSpectrogramTransformer(
    num_classes = NUM_CLASSES,
    freq_bins   = snap.get("N_MELS",           cfg.N_MELS),
    time_frames = snap.get("SPEC_TIME_FRAMES", cfg.SPEC_TIME_FRAMES),
    patch_freq  = snap.get("PATCH_FREQ",       cfg.PATCH_FREQ),
    patch_time  = snap.get("PATCH_TIME",       cfg.PATCH_TIME),
    d_model     = snap.get("D_MODEL",          cfg.D_MODEL),
    n_heads     = snap.get("N_HEADS",          cfg.N_HEADS),
    n_layers    = snap.get("N_LAYERS",         cfg.N_LAYERS),
    mlp_ratio   = snap.get("MLP_RATIO",        cfg.MLP_RATIO),
    dropout     = 0.0,
).to(cfg.DEVICE)

model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"Model loaded ({cfg.DEVICE}).")



#  Audio transforms & inference helpers

mel_transform = T.MelSpectrogram(
    sample_rate = cfg.TARGET_SR,
    n_fft       = cfg.N_FFT,
    hop_length  = cfg.HOP_LENGTH,
    n_mels      = cfg.N_MELS,
    f_min       = cfg.FMIN,
    f_max       = cfg.FMAX,
)
db_transform = T.AmplitudeToDB(stype="power", top_db=80)


def audio_to_tensor(wav_segment: torch.Tensor) -> torch.Tensor:
    spec    = db_transform(mel_transform(wav_segment))
    T_fixed = cfg.SPEC_TIME_FRAMES
    if spec.shape[2] >= T_fixed:
        spec = spec[:, :, :T_fixed]
    else:
        spec = F.pad(spec, (0, T_fixed - spec.shape[2]))
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec.unsqueeze(0)


def predict_file(path: str) -> list:
    fname = Path(path).stem
    try:
        wav, sr = torchaudio.load(path)
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return []

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != cfg.TARGET_SR:
        wav = T.Resample(orig_freq=sr, new_freq=cfg.TARGET_SR)(wav)

    seg_len   = int(cfg.SEGMENT_SEC * cfg.TARGET_SR)
    n_windows = math.ceil(wav.shape[1] / seg_len)
    results   = []

    with torch.no_grad():
        for i in range(n_windows):
            start   = i * seg_len
            segment = wav[:, start : start + seg_len]

            if segment.shape[1] < seg_len:
                segment = F.pad(segment, (0, seg_len - segment.shape[1]))

            spec = audio_to_tensor(segment).to(cfg.DEVICE)

            with torch.amp.autocast(device_type=cfg.AMP_DEVICE):
                probs = model(spec)

            probs  = probs[0].float().cpu().numpy()
            row_id = f"{fname}_{(i + 1) * 5}"
            results.append((row_id, probs))

    return results



#  Inference and submission

sample_sub   = pd.read_csv(cfg.SAMPLE_SUB)
SUB_COLS     = list(sample_sub.columns)
SPECIES_COLS = SUB_COLS[1:]

test_files = sorted(glob.glob(f"{cfg.TEST_DIR}/*.ogg"))

print(f"Test files found : {len(test_files)}")
print(f"Submission cols  : {len(SPECIES_COLS)} bird species")
print(f"Model outputs    : {NUM_CLASSES} classes")

model_cols_set = set(INF_TARGET_COLUMNS)
sub_cols_set   = set(SPECIES_COLS)
only_in_model  = model_cols_set - sub_cols_set
only_in_sub    = sub_cols_set - model_cols_set
print(f"In model but NOT in submission : {len(only_in_model)} → {list(only_in_model)[:5]}")
print(f"In submission but NOT in model : {len(only_in_sub)} → {list(only_in_sub)[:5]}")

if test_files:
    predictions = {}
    for fp in tqdm(test_files, desc="Inference"):
        try:
            file_results = predict_file(fp)
        except Exception as e:
            print(f"  [ERROR] {fp}: {e}")
            file_results = []
        for row_id, probs in file_results:
            predictions[row_id] = probs

    print(f"Predictions collected: {len(predictions)}")
    if predictions:
        sample_pred_key = list(predictions.keys())[0]
        print(f"  Sample prediction key : {sample_pred_key}")
        print(f"  Sample submission key : {sample_sub['row_id'].iloc[0]}")

    records      = []
    missing_rows = 0

    for row_id in sample_sub["row_id"]:
        record = {"row_id": row_id}

        if row_id in predictions:
            probs = predictions[row_id]
            for col in SPECIES_COLS:
                if col in model_cols_set:
                    idx = INF_TARGET_COLUMNS.index(col)
                    record[col] = float(probs[idx])
                else:
                    record[col] = 0.0
        else:
            missing_rows += 1
            for col in SPECIES_COLS:
                record[col] = 0.0

        records.append(record)

    if missing_rows > 0:
        print(f" {missing_rows} rows from sample_submission not covered by predictions → filled with 0.0")

    submission_df = pd.DataFrame(records)[SUB_COLS]

    assert list(submission_df.columns) == SUB_COLS, "Column mismatch with sample_submission!"
    assert len(submission_df) == len(sample_sub), \
        f"Row count mismatch: {len(submission_df)} vs {len(sample_sub)}"

    submission_df.to_csv("submission.csv", index=False)
    print(f"submission.csv saved. Shape: {submission_df.shape}")

else:
    print(" No test files found - saving dummy submission (sample_submission as-is).")
    sample_sub.to_csv("submission.csv", index=False)
    print("Dummy submission.csv saved.")