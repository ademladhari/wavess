from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import torch
import torch.nn as nn


class AestheticScorer(nn.Module):
    def __init__(
        self,
        input_size=0,
        use_activation=False,
        dropout=0.2,
        config=None,
        hidden_dim=1024,
        reduce_dims=False,
        output_activation=None,
    ):
        super().__init__()
        self.config = {
            "input_size": input_size,
            "use_activation": use_activation,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "reduce_dims": reduce_dims,
            "output_activation": output_activation,
        }
        if config is not None:
            self.config.update(config)

        layers = [
            nn.Linear(self.config["input_size"], self.config["hidden_dim"]),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Dropout(self.config["dropout"]),
            nn.Linear(
                self.config["hidden_dim"],
                round(self.config["hidden_dim"] / (2 if reduce_dims else 1)),
            ),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Dropout(self.config["dropout"]),
            nn.Linear(
                round(self.config["hidden_dim"] / (2 if reduce_dims else 1)),
                round(self.config["hidden_dim"] / (4 if reduce_dims else 1)),
            ),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Dropout(self.config["dropout"]),
            nn.Linear(
                round(self.config["hidden_dim"] / (4 if reduce_dims else 1)),
                round(self.config["hidden_dim"] / (8 if reduce_dims else 1)),
            ),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Linear(round(self.config["hidden_dim"] / (8 if reduce_dims else 1)), 1),
        ]
        if self.config["output_activation"] == "sigmoid":
            layers.append(nn.Sigmoid())
        layers = [x for x in layers if x is not None]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.config["output_activation"] == "sigmoid":
            upper, lower = 10, 1
            scale = upper - lower
            return (self.layers(x) * scale) + lower
        return self.layers(x)

    def save(self, save_name):
        split_name = os.path.splitext(save_name)
        with open(f"{split_name[0]}.config", "w") as outfile:
            outfile.write(json.dumps(self.config, indent=4))

        for i in range(6):
            try:
                torch.save(self.state_dict(), save_name)
                break
            except RuntimeError as e:
                if "cannot be opened" in str(e) and i < 5:
                    print("Model save failed, retrying...")
                else:
                    raise e


def preprocess(embeddings):
    return embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)


def _candidate_weights_dirs() -> list[Path]:
    env_dir = os.environ.get("WMBENCH_AESTHETICS_WEIGHTS_DIR")
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir))

    # Preferred: colocated with this module.
    candidates.append(Path(inspect.getfile(_candidate_weights_dirs)).resolve().parent / "weights")

    # Fallback: weights vendored in upstream repo snapshot (file dependency only; no imports).
    candidates.append(Path(__file__).resolve().parents[3] / "waves" / "metrics" / "metrics" / "aesthetics_scorer" / "weights")

    # Also try common MODEL_DIR layout.
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        candidates.append(Path(model_dir) / "aesthetics_scorer")
        candidates.append(Path(model_dir))

    # De-dup preserving order.
    seen = set()
    out = []
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            pass
        if str(c) not in seen:
            seen.add(str(c))
            out.append(c)
    return out


def load_model(weight_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    for weight_folder in _candidate_weights_dirs():
        weight_path = weight_folder / f"{weight_name}.pth"
        config_path = weight_folder / f"{weight_name}.config"
        if weight_path.exists() and config_path.exists():
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
            model = AestheticScorer(config=config)
            model.load_state_dict(torch.load(str(weight_path), map_location=device))
            model.eval()
            return model

    searched = "\n".join([f"- {p}" for p in _candidate_weights_dirs()])
    raise FileNotFoundError(
        f"Could not find aesthetics_scorer weights '{weight_name}' in any of:\n{searched}"
    )
