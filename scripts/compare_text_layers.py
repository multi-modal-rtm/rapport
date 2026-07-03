"""One-off comparison: final-layer vs. second-to-last-layer masked mean-pooling
for frozen RoBERTa text features (diagnostics only, not part of the persistent cache).

Computes both poolings for the full train+test split in a single forward
pass per batch (output_hidden_states=True gives every layer at once), runs
the same unbalanced-weighted-F1 linear probe on each, and reports which is
better.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import RobertaModel, RobertaTokenizerFast

TEXT_TOKEN_MAX_LEN = 64


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


@torch.no_grad()
def extract_split(
    processed_dir: Path, split: str, model: RobertaModel, tokenizer: RobertaTokenizerFast,
    device: torch.device, batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(processed_dir / f"{split}.parquet")
    final_feats, second_last_feats, labels = [], [], []

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        inputs = tokenizer(
            list(batch["text"]), padding=True, truncation=True, max_length=TEXT_TOKEN_MAX_LEN, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        final_pooled = masked_mean(outputs.hidden_states[-1], attention_mask)
        second_last_pooled = masked_mean(outputs.hidden_states[-2], attention_mask)

        final_feats.append(final_pooled.cpu().numpy())
        second_last_feats.append(second_last_pooled.cpu().numpy())
        labels.append(batch["label"].to_numpy())

    return np.concatenate(final_feats), np.concatenate(second_last_feats), np.concatenate(labels)


def probe_unbalanced(X_train, y_train, X_test, y_test, max_iter: int = 2000) -> float:
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)
    return f1_score(y_test, clf.predict(X_test), average="weighted", zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/meld/processed", type=Path)
    parser.add_argument("--model-name", default="roberta-base")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaModel.from_pretrained(args.model_name, add_pooling_layer=False).to(device)
    model.eval()
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)

    print("[compare_text_layers] extracting train...")
    final_train, second_last_train, y_train = extract_split(args.processed_dir, "train", model, tokenizer, device)
    print("[compare_text_layers] extracting test...")
    final_test, second_last_test, y_test = extract_split(args.processed_dir, "test", model, tokenizer, device)

    wf1_final = probe_unbalanced(final_train, y_train, final_test, y_test)
    wf1_second_last = probe_unbalanced(second_last_train, y_train, second_last_test, y_test)

    print(f"\n[compare_text_layers] final layer (hidden_states[-1]) masked-mean: unbalanced weighted F1 = {wf1_final:.4f}")
    print(f"[compare_text_layers] second-to-last layer (hidden_states[-2]) masked-mean: unbalanced weighted F1 = {wf1_second_last:.4f}")
    winner = "final layer" if wf1_final >= wf1_second_last else "second-to-last layer"
    print(f"[compare_text_layers] winner: {winner}")


if __name__ == "__main__":
    main()
