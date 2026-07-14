import pandas as pd
import torch
from transformers import AutoTokenizer

from rapport.data.context_text import (
    ContextTextCollator,
    MELDContextTextDataset,
    context_window,
    encode_with_context,
)

# NOTE: use AutoTokenizer, not RobertaTokenizerFast directly -- on this
# environment's transformers version, RobertaTokenizerFast's constructor
# leaves backend_tokenizer.pre_tokenizer/decoder as None (silently falling
# back to character-level tokenization with no byte-level merges), while
# AutoTokenizer resolves to a correctly-configured fast tokenizer. Verified
# directly: RobertaTokenizerFast('Hello world') -> 10 char-level tokens;
# AutoTokenizer('Hello world') -> ['Hello', 'Ġworld'] as expected.
TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")


def _dialogue(n: int) -> tuple[list[str], list[str]]:
    speakers = [f"Speaker{i % 3}" for i in range(n)]
    utterances = [f"utterance number {i}" for i in range(n)]
    return speakers, utterances


def test_context_window_no_future_leakage():
    speakers, utterances = _dialogue(12)
    for t in range(len(utterances)):
        window = context_window(speakers, utterances, t, k=8)
        # the last segment is always utterance t itself
        assert window[-1] == (speakers[t], utterances[t])
        # every segment comes from index <= t -- never future utterances
        included_texts = {u for _, u in window}
        assert included_texts <= set(utterances[: t + 1])
        assert utterances[t] in included_texts
        for future in utterances[t + 1 :]:
            assert future not in included_texts


def test_context_window_uses_at_most_k_preceding_utterances():
    speakers, utterances = _dialogue(20)
    window = context_window(speakers, utterances, t=15, k=8)
    assert len(window) == 9  # 8 preceding + current
    assert window[0] == (speakers[7], utterances[7])
    assert window[-1] == (speakers[15], utterances[15])


def test_first_utterance_of_dialogue_has_no_context():
    speakers, utterances = _dialogue(5)
    window = context_window(speakers, utterances, t=0, k=8)
    assert window == [(speakers[0], utterances[0])]


def test_encode_with_context_no_future_leakage_in_decoded_text():
    speakers, utterances = _dialogue(6)
    encoded = encode_with_context(TOKENIZER, speakers, utterances, t=2, k=8, max_length=256)
    decoded = TOKENIZER.decode(encoded["input_ids"], skip_special_tokens=True)
    for future in utterances[3:]:
        assert future not in decoded


def test_truncation_preserves_current_utterance_intact():
    speakers = [f"Speaker{i % 3}" for i in range(9)]
    # Long context segments force truncation; the current utterance is short
    # and distinctive, and must survive intact even once earlier segments
    # are dropped to fit max_length.
    utterances = [f"a long padding utterance repeated many times {i} " * 8 for i in range(8)]
    current = "the current utterance must survive"
    utterances.append(current)

    encoded = encode_with_context(TOKENIZER, speakers, utterances, t=8, k=8, max_length=48)
    decoded = TOKENIZER.decode(encoded["input_ids"], skip_special_tokens=True)

    assert current in decoded
    assert encoded["input_ids"].shape[0] <= 48
    # confirms context was actually dropped, not just coincidentally short
    assert utterances[0] not in decoded


def test_encode_with_context_is_deterministic():
    speakers, utterances = _dialogue(10)
    first = encode_with_context(TOKENIZER, speakers, utterances, t=6, k=8, max_length=256)
    second = encode_with_context(TOKENIZER, speakers, utterances, t=6, k=8, max_length=256)
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["attention_mask"], second["attention_mask"])


def test_attention_mask_is_all_ones_no_padding_yet():
    speakers, utterances = _dialogue(3)
    encoded = encode_with_context(TOKENIZER, speakers, utterances, t=2, k=8, max_length=256)
    assert encoded["attention_mask"].sum().item() == encoded["input_ids"].shape[0]


def test_dataset_flattens_dialogues_to_utterance_level(tmp_path):
    index_df = pd.DataFrame(
        {
            "dialogue_id": [0, 0, 0, 1, 1],
            "utterance_id": [0, 1, 2, 0, 1],
            "speaker": ["A", "B", "A", "C", "A"],
            "label": [0, 1, 3, 2, 0],
            "text": ["hi", "hello there", "grr", "sigh", "ok then"],
        }
    )
    index_path = tmp_path / "train.parquet"
    index_df.to_parquet(index_path, index=False)

    dataset = MELDContextTextDataset(index_path, TOKENIZER, k=8, max_length=256)
    assert len(dataset) == 5  # utterance-level, not dialogue-level

    item = dataset[2]  # dialogue 0, t=2 ("grr")
    decoded = TOKENIZER.decode(item["input_ids"], skip_special_tokens=True)
    assert "grr" in decoded
    assert "sigh" not in decoded  # belongs to a different dialogue
    assert item["label"].item() == 3


def test_collator_pads_to_batch_max_length(tmp_path):
    index_df = pd.DataFrame(
        {
            "dialogue_id": [0, 0],
            "utterance_id": [0, 1],
            "speaker": ["A", "B"],
            "label": [0, 1],
            "text": ["short", "a somewhat longer utterance than the first one"],
        }
    )
    index_path = tmp_path / "train.parquet"
    index_df.to_parquet(index_path, index=False)

    dataset = MELDContextTextDataset(index_path, TOKENIZER, k=8, max_length=256)
    batch = [dataset[0], dataset[1]]
    collate = ContextTextCollator(pad_token_id=TOKENIZER.pad_token_id)
    out = collate(batch)

    max_len = max(item["input_ids"].shape[0] for item in batch)
    assert out["input_ids"].shape == (2, max_len)
    assert out["attention_mask"].shape == (2, max_len)
    assert out["labels"].tolist() == [0, 1]

    for i, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        assert torch.equal(out["input_ids"][i, :length], item["input_ids"])
        if length < max_len:
            assert out["attention_mask"][i, length:].sum().item() == 0
            assert (out["input_ids"][i, length:] == TOKENIZER.pad_token_id).all()
