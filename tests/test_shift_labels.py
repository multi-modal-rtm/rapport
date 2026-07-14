import pandas as pd
import pytest

from rapport.data.shift_labels import add_shift_labels, compute_shift_pos_weight


def _toy_dialogue() -> pd.DataFrame:
    """3 speakers, one dialogue: covers a speaker returning after others
    spoke (both same-label and different-label cases) and every speaker's
    first utterance.
    """
    return pd.DataFrame(
        {
            "dialogue_id": [0] * 7,
            "utterance_id": list(range(7)),
            "speaker": ["A", "B", "A", "C", "A", "B", "C"],
            "label": [0, 1, 0, 3, 3, 1, 0],  # neutral, joy, neutral, anger, anger, joy, neutral
        }
    )


def test_first_utterance_per_speaker_is_masked():
    df = add_shift_labels(_toy_dialogue())
    first_utterances = df[df["utterance_id"].isin([0, 1, 3])]  # A's, B's, C's first
    assert (first_utterances["shift_mask"] == 0).all()


def test_consecutive_same_label_is_zero():
    df = add_shift_labels(_toy_dialogue())
    # utterance_id=2: speaker A, same label (0) as A's previous (utterance_id=0, label=0)
    row = df[df["utterance_id"] == 2].iloc[0]
    assert row["shift_mask"] == 1
    assert row["shift_label"] == 0

    # utterance_id=5: speaker B, same label (1) as B's previous (utterance_id=1, label=1)
    row = df[df["utterance_id"] == 5].iloc[0]
    assert row["shift_mask"] == 1
    assert row["shift_label"] == 0


def test_speaker_returning_after_others_spoke_with_different_label():
    df = add_shift_labels(_toy_dialogue())
    # utterance_id=4: speaker A, previous A label=0 (utterance_id=2), current=3 -> shift
    row = df[df["utterance_id"] == 4].iloc[0]
    assert row["shift_mask"] == 1
    assert row["shift_label"] == 1

    # utterance_id=6: speaker C, previous C label=3 (utterance_id=3), current=0 -> shift
    row = df[df["utterance_id"] == 6].iloc[0]
    assert row["shift_mask"] == 1
    assert row["shift_label"] == 1


def test_multiple_dialogues_and_splits_dont_leak_speaker_history():
    df = pd.concat([_toy_dialogue(), _toy_dialogue().assign(dialogue_id=1)], ignore_index=True)
    result = add_shift_labels(df)
    # dialogue 1's first utterances must be masked exactly like dialogue 0's,
    # not carrying over "A's last label" from dialogue 0.
    dia1_firsts = result[(result["dialogue_id"] == 1) & (result["utterance_id"].isin([0, 1, 3]))]
    assert (dia1_firsts["shift_mask"] == 0).all()


def test_compute_shift_pos_weight():
    df = add_shift_labels(_toy_dialogue())
    # eligible rows (mask=1): utterance_id 2(label 0), 4(label 1), 5(label 0), 6(label 1) -> 2 pos, 2 neg
    pos_weight = compute_shift_pos_weight(df["shift_label"], df["shift_mask"])
    assert pos_weight == pytest.approx(1.0)


def test_compute_shift_pos_weight_upweights_rarer_positive():
    labels = pd.Series([1, 0, 0, 0, 0])
    mask = pd.Series([1, 1, 1, 1, 1])
    pos_weight = compute_shift_pos_weight(labels, mask)
    assert pos_weight == pytest.approx(4.0)  # 4 negatives / 1 positive


def test_compute_shift_pos_weight_ignores_masked_rows():
    # unmasked rows would give pos_weight 1.0 if counted, but they're excluded
    labels = pd.Series([1, 0, 1, 1])
    mask = pd.Series([1, 1, 0, 0])
    pos_weight = compute_shift_pos_weight(labels, mask)
    assert pos_weight == pytest.approx(1.0)  # only the first two rows count: 1 pos, 1 neg
