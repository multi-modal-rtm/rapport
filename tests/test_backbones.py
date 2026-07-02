import numpy as np
import pytest
import torch
from transformers import RobertaTokenizerFast, Wav2Vec2FeatureExtractor

from rapport.models.backbones import RobertaBackbone, Wav2Vec2Backbone


def test_roberta_text_features_are_process_independent():
    """Regression guard for the bug in docs/DIAGNOSIS.md: RobertaModel's
    pooler_output is randomly re-initialized per instantiation, so two
    fresh backbones given identical input used to produce different
    features. RobertaBackbone must use the deterministic pretrained <s>
    token instead, so this must hold across independently-constructed
    instances (the closest a same-process unit test can get to simulating
    separate processes).
    """
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    encoded = tokenizer(["hello world", "a slightly longer sentence for padding"], padding=True, return_tensors="pt")

    backbone_1 = RobertaBackbone()
    backbone_2 = RobertaBackbone()

    with torch.no_grad():
        pooled_1, _ = backbone_1(encoded["input_ids"], encoded["attention_mask"])
        pooled_2, _ = backbone_2(encoded["input_ids"], encoded["attention_mask"])

    assert torch.equal(pooled_1, pooled_2), (
        "text features differ between two fresh RobertaBackbone instances given identical "
        "input — the pooler-randomness bug has regressed"
    )


@pytest.mark.xfail(
    reason=(
        "wav2vec2's convolutional feature encoder sees padding zeros before the attention "
        "mask can exclude them, so a clip's features depend on its batch neighbors. Documents "
        "the hazard for the future end-to-end LoRA path, where audio may need to be batched "
        "for training throughput and this must be solved with proper masking or per-sample "
        "forward. The cache builder works around this today by encoding audio unbatched."
    ),
    strict=True,
)
def test_wav2vec2_padding_invariance():
    rng = np.random.default_rng(0)
    short_clip = rng.standard_normal(3413).astype(np.float32) * 0.05  # ~0.2s at 16kHz, matches the diagnosed outlier
    long_clip = rng.standard_normal(160000).astype(np.float32) * 0.05  # ~10s, forces heavy padding

    backbone = Wav2Vec2Backbone()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    alone = feature_extractor(short_clip, sampling_rate=16000, return_tensors="pt")
    batched = feature_extractor(
        [short_clip, long_clip], sampling_rate=16000, padding=True, return_tensors="pt", return_attention_mask=True
    )

    with torch.no_grad():
        pooled_alone, _ = backbone(alone["input_values"])
        pooled_batched, _ = backbone(batched["input_values"], batched["attention_mask"])

    assert torch.allclose(pooled_alone[0], pooled_batched[0], atol=1e-4), (
        "short clip's pooled feature should be identical whether encoded alone or padded "
        "in a batch with a much longer clip"
    )
