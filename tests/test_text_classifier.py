import torch
from transformers import AutoTokenizer

from rapport.models.text_classifier import ContextTextClassifier

TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")


def test_forward_shape():
    model = ContextTextClassifier(num_classes=7)
    encoded = TOKENIZER(["hello there", "a short one"], padding=True, return_tensors="pt")
    logits = model(encoded["input_ids"], encoded["attention_mask"])
    assert logits.shape == (2, 7)


def test_only_lora_and_head_are_trainable():
    model = ContextTextClassifier(num_classes=7)

    for name, param in model.encoder.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, f"LoRA param {name} should be trainable"
        else:
            assert not param.requires_grad, f"non-LoRA base param {name} should be frozen"

    for param in model.classifier.parameters():
        assert param.requires_grad


def test_trainable_param_budget_is_small():
    model = ContextTextClassifier(num_classes=7)
    fraction = model.trainable_params / model.total_params
    assert 0 < fraction < 0.05
    # sanity: LoRA (r=8, alpha=16, query+value, 12 layers) + a 768x7 head is a
    # few hundred thousand params, not hundreds of millions.
    assert model.trainable_params < 1_000_000


def test_gradients_flow_only_to_trainable_params():
    model = ContextTextClassifier(num_classes=7)
    encoded = TOKENIZER(["hello there", "a short one"], padding=True, return_tensors="pt")
    logits = model(encoded["input_ids"], encoded["attention_mask"])
    loss = logits.sum()
    loss.backward()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
        else:
            assert param.grad is None
