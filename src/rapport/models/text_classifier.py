"""Phase T contextual text encoder: roberta-base + LoRA, trained end-to-end.

Unlike `rapport.models.backbones.RobertaBackbone` (frozen, no gradient),
this wraps a LoRA-adapted RoBERTa whose adapter weights and classification
head ARE trained. Pooling is masked mean over `last_hidden_state` -- the
pooling method the frozen-backbone layer comparison (docs/DIAGNOSIS.md,
Recalibration Step 2) found beats raw `<s>`/`pooler_output` pooling; kept
here for the trainable regime rather than re-deriving it, since with an
end-to-end-tuned encoder the frozen-probe layer-choice result (final vs.
second-to-last layer) doesn't straightforwardly transfer.

NOTE: build tokenizers via `transformers.AutoTokenizer`, not
`RobertaTokenizerFast` directly -- on this environment's transformers
version, constructing `RobertaTokenizerFast` directly leaves
`backend_tokenizer.pre_tokenizer`/`decoder` as None, silently falling back
to character-level tokenization with no byte-level merges (verified
directly: `RobertaTokenizerFast('Hello world')` -> 10 char tokens instead of
2). `AutoTokenizer.from_pretrained("roberta-base")` resolves to a correctly
configured fast tokenizer and must be used everywhere in this phase.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

log = logging.getLogger(__name__)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query", "value"]

# Trainable params (LoRA adapters + head) must stay small relative to the
# frozen backbone -- this is an adapter-tuning regime, not full fine-tuning.
# 124M-param roberta-base + a 7-class head comes out to ~0.3%; 5% is a loose
# guard against a config change accidentally unfreezing the base model.
MAX_TRAINABLE_FRACTION = 0.05


class ContextTextClassifier(nn.Module):
    """roberta-base + LoRA (r=8, alpha=16, dropout=0.05, target=[query,value])
    + a linear classification head on the masked mean of `last_hidden_state`.

    Trainable parameters are exactly the LoRA adapter weights + the head;
    every other RoBERTa parameter is frozen by `get_peft_model`.
    """

    def __init__(self, num_classes: int, model_name: str = "roberta-base"):
        super().__init__()
        base = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        self.encoder = get_peft_model(base, lora_config)
        self.classifier = nn.Linear(base.config.hidden_size, num_classes)

        self.trainable_params, self.total_params = self._log_trainable_param_budget()

    def _log_trainable_param_budget(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        fraction = trainable / total
        log.info(
            "ContextTextClassifier trainable params: %d / %d (%.4f%%)", trainable, total, fraction * 100
        )
        assert fraction < MAX_TRAINABLE_FRACTION, (
            f"trainable param fraction {fraction:.4f} exceeds the {MAX_TRAINABLE_FRACTION} adapter-tuning "
            "budget -- something beyond the LoRA adapters + head is unexpectedly trainable"
        )
        # head-only sanity: classifier params must all be included in "trainable"
        head_params = sum(p.numel() for p in self.classifier.parameters())
        assert head_params <= trainable, "classification head params must be a subset of trainable params"
        return trainable, total

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids/attention_mask: [B, L] -> pooled embedding [B, H] (masked mean of
        last_hidden_state). This is the U_e utterance embedding consumed by Phase N4's
        fusion/relational-memory pipeline (docs/SPEC_RAPPORT_COMPONENTS.md), extracted
        separately from `forward` so callers can cache it without running the
        classification head.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        tokens = outputs.last_hidden_state  # [B, L, H]

        mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids/attention_mask: [B, L] -> logits [B, num_classes]."""
        pooled = self.encode(input_ids, attention_mask)
        return self.classifier(pooled)
