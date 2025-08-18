from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


@dataclass
class QEffBaseModelOutputWithPast(BaseModelOutputWithPast):
    """Base model outputs with prefill queries."""

    # Stacked last-token queries per layer.
    prefill_queries: Optional[torch.FloatTensor] = None


@dataclass
class QEffCausalLMOutputWithPast(CausalLMOutputWithPast):
    """CausalLM outputs with prefill queries."""

    prefill_queries: Optional[torch.FloatTensor] = None
