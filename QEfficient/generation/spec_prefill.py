from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple, List

from QEfficient.generation.cloud_infer import QAICInferenceSession


class SpecPrefillHelper:
    """Speculator prefill helper mirroring runtime prefill semantics."""

    def __init__(
        self,
        qpc_path: str,
        tokenizer,
        ctx_len: int,
        prefill_seq_len: int,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        self._session = QAICInferenceSession(qpc_path, device_ids=device_ids)
        self._tokenizer = tokenizer
        if getattr(self._tokenizer, "padding_side", "right") != "right":
            self._tokenizer.padding_side = "right"
        if getattr(self._tokenizer, "pad_token_id", None) is None:
            self._tokenizer.pad_token_id = getattr(self._tokenizer, "eos_token_id", 0)
        self._ctx_len = int(ctx_len)
        self._prefill_seq_len = int(prefill_seq_len)
        self._vocab_size: Optional[int] = None

    # --- helpers ---
    def _fetch_vocab_size(self) -> int:
        if self._vocab_size is not None:
            return self._vocab_size
        bidx = self._session.binding_index_map["logits"]
        dims = list(self._session.bindings[bidx].dims)
        self._vocab_size = int(dims[-1])
        return self._vocab_size

    def _fetch_generation_len(self, generation_len: Optional[int], max_gen_len: int) -> int:
        if generation_len is None:
            return int(max_gen_len)
        return int(min(int(generation_len), int(max_gen_len)))

    # --- PREFILL: mirror runtime/base run_prefill ---
    def run_prefill(self, prompt, generation_len, prefill_logit_bs=1, decode_batch_id=None):
        """Run prefill for a given prompt."""
        inputs = self._tokenizer(prompt, return_tensors="np", padding=True)
        position_ids = inputs["attention_mask"].sum(1, keepdims=True)
        padded_len = inputs["input_ids"].shape[1]
        num_chunks = -(padded_len // -self._prefill_seq_len)
        padded_len = num_chunks * self._prefill_seq_len

        max_gen_len = self._ctx_len - position_ids.max()
        generation_len = self._fetch_generation_len(generation_len, max_gen_len)

        self._fetch_vocab_size()
        logits_out_placeholder = np.zeros((prefill_logit_bs, 1, self._vocab_size), dtype=np.float32)
        self._session.set_buffers({"logits": logits_out_placeholder})

        inputs = self._tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
        inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
        inputs.pop("token_type_ids", None)

        if decode_batch_id is not None:
            inputs["batch_index"] = decode_batch_id

        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]
            chunk_inputs["position_ids"] = inputs["position_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]
            outputs = self._session.run(chunk_inputs)
        return outputs, position_ids, generation_len

    @staticmethod
    def collect_for_scoring(outputs: Dict[str, Any]):
        prefill_queries = outputs["prefill_queries"]
        num_layers = prefill_queries.shape[0]
        past_keys = [outputs[f"past_key.{i}_RetainedState"] for i in range(num_layers)]
        return prefill_queries, past_keys
