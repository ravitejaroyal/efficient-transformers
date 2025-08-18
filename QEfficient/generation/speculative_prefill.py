# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from QEfficient.generation.cloud_infer import QAICInferenceSession


class SpeculativePrefillEngine:
    """Run speculator prefill and collect tensors for later scoring."""

    def __init__(
        self,
        spec_qpc_path: str,
        tokenizer,
        ctx_len: int,
        prompt_len: int,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        self._spec = QAICInferenceSession(spec_qpc_path, device_ids=device_ids)
        self._tokenizer = tokenizer
        self._ctx_len = ctx_len
        self._prefill_seq_len = prompt_len
        self.L: Optional[int] = None
        self.H: Optional[int] = None
        self.H_kv: Optional[int] = None
        self.D: Optional[int] = None

    def spec_prefill(self, prompt: str) -> Dict[str, Any]:
        """
        Run speculator prefill on ``prompt`` and return required tensors.

        Returns:
            Dict[str, Any]:
                ``prefill_queries`` (np.ndarray): stacked prefill queries ``[L, H, D]``
                ``past_keys`` (List[np.ndarray]): per-layer key tensors ``[1, H_kv, S, D]``
                ``logits`` (np.ndarray): logits from the last chunk
                ``S`` (int): tokenized length before padding
        """

        enc = self._tokenizer(prompt, return_tensors="np")
        ids = enc["input_ids"].astype("int64")
        S = ids.shape[1]
        padded_len = ((S + self._prefill_seq_len - 1) // self._prefill_seq_len) * self._prefill_seq_len
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        ids_pad = np.pad(ids, ((0, 0), (0, padded_len - S)), constant_values=pad_id)
        attn = (ids_pad != pad_id).astype("int64")
        pos = np.where(attn == 1, np.arange(padded_len, dtype="int64"), -1).reshape(1, padded_len)

        last_outs: Dict[str, np.ndarray] = {}
        for i in range(padded_len // self._prefill_seq_len):
            start = i * self._prefill_seq_len
            end = (i + 1) * self._prefill_seq_len
            chunk_inputs = {
                "input_ids": ids_pad[:, start:end],
                "position_ids": pos[:, start:end],
            }
            last_outs = self._spec.run(chunk_inputs)

        prefill_queries = last_outs["prefill_queries"]
        past_keys: List[np.ndarray] = []
        idx = 0
        while f"past_key.{idx}_RetainedState" in last_outs:
            past_keys.append(last_outs[f"past_key.{idx}_RetainedState"])
            idx += 1
        L = len(past_keys)
        logits = last_outs["logits"]

        if self.L is None:
            self.L = L
            if L > 0:
                first_key = past_keys[0]
                self.H_kv = first_key.shape[1]
                self.D = first_key.shape[-1]
            self.H = prefill_queries.shape[1]

        print("[spec] prefill_queries:", prefill_queries.shape)
        if past_keys:
            print("[spec] past_keys[0]:", past_keys[0].shape, "layers:", len(past_keys))
        print("[spec] logits:", logits.shape)
        print("[spec] S:", S)

        return {
            "prefill_queries": prefill_queries,
            "past_keys": past_keys,
            "logits": logits,
            "S": int(S),
        }
