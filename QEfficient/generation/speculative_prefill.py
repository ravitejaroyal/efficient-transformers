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
        # align with _set_tokenizer_params()
        if getattr(self._tokenizer, "padding_side", "right") != "right":
            self._tokenizer.padding_side = "right"
        if getattr(self._tokenizer, "pad_token_id", None) is None:
            self._tokenizer.pad_token_id = getattr(self._tokenizer, "eos_token_id", 0)
        self._ctx_len = ctx_len
        self._prefill_seq_len = prompt_len
        self.L: Optional[int] = None
        self.H: Optional[int] = None
        self.H_kv: Optional[int] = None
        self.D: Optional[int] = None

        # prefer QPC-specialized values if present
        try:
            qpc_prefill = self._probe_prefill_seq_len()
            if qpc_prefill != self._prefill_seq_len:
                print(f"[spec] overriding prompt_len: QPC={qpc_prefill} vs user={self._prefill_seq_len}")
                self._prefill_seq_len = qpc_prefill
        except Exception:
            pass

    def _probe_vocab_size(self) -> int:
        # look up binding for "logits" and read last dim
        bidx = self._spec.binding_index_map["logits"]
        dims = list(self._spec.bindings[bidx].dims)
        return int(dims[-1])

    def _probe_prefill_seq_len(self) -> int:
        # read input_ids dims and take the time dimension used for prefill
        bidx = self._spec.binding_index_map["input_ids"]
        dims = list(self._spec.bindings[bidx].dims)
        return int(dims[-1])

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
        # preallocate logits like runtime does (B=1, step=1)
        try:
            vocab = self._probe_vocab_size()
            logits_placeholder = np.zeros((1, 1, vocab), dtype=np.float32)
            self._spec.set_buffers({"logits": logits_placeholder})
        except Exception:
            pass

        enc = self._tokenizer(prompt, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        S = ids.shape[1]

        pad_id = self._tokenizer.pad_token_id or getattr(self._tokenizer, "eos_token_id", 0)
        padded_len = ((S + self._prefill_seq_len - 1) // self._prefill_seq_len) * self._prefill_seq_len
        if padded_len > S:
            ids_pad = np.pad(ids, ((0, 0), (0, padded_len - S)), constant_values=pad_id)
        else:
            ids_pad = ids

        attn = (ids_pad != pad_id).astype(np.int64)
        pos = np.where(attn == 1, np.arange(padded_len, dtype=np.int64), -1).reshape(1, padded_len)

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

if __name__ == "__main__":
    import argparse
    import sys
    from QEfficient.utils import load_hf_tokenizer

    parser = argparse.ArgumentParser(
        description="Step 4.1 sanity check: run speculator prefill and print shapes."
    )
    parser.add_argument("--spec-qpc", required=True, help="Path to speculator QPC directory (…/qpc)")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B", help="HF tokenizer card")
    parser.add_argument("--prompt", default="Hello from speculative prefill on AI100.",
                        help="Prompt text for a single-run sanity check")
    parser.add_argument("--prompt-len", type=int, default=16, help="prefill chunk length")
    parser.add_argument("--ctx-len", type=int, default=128, help="context length")
    parser.add_argument("--device-ids", default="[0]", help="Device IDs like [0] or [0,1]")
    parser.add_argument("--also-long", action="store_true",
                        help="Optionally run a longer prompt to force multiple chunks")
    args = parser.parse_args()

    # Parse device list safely
    try:
        dev_ids = [int(x) for x in args.device_ids.strip("[] ").split(",") if x.strip() != ""]
    except Exception:
        print(f"[warn] Could not parse --device-ids={args.device_ids!r}, defaulting to [0]")
        dev_ids = [0]

    # Load tokenizer and engine
    tok = load_hf_tokenizer(args.model_name)
    eng = SpeculativePrefillEngine(
        spec_qpc_path=args.spec_qpc,
        tokenizer=tok,
        ctx_len=int(args.ctx_len),
        prompt_len=int(args.prompt_len),
        device_ids=dev_ids,
    )

    def run_once(prompt_text: str) -> int:
        print("\n=== PROMPT ===", prompt_text[:120])
        out = eng.spec_prefill(prompt_text)
        q = out.get("prefill_queries", None)
        keys = out.get("past_keys", [])
        logits = out.get("logits", None)
        S = int(out.get("S", 0))

        # Basic presence checks
        if q is None or logits is None or not isinstance(keys, list) or len(keys) == 0:
            print("[fail] Missing one of: prefill_queries / logits / past_keys")
            return 1

        # Shapes & dims
        print(f"[spec] prefill_queries shape: {q.shape}")
        print(f"[spec] first past_key shape: {keys[0].shape}   layers: {len(keys)}")
        print(f"[spec] logits shape: {logits.shape}")
        print(f"[spec] S (tokenized): {S}")

        # Expected dims for Meta-Llama-3-8B spec (L=32, H=32, H_kv=8, D=128)
        L, H, D = q.shape
        H_kv = keys[0].shape[1]
        S_key = keys[0].shape[2]

        ok = True
        if L != 32:
            print(f"[warn] L mismatch: got {L}, expected 32"); ok = False
        if H != 32:
            print(f"[warn] H mismatch: got {H}, expected 32"); ok = False
        if H_kv != 8:
            print(f"[warn] H_kv mismatch: got {H_kv}, expected 8"); ok = False
        if D != 128:
            print(f"[warn] D mismatch: got {D}, expected 128"); ok = False

        # Prefill chunking sanity
        cs = eng._prefill_seq_len
        padded_len = ((S + cs - 1) // cs) * cs
        chunks = padded_len // cs
        print(f"[spec] prefill_seq_len={cs}  padded_len={padded_len}  chunks={chunks}")
        if not (S <= S_key <= padded_len):
            print(f"[warn] Key seq length unexpected: S={S}  key_S={S_key}  padded_len={padded_len}")
            ok = False

        # Dtypes
        if logits.dtype != np.float32:
            print(f"[warn] logits dtype is {logits.dtype}, expected float32"); ok = False

        print("[OK]" if ok else "[WARN] see messages above")
        return 0 if ok else 2

    rc = run_once(args.prompt)

    if args.also_long:
        long_prompt = " ".join(["This is a longer prompt to force multiple prefill chunks."] * 24)
        rc2 = run_once(long_prompt)
        rc = rc or rc2

    sys.exit(rc)
