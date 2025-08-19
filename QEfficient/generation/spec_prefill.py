from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple, List

from QEfficient.generation.cloud_infer import QAICInferenceSession


class SpecPrefillEngine:
    """
    Mirror QEfficient/generation/text_generation_inference.py::run_prefill (variable names & logic)
    for a SPECULATOR QPC. Reuses QAICInferenceSession and exposes a helper to collect tensors
    needed for speculative scoring (prefill_queries and per-layer past_key.*_RetainedState).
    """

    def __init__(
        self,
        spec_qpc_path: str,
        tokenizer,
        ctx_len: int,
        prefill_seq_len: int,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        self._session = QAICInferenceSession(spec_qpc_path, device_ids=device_ids)
        self._tokenizer = tokenizer
        # _set_tokenizer_params parity
        if getattr(self._tokenizer, "padding_side", "right") != "right":
            self._tokenizer.padding_side = "right"
        if getattr(self._tokenizer, "pad_token_id", None) is None:
            self._tokenizer.pad_token_id = getattr(self._tokenizer, "eos_token_id", 0)
        self._ctx_len = int(ctx_len)
        self._prefill_seq_len = int(prefill_seq_len)
        self._vocab_size: Optional[int] = None

    # --- session probing helpers (runtime parity) ---
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

    # --- PREFILL: identical variable names & logic ---
    def run_prefill(
        self,
        prompt: str,
        generation_len: Optional[int] = None,
        prefill_logit_bs: int = 1,
        decode_batch_id: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray, int]:
        """
        Runs prefill for a given prompt and generation length (speculator).
        Returns: (outputs, position_ids, generation_len) from the last chunk.
        """

        # First pass (padding=True)
        inputs = self._tokenizer(prompt, return_tensors="np", padding=True)
        position_ids = inputs["attention_mask"].sum(1, keepdims=True)
        padded_len = inputs["input_ids"].shape[1]
        # ceil divide to chunk count
        num_chunks = -(padded_len // -self._prefill_seq_len)
        # Convert to a multiple of prefill_len
        padded_len = num_chunks * self._prefill_seq_len

        # Compute generation length parity
        max_gen_len = self._ctx_len - position_ids.max()
        generation_len = self._fetch_generation_len(generation_len, max_gen_len)

        # Preallocate logits buffer (parity with runtime)
        try:
            vocab_size = self._fetch_vocab_size()
            logits_out_placeholder = np.zeros((prefill_logit_bs, 1, vocab_size), dtype=np.float32)
            self._session.set_buffers({"logits": logits_out_placeholder})
        except Exception:
            pass

        # Second pass: padding="max_length" and build position_ids via np.where
        inputs = self._tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
        inputs["position_ids"] = np.where(
            inputs.pop("attention_mask"),
            np.arange(padded_len, dtype=np.int64),
            -1,
        )
        inputs.pop("token_type_ids", None)

        if decode_batch_id is not None:
            inputs["batch_index"] = decode_batch_id
        # NOTE: no TLM/LoRA extras in speculator path.

        # Chunk loop (names & slicing identical)
        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ].astype(np.int64, copy=False)
            chunk_inputs["position_ids"] = inputs["position_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ].astype(np.int64, copy=False)
            outputs = self._session.run(chunk_inputs)

        return outputs, position_ids, generation_len

    # --- helper to collect tensors for scoring from a prefill 'outputs' dict ---
    def collect_for_scoring(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract speculator tensors needed for scoring:
          - prefill_queries: np.ndarray [L,H,D]
          - past_keys: List[np.ndarray] len L, each [1,H_kv,S,D]
        Returns: {"prefill_queries": ..., "past_keys": ...}
        """
        if "prefill_queries" not in outputs:
            raise KeyError(
                "prefill_queries not found in outputs (ensure spec QPC was exported with this output)"
            )
        prefill_queries = outputs["prefill_queries"]
        past_keys: List[np.ndarray] = []
        idx = 0
        while True:
            key_name = f"past_key.{idx}_RetainedState"
            if key_name not in outputs:
                break
            past_keys.append(outputs[key_name])
            idx += 1
        if not past_keys:
            raise KeyError("No past_key.{i}_RetainedState outputs found in speculator outputs")
        return {"prefill_queries": prefill_queries, "past_keys": past_keys}


# --------------------- simple __main__ validator ---------------------
def main() -> None:
    import argparse
    import sys
    from QEfficient.utils import load_hf_tokenizer

    parser = argparse.ArgumentParser(
        description="Step 4.1: speculator prefill parity check (mirrors runtime prefill)."
    )
    parser.add_argument("--spec-qpc", required=True, help="Path to speculator QPC directory (…/qpc)")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B", help="HF tokenizer card for speculator")
    parser.add_argument("--prompt", default="Hello from spec prefill on AI100.", help="Prompt string")
    parser.add_argument("--prompt-len", type=int, default=16, help="prefill chunk length")
    parser.add_argument("--ctx-len", type=int, default=128, help="context length")
    parser.add_argument("--device-ids", default="[0]", help="Device IDs like [0] or [0,1]")
    parser.add_argument("--prefill-logit-bs", type=int, default=1, help="logits placeholder batch size")
    args = parser.parse_args()

    try:
        dev_ids = [int(x) for x in args.device_ids.strip("[] ").split(",") if x.strip() != ""]
    except Exception:
        print(f"[warn] Could not parse --device-ids={args.device_ids!r}, defaulting to [0]")
        dev_ids = [0]

    tok = load_hf_tokenizer(args.model_name)
    eng = SpecPrefillEngine(
        spec_qpc_path=args.spec_qpc,
        tokenizer=tok,
        ctx_len=int(args.ctx_len),
        prefill_seq_len=int(args.prompt_len),
        device_ids=dev_ids,
    )

    # 1) run prefill
    outputs, position_ids, generation_len = eng.run_prefill(
        args.prompt, generation_len=None, prefill_logit_bs=args.prefill_logit_bs
    )
    if "logits" not in outputs or "prefill_queries" not in outputs:
        print("[fail] required outputs missing:", list(outputs.keys()))
        sys.exit(2)

    # 2) collect tensors for scoring
    tensors = eng.collect_for_scoring(outputs)
    q = tensors["prefill_queries"]
    keys = tensors["past_keys"]

    print(f"[spec] prefill_queries: {q.shape}")
    print(f"[spec] past_keys[0]: {keys[0].shape}  layers: {len(keys)}")
    print(f"[spec] logits: {outputs['logits'].shape}")
    sys.exit(0)


if __name__ == "__main__":
    main()

