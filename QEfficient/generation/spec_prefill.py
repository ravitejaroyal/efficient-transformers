from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from QEfficient.generation.cloud_infer import QAICInferenceSession


@dataclass
class KeepConfig:
    strategy: str = "percentage"  # only "percentage" in this step
    percentage: float = 0.1  # 10%
    chunk: bool = True
    chunk_size: int = 32


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

    def _softmax_axis(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Numerically-stable softmax along `axis`."""
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        return ex / np.sum(ex, axis=axis, keepdims=True)

    def _avg_pool1d_same(self, x: np.ndarray, kernel: int) -> np.ndarray:
        """Moving-average along last axis with 'same' length via edge padding."""
        if kernel is None or kernel <= 1:
            return x
        pad = kernel // 2
        xpad = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="edge")
        csum = np.cumsum(xpad, axis=-1, dtype=np.float32)
        out = (csum[..., kernel:] - csum[..., :-kernel]) / float(kernel)
        return out

    def compute_importance(
        self,
        prefill_queries: np.ndarray,
        past_keys: List[np.ndarray],
        pool_kernel_size: Optional[int] = 13,
    ) -> np.ndarray:
        """Compute token-importance [S] for k=0."""
        q = prefill_queries.astype(np.float32, copy=False)
        k_list = []
        for k in past_keys:
            if k.ndim != 4:
                raise ValueError(f"past_key must be [1,H_kv,S,D], got {k.shape}")
            k_list.append(k.squeeze(0))
        K = np.stack(k_list, axis=0)

        L, H, D = q.shape
        _, H_kv, S, Dk = K.shape
        if Dk != D:
            raise ValueError(f"D mismatch: queries D={D} vs keys D={Dk}")

        if H != H_kv:
            if H % H_kv != 0:
                raise ValueError(f"H ({H}) must be multiple of H_kv ({H_kv})")
            rep = H // H_kv
            K = np.repeat(K, repeats=rep, axis=1)

        scale = 1.0 / math.sqrt(float(D))
        scores = np.einsum("lhd,lhsd->lhs", q, K, optimize=True) * scale
        attn = self._softmax_axis(scores, axis=-1)        # [L,H,S]
        attn = self._avg_pool1d_same(attn, kernel=pool_kernel_size)

        # Optional numerics check: softmax sums ~1 along last axis
        import os
        if os.getenv("QEFF_SPEC_ASSERT", ""):
            if not pool_kernel_size or pool_kernel_size <= 1:
                sums = np.sum(attn, axis=-1)
                assert np.allclose(sums, 1.0, atol=1e-3), (
                    f"softmax sums not ~1; max dev: {np.max(np.abs(sums - 1.0))}"
                )

        attn_lh = attn.reshape(L * H, S)
        importance = np.max(attn_lh, axis=0).astype(np.float32, copy=False)
        return importance

    def select_tokens(self, importance: np.ndarray, keep_cfg: KeepConfig) -> np.ndarray:
        """Keep indices per policy."""
        if keep_cfg.strategy != "percentage":
            raise ValueError(f"Unsupported keep strategy: {keep_cfg.strategy}")
        S = int(importance.shape[0])
        must_keep = np.array([S - 1], dtype=np.int64)

        if keep_cfg.chunk:
            cs = int(keep_cfg.chunk_size)
            if cs <= 0:
                raise ValueError("chunk_size must be > 0")
            n_chunks = (S + cs - 1) // cs
            chunk_scores = np.empty((n_chunks,), dtype=np.float32)
            for i in range(n_chunks):
                start = i * cs
                end = min((i + 1) * cs, S)
                chunk_scores[i] = float(np.mean(importance[start:end], dtype=np.float32))
            k_chunks = max(1, int(math.ceil(n_chunks * keep_cfg.percentage)))
            keep_chunk_idx = np.argpartition(-chunk_scores, k_chunks - 1)[:k_chunks]
            keep_chunk_idx.sort()
            kept_slices = [
                np.arange(i * cs, min((i + 1) * cs, S), dtype=np.int64)
                for i in keep_chunk_idx.tolist()
            ]
            kept = (
                np.concatenate(kept_slices, axis=0)
                if kept_slices
                else np.empty((0,), dtype=np.int64)
            )
        else:
            k = max(1, int(math.ceil(S * keep_cfg.percentage)))
            kept = np.argpartition(-importance, k - 1)[:k].astype(np.int64)
            kept.sort()

        kept = np.unique(np.concatenate([kept, must_keep], axis=0))
        return kept

    def prefill_and_score(
        self,
        prompt: str,
        *,
        pool_kernel_size: int = 13,
        keep_cfg: Optional[KeepConfig] = None,
    ) -> Dict[str, Any]:
        """
        Run spec prefill, compute importance and select keep indices.
        IMPORTANT: scoring is performed only over the true prompt length (S_orig),
        excluding any padded/capacity tail present in QPC buffers.
        """
        if keep_cfg is None:
            keep_cfg = KeepConfig(strategy="percentage", percentage=0.1, chunk=True, chunk_size=32)

        # Run prefill exactly as before; get position_ids from the first pass
        outputs, position_ids, _ = self.run_prefill(
            prompt, generation_len=None, prefill_logit_bs=1
        )

        # Extract tensors for scoring
        tensors = self.collect_for_scoring(outputs)
        q = tensors["prefill_queries"]          # [L,H,D]
        keys_raw = tensors["past_keys"]         # list of [1,H_kv,S_cap,D]
        if len(keys_raw) == 0:
            raise KeyError("No past_key.{i}_RetainedState outputs to score against")

        # True prompt length (number of real tokens) from first-pass position_ids
        S_orig = int(position_ids.max())        # NOT padded_len or ctx capacity
        S_cap  = int(keys_raw[0].shape[2])

        # Trim keys to true prompt length to ignore pad/capacity tail
        keys = keys_raw
        if S_cap > S_orig:
            keys = [k[:, :, :S_orig, :].copy() for k in keys]   # keep [1,H_kv,S_orig,D]

        # Compute importance and defensively ensure len == S_orig
        imp = self.compute_importance(q, keys, pool_kernel_size=pool_kernel_size)  # [S_trim]
        if imp.shape[0] > S_orig:
            imp = imp[:S_orig]
        # Assert we didn’t leak padded positions into importance
        assert imp.shape[0] == S_orig, f"importance length {imp.shape[0]} != S_orig {S_orig}"
        assert np.isfinite(imp).all(), "importance contains non-finite values"
        assert float(imp.sum()) > 0.0, "importance sums to zero"

        # Select keep indices over [0..S_orig-1]
        keep_idx = self.select_tokens(imp, keep_cfg)

        # Simple anti-pad invariants
        if keep_idx.size > 0:
            assert keep_idx.dtype == np.int64, f"keep_idx dtype must be int64, got {keep_idx.dtype}"
            assert keep_idx.min() >= 0, f"negative keep index found: {keep_idx.min()}"
            assert keep_idx.max() < S_orig, (
                f"keep_idx contains padded indices >= S_orig={S_orig}: {keep_idx.max()}"
            )
            assert keep_idx[-1] == S_orig - 1, (
                f"last real token must be kept (expected {S_orig-1}, got {keep_idx[-1]})"
            )

        # Optional debug: set QEFF_SPEC_DEBUG=1 to print one-liner summary
        import os
        if os.getenv("QEFF_SPEC_DEBUG", ""):
            kept_pct = (100.0 * len(keep_idx) / S_orig) if S_orig > 0 else 0.0
            print(f"[spec:score] S_orig={S_orig} S_cap={S_cap} imp_len={imp.shape[0]} kept={len(keep_idx)} ({kept_pct:.1f}%)")

        return {
            "importance": imp,         # length S_orig
            "keep_idx": keep_idx,      # sorted unique, includes S_orig-1
            "S": S_orig,               # true prompt length (non-pad count)
            "shapes": {
                "prefill_queries": q.shape,
                "first_key": keys_raw[0].shape if len(keys_raw) else None,  # report pre-trim shape for visibility
            },
        }


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

