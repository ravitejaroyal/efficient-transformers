from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.generation.base_infer import write_io_files
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.logging_utils import logger


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
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        qpc_path: str,
        full_batch_size: Optional[int] = None,
        ctx_len: Optional[int] = None,
        device_id: Optional[List[int]] = None,
        enable_debug_logs: bool = False,
        write_io_dir: Optional[str] = None,
        is_tlm: Optional[int] = None,
    ) -> None:
        self._ctx_len = ctx_len
        self._write_io_dir = write_io_dir
        self.is_tlm = is_tlm

        # Load QPC
        self._session = QAICInferenceSession(
            qpc_path, device_id, enable_debug_logs=enable_debug_logs
        )

        # Fetch variables from the QPC
        self._vocab_size = self._fetch_vocab_size()
        self.batch_size, self._prefill_seq_len = (
            self._fetch_batch_size_prefill_seq_len()
        )
        self._decode_seq_len = self._fetch_decode_seq_len()
        self.full_batch_size = (
            full_batch_size if full_batch_size else self._fetch_full_batch_size()
        )

        # Initialize storage variables
        self.batch_index = None
        self._prompt_to_lora_id_mapping_prefill = None
        self._prompt_to_lora_id_mapping_decode = None
        self.generated_ids = None
        self.decode_input_ids = None
        self.decode_pos_ids = None
        self.generation_len = None

        self.tokenizer = tokenizer
        self._set_tokenizer_params()

        # Speculative prefill relies on past_key.* outputs for scoring, so only
        # skip past inputs and past_value.* outputs. This preserves the
        # retained key states while dropping value caches.
        past_inputs = [n for n in self._session.input_names if n.startswith("past_")]
        past_val_outs = [
            n for n in self._session.output_names if n.startswith("past_value.")
        ]
        self._session.skip_buffers(past_inputs + past_val_outs)

    def _set_tokenizer_params(self):
        """
        Sets the tokenizer parameters for the model.
        """
        if self.tokenizer.padding_side != "right":
            logger.warning(
                "Please use padding_side='right' while initializing the tokenizer"
            )
            self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # --- session probing helpers (runtime parity) ---
    def _fetch_vocab_size(self):
        """Fetches and caches the vocabulary size from the session's allowed shapes."""
        if getattr(self, "_vocab_size", None) is not None:
            return self._vocab_size
        if self._session.allowed_shapes:
            self._vocab_size = [
                x[self._session.binding_index_map["logits"]]
                for x in self._session.allowed_shapes
            ][0][1][2]
        else:
            self._vocab_size = self._session.bindings[
                self._session.binding_index_map["logits"]
            ].dims[2]
        return self._vocab_size

    def _fetch_batch_size_prefill_seq_len(self):
        """Fetches batch size and prefill sequence length from the session."""
        if self._session.allowed_shapes:
            batch_size = max(
                [
                    x[self._session.binding_index_map["input_ids"]][1][0]
                    for x in self._session.allowed_shapes
                ]
            )
            prefill_seq_len = max(
                [
                    x[self._session.binding_index_map["input_ids"]][1][1]
                    for x in self._session.allowed_shapes
                ]
            )
        else:
            batch_size, prefill_seq_len = self._session.bindings[
                self._session.binding_index_map["input_ids"]
            ].dims
        return batch_size, prefill_seq_len

    def _fetch_decode_seq_len(self):
        """Fetches decode sequence length from the session."""
        decode_seq_len = None
        if self._session.allowed_shapes:
            decode_seq_len = min(
                [
                    x[self._session.binding_index_map["input_ids"]][1][1]
                    for x in self._session.allowed_shapes
                ]
            )
        return decode_seq_len

    def _fetch_full_batch_size(self):
        """Fetches full batch size from the session."""
        full_batch_size = None
        if "batch_index" in self._session.binding_index_map:
            if self._session.allowed_shapes:
                full_batch_size, _ = [
                    x[self._session.binding_index_map["batch_index"]][1][0]
                    for x in self._session.allowed_shapes
                ]
            else:
                full_batch_size, _ = self._session.bindings[
                    self._session.binding_index_map["batch_index"]
                ].dims
        return full_batch_size

    def _fetch_generation_len(self, generation_len, max_gen_len):
        """Fetches the generation length for the model."""
        if generation_len is None:
            if self._ctx_len is None:
                raise ValueError("At least one of ctx_len or generation_len is needed")
            generation_len = max_gen_len
        elif generation_len > max_gen_len:
            logger.warning(
                "Passed generation_len is greater than allowed length. "
                "Make sure this model supports sliding window, such as Mistral"
            )
        if generation_len <= 0:
            raise ValueError("generation length should be greater than zero")
        return generation_len

    # --- PREFILL: identical variable names & logic ---
    def run_prefill(
        self, prompt, generation_len, prefill_logit_bs=1, decode_batch_id=None
    ):
        """Runs prefill for a given prompt and generation length."""
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        position_ids = inputs["attention_mask"].sum(1, keepdims=True)
        padded_len = inputs["input_ids"].shape[1]
        num_chunks = -(padded_len // -self._prefill_seq_len)
        padded_len = num_chunks * self._prefill_seq_len

        max_gen_len = self._ctx_len - position_ids.max()
        generation_len = self._fetch_generation_len(generation_len, max_gen_len)

        logits_out_placeholder = np.zeros(
            (prefill_logit_bs, 1, self._vocab_size), dtype=np.float32
        )
        self._session.set_buffers({"logits": logits_out_placeholder})

        inputs = self.tokenizer(
            prompt, return_tensors="np", padding="max_length", max_length=padded_len
        )
        inputs["position_ids"] = np.where(
            inputs.pop("attention_mask"), np.arange(padded_len), -1
        )
        inputs.pop("token_type_ids", None)

        if decode_batch_id is not None:
            inputs["batch_index"] = decode_batch_id
        if self.is_tlm:
            inputs["num_logits_to_keep"] = np.zeros((1, 1))

        if self._prompt_to_lora_id_mapping_prefill:
            if self.full_batch_size:
                inputs["lora_ids"] = np.array(
                    self._prompt_to_lora_id_mapping_prefill.popleft(), dtype=np.int64
                ).reshape(1, 1)
            else:
                batch_lora_ids = [
                    self._prompt_to_lora_id_mapping_prefill.popleft()
                    for i in range(self.batch_size)
                ]
                inputs["lora_ids"] = np.array(batch_lora_ids, dtype=np.int64).reshape(
                    self.batch_size, 1
                )

        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]
            chunk_inputs["position_ids"] = inputs["position_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]
            outputs = self._session.run(chunk_inputs)
            if self._write_io_dir is not None:
                write_io_files(
                    inputs,
                    outputs,
                    self._write_io_dir,
                    "prefill",
                    "aic_batch_io",
                    True,
                    False,
                )

        last_logits = outputs["logits"][0, -1]
        token_id = int(last_logits.argmax())
        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        import os

        if os.getenv("QEFF_SPEC_DEBUG", ""):
            print(f"[base:prefill] final token: {token_text!r}", flush=True)
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
            raise KeyError(
                "No past_key.{i}_RetainedState outputs found in speculator outputs"
            )
        return {"prefill_queries": prefill_queries, "past_keys": past_keys}

    def _softmax_axis(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Numerically-stable softmax along `axis`."""
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        return ex / np.sum(ex, axis=axis, keepdims=True)

    def _avg_pool1d_same(self, x: np.ndarray, kernel: int) -> np.ndarray:
        """
        Moving-average along last axis with 'same' output length.
        Works for odd or even kernels by asymmetric edge padding.
        x: [..., S] -> returns [..., S]
        """
        if kernel is None or kernel <= 1:
            return x

        # Asymmetric padding ensures output length matches input length
        pad_left = kernel // 2
        pad_right = kernel - 1 - pad_left  # total padding = kernel - 1

        xpad = np.pad(
            x,
            [(0, 0)] * (x.ndim - 1) + [(pad_left, pad_right)],
            mode="edge",
        )
        # Prepend zero for sliding window via cumsum
        cs = np.cumsum(xpad, axis=-1, dtype=np.float32)
        zero = np.zeros_like(xpad[..., :1], dtype=np.float32)
        cs = np.concatenate([zero, cs], axis=-1)

        out = (cs[..., kernel:] - cs[..., :-kernel]) / float(kernel)
        return out

    def compute_importance(
        self,
        prefill_queries: np.ndarray,  # [L,H,D]
        past_keys: List[np.ndarray],  # len L, each [1,H_kv,S,D]
        pool_kernel_size: Optional[int] = 13,
    ) -> np.ndarray:
        """Compute token-importance [S] for k=0."""
        # ---- sanitize inputs (defensive) ----
        # Convert to float32 and replace NaN/+Inf/-Inf with 0.0 so softmax is well-posed
        q = np.nan_to_num(
            prefill_queries.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )  # [L,H,D]

        k_list = []
        for k in past_keys:
            if k.ndim != 4:
                raise ValueError(f"past_key must be [1,H_kv,S,D], got {k.shape}")
            k_list.append(k.squeeze(0))  # [H_kv,S,D]
        K = np.stack(k_list, axis=0).astype(np.float32, copy=False)  # [L,H_kv,S,D]
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

        L, H, D = q.shape
        _, H_kv, S, Dk = K.shape
        if Dk != D:
            raise ValueError(f"D mismatch: queries D={D} vs keys D={Dk}")

        if H != H_kv:
            if H % H_kv != 0:
                raise ValueError(f"H ({H}) must be multiple of H_kv ({H_kv})")
            rep = H // H_kv
            K = np.repeat(K, repeats=rep, axis=1)  # [L,H,S,D]

        scale = 1.0 / math.sqrt(float(D))
        scores = np.einsum("lhd,lhsd->lhs", q, K, optimize=True) * scale
        attn = self._softmax_axis(scores, axis=-1)  # [L,H,S]
        attn = self._avg_pool1d_same(attn, kernel=pool_kernel_size)

        assert (
            attn.shape[-1] == S
        ), f"avg_pool1d_same produced length {attn.shape[-1]} != expected {S}"

        # Optional numerics check: softmax sums ~1 along last axis
        import os

        if os.getenv("QEFF_SPEC_ASSERT", ""):
            if not pool_kernel_size or pool_kernel_size <= 1:
                sums = np.sum(attn, axis=-1)
                assert np.allclose(
                    sums, 1.0, atol=1e-3
                ), f"softmax sums not ~1; max dev: {np.max(np.abs(sums - 1.0))}"

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
                chunk_scores[i] = float(
                    np.mean(importance[start:end], dtype=np.float32)
                )
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
        IMPORTANT: scoring is performed only over the true prompt length (orig_seq_len),
        excluding any padded/capacity tail present in QPC buffers.
        """
        if keep_cfg is None:
            keep_cfg = KeepConfig(
                strategy="percentage", percentage=0.1, chunk=True, chunk_size=32
            )

        # Run prefill exactly as before; get position_ids from the first pass
        outputs, position_ids, _ = self.run_prefill(
            prompt, generation_len=None, prefill_logit_bs=1
        )

        # Extract tensors for scoring
        tensors = self.collect_for_scoring(outputs)
        q = tensors["prefill_queries"]  # [L,H,D]
        keys_raw = tensors["past_keys"]  # list of [1,H_kv,capacity_seq_len,D]
        if len(keys_raw) == 0:
            raise KeyError("No past_key.{i}_RetainedState outputs to score against")

        # True prompt length (number of real tokens) from first-pass position_ids
        orig_seq_len = int(position_ids.max())  # NOT padded_len or ctx capacity
        capacity_seq_len = int(keys_raw[0].shape[2])

        # Trim keys to true prompt length to ignore pad/capacity tail
        keys = keys_raw
        if capacity_seq_len > orig_seq_len:
            keys = [
                k[:, :, :orig_seq_len, :].copy() for k in keys
            ]  # keep [1,H_kv,orig_seq_len,D]

        # Compute importance and defensively ensure len == orig_seq_len
        imp = self.compute_importance(
            q, keys, pool_kernel_size=pool_kernel_size
        )  # [S_trim]
        if imp.shape[0] > orig_seq_len:
            imp = imp[:orig_seq_len]
        # Assert we didn’t leak padded positions into importance
        assert (
            imp.shape[0] == orig_seq_len
        ), f"importance length {imp.shape[0]} != orig_seq_len {orig_seq_len}"
        assert np.isfinite(imp).all(), "importance contains non-finite values"
        assert float(imp.sum()) > 0.0, "importance sums to zero"

        # Select keep indices over [0..orig_seq_len-1]
        keep_idx = self.select_tokens(imp, keep_cfg)

        # Simple anti-pad invariants
        if keep_idx.size > 0:
            assert (
                keep_idx.dtype == np.int64
            ), f"keep_idx dtype must be int64, got {keep_idx.dtype}"
            assert keep_idx.min() >= 0, f"negative keep index found: {keep_idx.min()}"
            assert (
                keep_idx.max() < orig_seq_len
            ), f"keep_idx contains padded indices >= orig_seq_len={orig_seq_len}: {keep_idx.max()}"
            assert (
                keep_idx[-1] == orig_seq_len - 1
            ), f"last real token must be kept (expected {orig_seq_len-1}, got {keep_idx[-1]})"

        # Optional debug: set QEFF_SPEC_DEBUG=1 to print one-liner summary
        import os

        if os.getenv("QEFF_SPEC_DEBUG", ""):
            kept_pct = (
                (100.0 * len(keep_idx) / orig_seq_len) if orig_seq_len > 0 else 0.0
            )
            print(
                f"[spec:score] orig_seq_len={orig_seq_len} capacity_seq_len={capacity_seq_len} "
                f"imp_len={imp.shape[0]} kept={len(keep_idx)} ({kept_pct:.1f}%)"
            )

        return {
            "importance": imp,  # length orig_seq_len
            "keep_idx": keep_idx,  # sorted unique, includes orig_seq_len-1
            "orig_seq_len": orig_seq_len,  # true prompt length (non-pad count)
            "shapes": {
                "prefill_queries": q.shape,
                "first_key": (
                    keys_raw[0].shape if len(keys_raw) else None
                ),  # report pre-trim shape for visibility
            },
        }

    def prune_and_base_prefill(
        self,
        base_engine,
        prompt: str,
        *,
        pool_kernel_size: int = 13,
        keep_cfg: Optional[KeepConfig] = None,
        prefill_logit_bs: int = 1,
    ) -> Dict[str, Any]:
        """
        Step 4.3: spec prefill+score -> build pruned ids/pos -> run base prefill
        (baseline and pruned) with simple timings.
        """
        res = self.prefill_and_score(
            prompt,
            pool_kernel_size=pool_kernel_size,
            keep_cfg=keep_cfg,
        )
        keep_idx = res["keep_idx"]
        orig_seq_len = res["orig_seq_len"]
        enc = self.tokenizer(prompt, return_tensors="np")
        final_ids = enc["input_ids"][:, keep_idx].astype(np.int64)
        final_pos = keep_idx.reshape(1, -1).astype(np.int64)

        # ---- debug: print pruned tokens that will be fed to the base (IDs & tokens) ----
        try:
            import os

            if os.getenv("QEFF_SPEC_DEBUG", ""):
                ids_list = final_ids[0].tolist()
                tok_list = self.tokenizer.convert_ids_to_tokens(ids_list)
                k_preview = (
                    keep_idx[:32].tolist()
                    if hasattr(keep_idx, "tolist")
                    else list(keep_idx)[:32]
                )
                t_preview = tok_list[:32]
                print(
                    f"[spec:pruned] kept={len(ids_list)}/{orig_seq_len} keep_idx[:32]={k_preview}"
                )
                print(f"[spec:pruned] tokens[:32]={t_preview}")
                txt_preview = self.tokenizer.decode(ids_list, skip_special_tokens=True)
                print(f"[spec:pruned] text_preview={txt_preview[:120]!r}")
        except Exception:
            pass

        t0 = time.perf_counter()
        base_engine.run_prefill(
            prompt, generation_len=None, prefill_logit_bs=prefill_logit_bs
        )
        ttft_baseline_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        _, _, padded_len, num_chunks = base_engine.prefill_from_ids(
            final_ids, final_pos, prefill_logit_bs=prefill_logit_bs
        )
        ttft_pruned_s = time.perf_counter() - t1

        print(
            f"[4.3] orig_seq_len={orig_seq_len} kept={keep_idx.size} "
            f"TTFT baseline={ttft_baseline_s*1000:.1f}ms pruned={ttft_pruned_s*1000:.1f}ms"
        )

        return {
            "orig_seq_len": orig_seq_len,
            "kept": int(keep_idx.size),
            "keep_idx": keep_idx,
            "ttft_baseline_s": ttft_baseline_s,
            "ttft_pruned_s": ttft_pruned_s,
            "padded_len_pruned": padded_len,
            "num_chunks_pruned": num_chunks,
        }


# --------------------- simple __main__ validator ---------------------
def main() -> None:
    import argparse

    from QEfficient.utils import load_hf_tokenizer

    parser = argparse.ArgumentParser(
        description="Step 4.1: speculator prefill parity check (mirrors runtime prefill)."
    )
    parser.add_argument(
        "--spec-qpc", required=True, help="Path to speculator QPC directory (…/qpc)"
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B",
        help="HF tokenizer card for speculator",
    )
    parser.add_argument(
        "--prompt", default="Hello from spec prefill on AI100.", help="Prompt string"
    )
    parser.add_argument(
        "--prompt-len", type=int, default=16, help="prefill chunk length"
    )
    parser.add_argument("--ctx-len", type=int, default=128, help="context length")
    parser.add_argument(
        "--device-ids", default="[0]", help="Device IDs like [0] or [0,1]"
    )
    parser.add_argument(
        "--prefill-logit-bs", type=int, default=1, help="logits placeholder batch size"
    )
    args = parser.parse_args()

    try:
        dev_ids = [
            int(x) for x in args.device_ids.strip("[] ").split(",") if x.strip() != ""
        ]
    except Exception:
        print(
            f"[warn] Could not parse --device-ids={args.device_ids!r}, defaulting to [0]"
        )
        dev_ids = [0]

    tok = load_hf_tokenizer(args.model_name)
    eng = SpecPrefillEngine(
        spec_qpc_path=args.spec_qpc,
        tokenizer=tok,
        ctx_len=int(args.ctx_len),
        prefill_seq_len=int(args.prompt_len),
        device_ids=dev_ids,
    )

    # Build keep policy (use existing KeepConfig; do not redefine it)
    keep_cfg = KeepConfig(
        strategy="percentage",
        percentage=0.1,  # keep 10%
        chunk=True,
        chunk_size=32,
    )
    pool_kernel_size = 13

    # Step 4.2: prefill + score + select (handles padding/capacity inside)
    res = eng.prefill_and_score(
        args.prompt,
        pool_kernel_size=pool_kernel_size,
        keep_cfg=keep_cfg,
    )

    # Minimal anti-padding invariants
    orig_seq_len = res["orig_seq_len"]
    imp = res["importance"]
    keep_idx = res["keep_idx"]
    assert (
        imp.shape[0] == orig_seq_len
    ), f"importance len {imp.shape[0]} != orig_seq_len {orig_seq_len}"
    if keep_idx.size > 0:
        assert (
            keep_idx.dtype == np.int64
        ), f"keep_idx dtype {keep_idx.dtype} != np.int64"
        assert keep_idx.min() >= 0, f"negative keep index {keep_idx.min()}"
        assert (
            keep_idx.max() < orig_seq_len
        ), f"padded index detected: {keep_idx.max()} >= orig_seq_len {orig_seq_len}"
        assert (
            keep_idx[-1] == orig_seq_len - 1
        ), f"must keep last real token {orig_seq_len-1}; got {keep_idx[-1]}"

    # One-line summary (quiet by default)
    kept_pct = (100.0 * len(keep_idx) / orig_seq_len) if orig_seq_len > 0 else 0.0
    print(
        f"[score] orig_seq_len={orig_seq_len} kept={len(keep_idx)} ({kept_pct:.1f}%) "
        f"prefill_queries={res['shapes']['prefill_queries']} "
        f"first_key={res['shapes']['first_key']} "
        f"last_kept={keep_idx[-1] if keep_idx.size else 'NA'}"
    )


if __name__ == "__main__":
    main()
